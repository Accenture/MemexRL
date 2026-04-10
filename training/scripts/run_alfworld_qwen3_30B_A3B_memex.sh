#!/bin/bash
# =============================================================================
# Memex + Slime: ALFWorld RL Training — Qwen3-30B-A3B (MoE, from scratch)
#
# 30B-A3B: 128 experts, top-8 routing, 3B active params per token.
# Stronger instruction following → better compress/retrieve behavior.
# =============================================================================

pkill -9 sglang 2>/dev/null; sleep 1
ray stop --force 2>/dev/null; pkill -9 ray 2>/dev/null; sleep 2

set -ex

# =============================================================================
# Memex Agent Configuration
# =============================================================================

MEMEX_ENV_TYPE="alfworld"
MEMEX_COMPRESSION_MODE="${COMPRESSION_MODE:-lossless_db}"
MEMEX_TOOL_CALL_FORMAT="qwen"
MEMEX_MAX_STEPS="${MAX_STEPS:-50}"
MEMEX_CONTEXT_THRESHOLD="${CONTEXT_THRESHOLD:-8000}"
MEMEX_AUTO_COMPRESS="true"
MEMEX_DISABLE_RETRIEVE="false"

ALFWORLD_HIDE_ADMISSIBLE_COMMANDS="${HIDE_ADMISSIBLE_COMMANDS:-true}"
ALFWORLD_HIDE_INITIAL_OBS="${HIDE_INITIAL_OBS:-true}"
ALFWORLD_LIMIT_LOOK="${LIMIT_LOOK:-true}"
MEMEX_MAX_SUMMARY_TOKENS="${MAX_SUMMARY_TOKENS:-300}"

MEMEX_MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-32000}"
MEMEX_PARALLEL_ENV="${PARALLEL_ENV:-true}"

MEMEX_REWARD_SHAPER_ENABLE="${REWARD_SHAPER_ENABLE:-true}"
MEMEX_LAMBDA_CTX="${LAMBDA_CTX:-1}"
MEMEX_LAMBDA_RED="${LAMBDA_RED:-0.05}"
MEMEX_LAMBDA_FORMAT="${LAMBDA_FORMAT:-1}"

# =============================================================================
# Model Configuration — Qwen3-30B-A3B (MoE, INT4)
# =============================================================================

# Thinking-2507 uses rotary_base=1e7 (not default 1e6)
MODEL_ARGS_ROTARY_BASE=10000000
source "${SLIME_ROOT:-/workspace/slime}/scripts/models/qwen3-30B-A3B.sh"

MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to your HF checkpoint, e.g. /models/Qwen3-30B-A3B-int4}"
PRECISION_MODE="${PRECISION_MODE:-int4}"

# =============================================================================
# Slime Training Configuration
# =============================================================================

CKPT_ARGS=(
    --hf-checkpoint "${MODEL_PATH}"
    --ref-load "${MODEL_PATH}_torch_dist/"
    --load "${MODEL_PATH}_slime_memex/"
    --save "${MODEL_PATH}_slime_memex/"
    --save-interval 20
)

DATA_DIR="${DATA_DIR:?Set DATA_DIR to converted JSONL data path}"
ROLLOUT_ARGS=(
    --prompt-data "${DATA_DIR}/alfworld_train.jsonl"
    --input-key prompt
    --rollout-shuffle
    --num-rollout "${NUM_ROLLOUT:-3000}"
    --rollout-batch-size "${ROLLOUT_BATCH_SIZE:-32}"
    --n-samples-per-prompt "${N_SAMPLES:-8}"
    --rollout-max-response-len "${MEMEX_MAX_CONTEXT_LEN}"
    --rollout-max-context-len "${MEMEX_MAX_CONTEXT_LEN}"
    --rollout-temperature "${ROLLOUT_TEMPERATURE:-1.0}"
    --global-batch-size "${GLOBAL_BATCH_SIZE:-128}"
    --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
    --balance-data
)

EVAL_ARGS=(
    --eval-interval "${EVAL_INTERVAL:-5}"
    --eval-prompt-data alfworld-test "${DATA_DIR}/alfworld_test.jsonl"
    --n-samples-per-eval-prompt 1
    --eval-max-response-len "${MEMEX_MAX_CONTEXT_LEN}"
    --eval-top-k 1
)

GRPO_ARGS=(
    --advantage-estimator grpo
    --use-kl-loss
    --kl-loss-coef "${KL_COEF:-0.001}"
    --kl-loss-type low_var_kl
    --entropy-coef "${ENTROPY_COEF:-0.002}"
    --eps-clip "${EPS_CLIP:-0.2}"
    --eps-clip-high "${EPS_CLIP_HIGH:-0.28}"
)

USE_TIS="${USE_TIS:-true}"
TIS_ARGS=()
if [ "${USE_TIS}" = "true" ]; then
    TIS_ARGS=(
        --use-tis
        --tis-clip "${TIS_CLIP:-2.0}"
        --tis-clip-low "${TIS_CLIP_LOW:-0}"
    )
fi

OPTIMIZER_ARGS=(
    --optimizer adam
    --lr "${LR:-5e-6}"
    --lr-decay-style constant
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.98
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
)

# MoE parallelism: TP=4, EP=8 (official config from slime repo)
NUM_GPUS="${NUM_GPUS:-8}"
PERF_ARGS=(
    --accumulate-allreduce-grads-in-fp32
    --attention-softmax-in-fp32
    --attention-backend flash
    --attention-dropout 0.0
    --hidden-dropout 0.0
    --tensor-model-parallel-size "${TP_SIZE:-4}"
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size "${EP_SIZE:-8}"
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
    --recompute-num-layers 1
    --use-dynamic-batch-size
    --max-tokens-per-gpu "${MAX_TOKENS_PER_GPU:-8192}"
)

SGLANG_ARGS=(
    --rollout-num-gpus-per-engine 1
    --sglang-mem-fraction-static 0.70
    --sglang-cuda-graph-bs 1 2 4 8 $(seq 16 8 256)
    --use-slime-router
)

CUSTOM_ARGS=(
    --custom-generate-function-path generate_with_memex.generate
)

WANDB_ARGS=(
    --use-wandb
    --wandb-project "${PROJECT_NAME:-memex-alfworld}"
    --wandb-group "${EXPERIMENT_NAME:-alfworld-grpo-qwen3-30b-a3b-memex}"
    --wandb-key "${WANDB_KEY:?Set WANDB_KEY env var}"
)

# =============================================================================
# Path Configuration
# =============================================================================

MEMEX_ROOT="${MEMEX_ROOT:-/workspace/Memex}"
SLIME_ROOT="${SLIME_ROOT:-/workspace/slime}"
MEMEX_SLIME_ROOT="${MEMEX_SLIME_ROOT:-/workspace/Memex/training}"

# =============================================================================
# NVLink Detection
# =============================================================================

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi

echo "=========================================="
echo "Memex + Slime: ALFWorld RL — Qwen3-30B-A3B"
echo "=========================================="
echo "Model: ${MODEL_PATH}"
echo "Precision: ${PRECISION_MODE}"
echo "Memory: ${MEMEX_COMPRESSION_MODE}"
echo "Context Threshold: ${MEMEX_CONTEXT_THRESHOLD}"
echo "EP=${EP_SIZE:-8}, TP=${TP_SIZE:-1}"
echo "Max Tokens Per GPU: ${MAX_TOKENS_PER_GPU:-16384}"
echo "=========================================="

# =============================================================================
# Launch
# =============================================================================

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

ray start --head \
    --node-ip-address "${MASTER_ADDR}" \
    --num-gpus "${NUM_GPUS}" \
    --disable-usage-stats \
    --dashboard-host=0.0.0.0 \
    --temp-dir ${RAY_TMPDIR:-/tmp/ray}

PRECISION_ENV_VARS=""
if [ "${PRECISION_MODE}" = "int4" ]; then
    PRECISION_ENV_VARS="$(cat <<'ENVEOF'
    "OPEN_TRAINING_INT4_FAKE_QAT_FLAG": "1",
    "OPEN_TRAINING_INT4_GROUP_SIZE": "128",
ENVEOF
)"
fi

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_ROOT:-/root/Megatron-LM}:${MEMEX_ROOT}:${MEMEX_SLIME_ROOT}:${SLIME_ROOT}\",
    \"ALFWORLD_DATA\": \"${ALFWORLD_DATA:?Set ALFWORLD_DATA}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    ${PRECISION_ENV_VARS}
    \"MEMEX_ENV_TYPE\": \"${MEMEX_ENV_TYPE}\",
    \"MEMEX_COMPRESSION_MODE\": \"${MEMEX_COMPRESSION_MODE}\",
    \"MEMEX_TOOL_CALL_FORMAT\": \"${MEMEX_TOOL_CALL_FORMAT}\",
    \"MEMEX_MAX_STEPS\": \"${MEMEX_MAX_STEPS}\",
    \"MEMEX_CONTEXT_THRESHOLD\": \"${MEMEX_CONTEXT_THRESHOLD}\",
    \"MEMEX_AUTO_COMPRESS\": \"${MEMEX_AUTO_COMPRESS}\",
    \"MEMEX_DISABLE_RETRIEVE\": \"${MEMEX_DISABLE_RETRIEVE}\",
    \"MEMEX_REWARD_SHAPER_ENABLE\": \"${MEMEX_REWARD_SHAPER_ENABLE}\",
    \"MEMEX_LAMBDA_CTX\": \"${MEMEX_LAMBDA_CTX}\",
    \"MEMEX_LAMBDA_RED\": \"${MEMEX_LAMBDA_RED}\",
    \"MEMEX_LAMBDA_FORMAT\": \"${MEMEX_LAMBDA_FORMAT}\",
    \"MEMEX_MAX_CONTEXT_LEN\": \"${MEMEX_MAX_CONTEXT_LEN}\",
    \"MEMEX_PARALLEL_ENV\": \"${MEMEX_PARALLEL_ENV}\",
    \"ALFWORLD_HIDE_ADMISSIBLE_COMMANDS\": \"${ALFWORLD_HIDE_ADMISSIBLE_COMMANDS}\",
    \"ALFWORLD_HIDE_INITIAL_OBS\": \"${ALFWORLD_HIDE_INITIAL_OBS}\",
    \"ALFWORLD_LIMIT_LOOK\": \"${ALFWORLD_LIMIT_LOOK}\",
    \"MEMEX_MAX_SUMMARY_TOKENS\": \"${MEMEX_MAX_SUMMARY_TOKENS}\"
  }
}"

ray job submit \
    --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 "${SLIME_ROOT}/train.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "${NUM_GPUS}" \
    --rollout-num-gpus "${NUM_GPUS}" \
    --colocate \
    "${MODEL_ARGS[@]}" \
    "${CKPT_ARGS[@]}" \
    "${ROLLOUT_ARGS[@]}" \
    "${EVAL_ARGS[@]}" \
    "${GRPO_ARGS[@]}" \
    "${TIS_ARGS[@]}" \
    "${OPTIMIZER_ARGS[@]}" \
    "${PERF_ARGS[@]}" \
    "${SGLANG_ARGS[@]}" \
    "${CUSTOM_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
