# Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory

This is the code release for the paper **[Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory](https://arxiv.org/abs/2603.04257)**.

```

## Prerequisites

Training runs inside the **[Slime Docker container](https://github.com/THUDM/slime)** (`slimerl/slime:latest`), which bundles all heavy dependencies (Megatron-LM, SGLang, Ray, CUDA, NCCL). You do **not** install Slime inside this repo — it lives in the Docker image.

**Hardware:** 8× NVIDIA GPUs (H100 recommended)

## Quick Start

All commands below assume you are on a machine with 8 GPUs and Docker installed.

### Step 1: Clone repos and pull Docker image

```bash
# On your host machine:

# Clone this repo
git clone https://github.com/accenture/memexrl.git

# Clone Slime — the RL training framework. The training script sources model
# architecture configs from this repo (e.g. scripts/models/qwen3-30B-A3B.sh).
# Pin to the tested commit to avoid breaking changes from upstream updates.
git clone https://github.com/THUDM/slime.git
cd slime && git checkout 67a21a1b && cd ..

# Download the base model from HuggingFace (BF16 version)
# You need a directory to store models. We'll use ~/models as an example.
mkdir -p ~/models
huggingface-cli download Qwen/Qwen3-30B-A3B-Thinking-2507 --local-dir ~/models/Qwen3-30B-A3B-Thinking-2507

# Pull Slime Docker image — pinned to the tested digest to avoid incompatibilities.
docker pull slimerl/slime@sha256:c464f461e78c4256cc44535b000abf01b60ab7097d5badc6483591fb18525a39
```

### Step 2: Start the training container

The `docker run` command below mounts three host directories into the container
at fixed paths. **The training script expects these exact container paths** — you
do not need to edit the script, just make sure the `-v` mounts are correct.

```bash
# Replace the left side of each -v with YOUR actual host paths:
docker run -d --name slime-train \
    --gpus all \
    --shm-size 16g \
    -v $(pwd)/Memex:/workspace/Memex \
    -v $(pwd)/slime:/workspace/slime \
    -v ~/models:/workspace/models \
    slimerl/slime@sha256:c464f461e78c4256cc44535b000abf01b60ab7097d5badc6483591fb18525a39 \
    sleep infinity
```

After this, inside the container:

| Container path | What's there | Used by |
|---------------|-------------|---------|
| `/workspace/Memex` | This repo (Memex source code) | `MEMEX_ROOT` in training script |
| `/workspace/slime` | Slime repo (trainer + model configs) | `SLIME_ROOT` in training script |
| `/workspace/models` | HF model weights | `MODEL_PATH` in training script |

### Step 3: Install dependencies inside the container

```bash
docker exec -it slime-train bash

# All commands below run inside the container.

# Install ALFWorld environment + download game data
pip install alfworld
alfworld-download -f
```

### Step 4: Prepare training data

```bash
cd /workspace/Memex
PYTHONPATH=.:training ALFWORLD_DATA=$HOME/.alfworld \
    python training/convert_data.py --output-dir /workspace/data/alfworld
```

This reads ALFWorld game files and produces two JSONL files the training script expects:
- `/workspace/data/alfworld/alfworld_train.jsonl`
- `/workspace/data/alfworld/alfworld_test.jsonl`

### Step 5: Prepare model checkpoints

The training script needs **two** pre-built checkpoint formats. A third one
(`_slime_memex/`) is auto-created by Slime on the first training run.

**Why two formats?** Slime uses Megatron-LM for training (needs `torch_dist`
format as the reference model for KL divergence) and SGLang for inference rollout
(needs INT4 HuggingFace format). The training loop itself runs in BF16 with
Fake QAT — INT4 quantization is applied on-the-fly during forward passes.

```bash
# ── 5a. Quantize BF16 → INT4 HuggingFace checkpoint ──────────────────────
# This creates a new HF-format model with INT4-packed expert weights.
# No calibration data needed — uses round-to-nearest quantization.
python /workspace/slime/tools/convert_hf_to_int4_direct.py \
    --model-dir /workspace/models/Qwen3-30B-A3B-Thinking-2507 \
    --save-dir  /workspace/models/Qwen3-30B-A3B-Thinking-2507-int4 \
    --group-size 128 \
    --is-symmetric True

# ── 5b. Convert BF16 model → torch_dist format (for reference model) ─────
# IMPORTANT: this must run on the original BF16 model, NOT the INT4 one.
# The converter loads BF16 weights and has no INT4 awareness.
# We must first source the model architecture config so MODEL_ARGS is set.
MODEL_ARGS_ROTARY_BASE=10000000
source /workspace/slime/scripts/models/qwen3-30B-A3B.sh
torchrun --nproc-per-node 1 /workspace/slime/tools/convert_hf_to_torch_dist.py \
    --hf-checkpoint /workspace/models/Qwen3-30B-A3B-Thinking-2507 \
    --save /workspace/models/Qwen3-30B-A3B-Thinking-2507-int4_torch_dist \
    "${MODEL_ARGS[@]}"
```

After Step 5, your `/workspace/models/` directory should look like:

```
/workspace/models/
├── Qwen3-30B-A3B-Thinking-2507/              # Original BF16 model (from HuggingFace)
├── Qwen3-30B-A3B-Thinking-2507-int4/         # INT4 HF checkpoint (created in 5a)  → used as --hf-checkpoint
└── Qwen3-30B-A3B-Thinking-2507-int4_torch_dist/  # Megatron format (created in 5b) → used as --ref-load
```

> **Note:** The training script also references `${MODEL_PATH}_slime_memex/`
> (i.e. `Qwen3-30B-A3B-Thinking-2507-int4_slime_memex/`). You do **not** need to create this
> — Slime auto-converts from `--hf-checkpoint` on the first run and saves
> training checkpoints there.

### Step 6: Launch training

The training script reads all configuration from environment variables.
**You do not need to edit the script.** Just `export` the 4 required variables below:

```bash
# ── Required: you must set these ──────────────────────────────────────────
export MODEL_PATH=/workspace/models/Qwen3-30B-A3B-Thinking-2507-int4
export DATA_DIR=/workspace/data/alfworld
export ALFWORLD_DATA=$HOME/.alfworld
export WANDB_KEY=<paste your Weights & Biases API key here>

# ── Launch ────────────────────────────────────────────────────────────────
bash /workspace/Memex/training/scripts/run_alfworld_qwen3_30B_A3B_memex.sh
```

The remaining environment variables (`MEMEX_ROOT`, `SLIME_ROOT`, `MEMEX_SLIME_ROOT`)
have defaults that match the container mount paths from Step 2, so you don't
need to set them if you followed the Docker setup above.

To monitor training, open the W&B dashboard — the run appears under project
`memex-alfworld`, group `alfworld-grpo-qwen3-30b-a3b-memex` (configurable via
`PROJECT_NAME` and `EXPERIMENT_NAME` env vars).

## Training Script

| Script | Description |
|--------|-------------|
| `run_alfworld_qwen3_30B_A3B_memex.sh` | Memex agent, Qwen3-30B-A3B (MoE, INT4) |

## Environment Variables

All configuration is via environment variables. The training script reads them
at launch — **you do not need to edit the script itself.**

**Required** (script exits with an error if not set):

| Variable | Description |
|----------|-------------|
| `MODEL_PATH` | Path to INT4 HF checkpoint (e.g., `/workspace/models/Qwen3-30B-A3B-Thinking-2507-int4`) |
| `DATA_DIR` | Path to converted JSONL data (e.g., `/workspace/data/alfworld`) |
| `ALFWORLD_DATA` | Path to ALFWorld game data (e.g., `$HOME/.alfworld`) |
| `WANDB_KEY` | Your [Weights & Biases](https://wandb.ai/authorize) API key |

**Optional** (have sensible defaults that work with the Docker setup in Step 2):

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMEX_ROOT` | `/workspace/Memex` | Path to this repo inside container |
| `SLIME_ROOT` | `/workspace/slime` | Path to Slime repo inside container |
| `MEMEX_SLIME_ROOT` | `/workspace/Memex/training` | Path to training adapter code |
| `COMPRESSION_MODE` | `lossless_db` | `none` (baseline), `lossless_db` (full memory) |
| `MAX_STEPS` | `50` | Max interaction steps per episode |
| `CONTEXT_THRESHOLD` | `8000` | Token count to trigger auto-compression |
| `NUM_GPUS` | `8` | Number of GPUs |
| `TP_SIZE` | `4` | Tensor-parallel size |
| `EP_SIZE` | `8` | Expert-parallel size (MoE) |
| `LR` | `5e-6` | Learning rate |
| `PROJECT_NAME` | `memex-alfworld` | W&B project name |
| `EXPERIMENT_NAME` | `alfworld-grpo-qwen3-30b-a3b-memex` | W&B group name |

## Project Structure

```
Memex/
├── src/                          # Core framework
│   ├── agents/                   # Agent implementations
│   │   ├── agent.py              # Base agent classes
│   │   ├── tool_agent.py         # Tool-calling agent with memory support
│   │   ├── memory/               # Memory compression & retrieval
│   │   └── alfworld/             # ALFWorld-specific agent
│   ├── environments/             # Environment wrappers
│   │   ├── alfworld/             # ALFWorld (TextWorld-based)
│   │   └── base/                 # Abstract base environment
│   ├── tools/                    # Tool base classes (Tool, ToolCall, ToolOutput)
│   ├── data/                     # Data loading & DatasetRegistry
│   ├── database/                 # Context storage (in-memory key-value)
│   ├── parser/                   # Tool call parsers (XML, Qwen format)
│   ├── rewards/                  # Reward shaping modules
│   └── engine/                   # Execution engine, rollout engines & token tracking
│
├── training/                     # Slime RL training integration
│   ├── memex_slime_adapter.py    # Multi-turn interaction runner
│   ├── generate_with_memex.py    # Slime custom generate function
│   ├── convert_data.py           # Data conversion utility
│   └── scripts/                  # Training launch scripts (run inside Docker)
│
└── pyproject.toml                # Package configuration
```

## External Dependencies

| Dependency | License | Purpose |
|-----------|---------|---------|
| [Slime](https://github.com/THUDM/slime) | Apache 2.0 | RL training framework (GRPO, distributed training) |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | BSD | Model parallelism (bundled in Slime Docker image) |
| [SGLang](https://github.com/sgl-project/sglang) | Apache 2.0 | Fast LLM inference for rollouts (bundled in Slime Docker image) |
| [ALFWorld](https://github.com/alfredo-lc/alfworld) | MIT | Text-based household environment |

## Citation

```bibtex
@article{wang2026memex,
    title={Memex(RL): Scaling Long-Horizon LLM Agents via Indexed Experience Memory},
    author={Wang, Zhenting and Chen, Huancheng and Wang, Jiayun and Wei, Wei},
    journal={arXiv preprint arXiv:2603.04257},
    year={2026}
}
```