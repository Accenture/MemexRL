"""Slime custom generate function for Memex agents.

This is the entry point called by Slime's rollout manager via:
    --custom-generate-function-path generate_with_memex.generate

Configuration is read from environment variables (set in shell script's RUNTIME_ENV_JSON).

Signature must match Slime's expected contract:
    async def generate(args, sample, sampling_params) -> Sample | list[Sample]

When compression creates segments, returns list[Sample] (Slime supports this
natively at sglang_rollout.py:240 for multi-agent systems).
"""

import asyncio
import copy
import json
import logging
import os
from argparse import Namespace

from memex_slime_adapter import MemexInteractionRunner
from slime.rollout.sglang_rollout import GenerateState
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def get_memex_config() -> dict:
    """Read Memex configuration from environment variables.

    These are set in the shell script's RUNTIME_ENV_JSON and injected into
    Ray workers by Slime.
    """
    return {
        "env_type": os.environ.get("MEMEX_ENV_TYPE", "alfworld"),
        "compression_mode": os.environ.get("MEMEX_COMPRESSION_MODE", "lossless_db"),
        "tool_call_format": os.environ.get("MEMEX_TOOL_CALL_FORMAT", "qwen"),
        "max_steps": int(os.environ.get("MEMEX_MAX_STEPS", "30")),
        "context_length_threshold": int(os.environ.get("MEMEX_CONTEXT_THRESHOLD", "3000")),
        "auto_compress_prompt": os.environ.get("MEMEX_AUTO_COMPRESS", "true").lower() == "true",
        "disable_retrieve": os.environ.get("MEMEX_DISABLE_RETRIEVE", "false").lower() == "true",
        "reward_shaper_enable": os.environ.get("MEMEX_REWARD_SHAPER_ENABLE", "true").lower() == "true",
        "reward_lambda_ctx": float(os.environ.get("MEMEX_LAMBDA_CTX", "0.5")),
        "reward_lambda_red": float(os.environ.get("MEMEX_LAMBDA_RED", "0.3")),
        "reward_lambda_format": float(os.environ.get("MEMEX_LAMBDA_FORMAT", "0.2")),
        "hide_admissible_commands": os.environ.get("ALFWORLD_HIDE_ADMISSIBLE_COMMANDS", "false").lower() == "true",
        "hide_initial_obs": os.environ.get("ALFWORLD_HIDE_INITIAL_OBS", "false").lower() == "true",
        "max_summary_tokens": int(os.environ.get("MEMEX_MAX_SUMMARY_TOKENS", "0")),
    }


def _create_agent(config: dict):
    """Create a Memex agent based on config."""
    env_type = config["env_type"]

    if env_type == "alfworld":
        from src.agents.alfworld.agent import ALFWorldAgentWithMemory

        return ALFWorldAgentWithMemory(
            tool_call_format=config["tool_call_format"],
            compression_mode=config["compression_mode"],
            context_length_threshold=config["context_length_threshold"],
            auto_compress_prompt=config["auto_compress_prompt"],
            disable_retrieve=config["disable_retrieve"],
            hide_admissible_commands=config.get("hide_admissible_commands", False),
            hide_initial_obs=config.get("hide_initial_obs", False),
            max_summary_tokens=config.get("max_summary_tokens", 0),
        )
    else:
        raise ValueError(f"Unsupported env_type: {env_type}")


def _create_env(config: dict, task: dict):
    """Create a Memex environment based on config and task.

    Uses AsyncParallelALFWorldEnv by default (each env in its own subprocess)
    to avoid TextWorld's global file lock that serializes all env.step() calls.
    Set MEMEX_PARALLEL_ENV=false to fall back to single-process ALFWorldEnv.
    """
    env_type = config["env_type"]

    if env_type == "alfworld":
        use_parallel = os.environ.get("MEMEX_PARALLEL_ENV", "true").lower() == "true"
        if use_parallel:
            from src.environments.alfworld.async_parallel_env import AsyncParallelALFWorldEnv
            return AsyncParallelALFWorldEnv.from_dict(task)
        else:
            from src.environments.alfworld.env import ALFWorldEnv
            return ALFWorldEnv.from_dict(task)
    else:
        raise ValueError(f"Unsupported env_type: {env_type}")


async def generate(args: Namespace, sample: Sample, sampling_params: dict):
    """Slime custom generate function for Memex agents.

    Called by Slime's rollout manager for each sample. Runs a complete
    agent-environment interaction.

    Returns:
        Single Sample when no compression (baseline mode).
        list[Sample] when compression creates segments (Slime native support at line 240).
    """
    assert not args.partial_rollout, (
        "Partial rollout is not supported for multi-turn Memex interactions."
    )

    config = get_memex_config()
    state = GenerateState(args)
    url = f"http://{args.sglang_router_ip}:{args.sglang_router_port}/generate"

    status_map = {
        "completed": Sample.Status.COMPLETED,
        "truncated": Sample.Status.TRUNCATED,
        "aborted": Sample.Status.ABORTED,
        "failed": Sample.Status.FAILED,
    }

    env = None
    try:
        # 1. Parse task from sample.prompt
        task = json.loads(sample.prompt)

        # 2. Create environment and agent
        from memex_slime_adapter import _ENV_EXECUTOR
        env = await asyncio.get_event_loop().run_in_executor(
            _ENV_EXECUTOR, _create_env, config, task
        )
        agent = _create_agent(config)

        # 3. Run interaction
        runner = MemexInteractionRunner(state.tokenizer, config)
        result = await runner.run(agent, env, url, sampling_params)

        # 4. Handle result
        if isinstance(result, list):
            # Segmented: create one Sample per segment, all sharing group_index & reward
            # Slime's generate_and_rm (sglang_rollout.py:240) natively handles list[Sample].
            episode_uid = f"{sample.group_index}_{sample.index}"
            samples = []
            for seg_idx, seg_result in enumerate(result):
                seg_sample = copy.copy(sample)  # preserves group_index, prompt, etc.
                seg_sample.tokens = seg_result.prompt_token_ids + seg_result.response_token_ids
                seg_sample.response = seg_result.response_text
                seg_sample.response_length = len(seg_result.loss_mask)
                seg_sample.reward = seg_result.reward
                seg_sample.loss_mask = seg_result.loss_mask
                seg_sample.rollout_log_probs = seg_result.rollout_log_probs if seg_result.rollout_log_probs else None
                seg_sample.index = sample.index * 10000 + seg_idx
                seg_sample.status = status_map.get(seg_result.status, Sample.Status.FAILED)
                seg_sample.metadata = {
                    **(seg_result.metadata or {}),
                    "episode_uid": episode_uid,
                }
                samples.append(seg_sample)
            return samples
        else:
            # Single result (baseline mode or no compression occurred)
            sample.tokens = result.prompt_token_ids + result.response_token_ids
            sample.response = result.response_text
            sample.response_length = len(result.loss_mask)
            sample.reward = result.reward
            sample.loss_mask = result.loss_mask
            sample.rollout_log_probs = result.rollout_log_probs if result.rollout_log_probs else None
            sample.status = status_map.get(result.status, Sample.Status.FAILED)
            sample.metadata = result.metadata
            return sample

    except Exception as e:
        logger.error(f"generate() failed: {e}", exc_info=True)
        sample.status = Sample.Status.FAILED
        sample.reward = 0.0
        # Provide valid defaults so data_preprocess won't crash on len(None)
        if not sample.tokens:
            sample.tokens = state.tokenizer.encode(sample.prompt)[:10]
        if not sample.response_length:
            sample.response_length = 0
        if sample.loss_mask is None:
            sample.loss_mask = []
        if sample.rollout_log_probs is None:
            sample.rollout_log_probs = []
        return sample

    finally:
        # Always clean up the environment to prevent Docker container leaks
        if env is not None:
            try:
                from memex_slime_adapter import _ENV_EXECUTOR
                await asyncio.get_event_loop().run_in_executor(
                    _ENV_EXECUTOR, env.close,
                )
            except Exception as cleanup_err:
                logger.warning(f"env.close() failed: {cleanup_err}")
