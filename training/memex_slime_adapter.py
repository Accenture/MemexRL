"""Core adapter: Memex agent-environment interaction loop for Slime training.

This module runs the multi-turn agent-env interaction with proper token-level
loss_mask tracking, using raw token IDs from SGLang (avoiding decode-reencode).

Key design:
- Assistant token IDs come from SGLang's output_token_logprobs (raw, no re-encode)
- Env/tool observation tokens are encoded from text (not model-generated, so no re-encode issue)
- Uses input_ids (not text) to send prompts to SGLang for token consistency
- Trajectory segmentation: when CompressExperience succeeds, the current segment is
  saved and a new segment starts from the compressed prompt.  Each segment becomes
  an independent training sample (matching Memex/verl's _build_segmented_token_results).
  No re-tokenization needed — raw token IDs are preserved per segment (token-in-token-out).
"""

import asyncio
import concurrent.futures
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Union

from slime.utils.http_utils import post

# Large thread pool for env.step() / env.reset() calls.
# Default asyncio pool is min(32, cpu+4), too small for 128+ concurrent envs.
_ENV_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=256)

logger = logging.getLogger(__name__)


@dataclass
class MemexInteractionResult:
    """Result of a complete agent-environment interaction."""

    prompt_token_ids: list[int] = field(default_factory=list)
    response_token_ids: list[int] = field(default_factory=list)
    response_text: str = ""
    loss_mask: list[int] = field(default_factory=list)
    rollout_log_probs: list[float] = field(default_factory=list)
    reward: float = 0.0
    status: str = "failed"  # completed / truncated / aborted / failed
    metadata: dict = field(default_factory=dict)


class MemexInteractionRunner:
    """Runs multi-turn Memex agent-env interactions with raw token tracking.

    Follows the pattern of tau-bench's TrainableAgentMixin.asolve() but
    fixes the decode-reencode problem by using SGLang's output_token_logprobs.
    """

    def __init__(self, tokenizer, config: dict):
        self.tokenizer = tokenizer
        self.config = config

    async def run(
        self,
        agent,
        env,
        sglang_url: str,
        sampling_params: dict,
    ) -> Union[MemexInteractionResult, list[MemexInteractionResult]]:
        """Execute multi-turn agent-environment interaction.

        Args:
            agent: Memex agent (ToolAgent or ToolAgentWithMemory).
            env: Memex environment (e.g. ALFWorldEnv).
            sglang_url: SGLang /generate endpoint URL.
            sampling_params: LLM sampling parameters.

        Returns:
            Single MemexInteractionResult when no compression occurred (baseline),
            or list[MemexInteractionResult] when compression created segments.
        """
        max_steps = self.config.get("max_steps", 30)

        # 1. Reset environment and agent (run in thread to avoid blocking event loop)
        obs, info = await asyncio.get_event_loop().run_in_executor(_ENV_EXECUTOR, env.reset)
        agent.reset()
        agent.update_from_env(observation=obs, reward=0.0, done=False, info=info, env=env)

        # 2. Set up token_manager for context status reporting
        self._setup_token_manager(agent)

        # 3. Tokenize initial prompt
        prompt_text = self.tokenizer.apply_chat_template(
            agent.chat_completions, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        accumulated_ids = list(prompt_ids)

        # 4. Initialize tracking
        response_ids: list[int] = []
        loss_masks: list[int] = []
        rollout_log_probs: list[float] = []
        status = "truncated"
        done = False
        total_bpe_mismatches = 0
        step_idx = -1
        # Observability counters
        peak_accumulated_len = len(accumulated_ids)
        compress_count = 0
        retrieve_count = 0
        env_action_count = 0
        # Segment tracking (for compression mode)
        segments: list[dict] = []

        # 5. Compute token budget (aligned with geo3k_vlm_multi_turn pattern)
        #    Uses MEMEX_MAX_CONTEXT_LEN env var as total context limit.
        #    Budget decreases each step; per-step max_new_tokens is capped to remaining budget.
        env_max_ctx = os.environ.get("MEMEX_MAX_CONTEXT_LEN")
        if env_max_ctx:
            budget = int(env_max_ctx) - len(accumulated_ids)
        elif sampling_params.get("max_new_tokens") is not None:
            budget = sampling_params["max_new_tokens"]
        else:
            budget = None  # no limit

        step_sampling_params = dict(sampling_params)

        # 6. Multi-turn loop
        for step_idx in range(max_steps):
            # a. Guard: truncate if budget exhausted
            if budget is not None and budget <= 0:
                logger.warning(
                    f"Budget exhausted at step {step_idx} (accumulated={len(accumulated_ids)}), truncating."
                )
                status = "truncated"
                break

            # b. Cap per-step max_new_tokens to remaining budget
            if budget is not None:
                step_sampling_params["max_new_tokens"] = min(
                    sampling_params.get("max_new_tokens", 1024), budget
                )

            # c. Send input_ids to SGLang (not text — avoids prompt re-tokenization)
            payload = {
                "input_ids": accumulated_ids,
                "sampling_params": step_sampling_params,
                "return_logprob": True,
            }

            try:
                output = await post(sglang_url, payload)
            except Exception as e:
                logger.error(f"SGLang call failed at step {step_idx}: {e}")
                status = "failed"
                break

            # b. Check abort
            finish_reason = output.get("meta_info", {}).get("finish_reason", {})
            if isinstance(finish_reason, dict) and finish_reason.get("type") == "abort":
                status = "aborted"
                break

            # c. Capture RAW assistant token IDs + log probs (core: no decode-reencode)
            logprobs_data = output.get("meta_info", {}).get("output_token_logprobs", [])
            raw_asst_ids = [item[1] for item in logprobs_data]
            raw_asst_logprobs = [item[0] for item in logprobs_data]
            asst_text = output.get("text", "")

            # Remove end-of-turn markers
            for marker in ("<|im_end|>", "<|endoftext|>"):
                if asst_text.endswith(marker):
                    asst_text = asst_text[: -len(marker)]

            # d. Append raw assistant tokens to accumulated sequence + update budget
            accumulated_ids.extend(raw_asst_ids)
            response_ids.extend(raw_asst_ids)
            peak_accumulated_len = max(peak_accumulated_len, len(accumulated_ids))
            loss_masks.extend([1] * len(raw_asst_ids))
            rollout_log_probs.extend(raw_asst_logprobs)
            if budget is not None:
                budget -= len(raw_asst_ids)

            # e. Update agent + determine action (offload to thread pool)
            def _sync_parse_action():
                parse_result = agent.update_from_model(asst_text)
                tool_calls = parse_result.tool_calls if parse_result else []
                tool_name = tool_calls[0].name if tool_calls else None
                tool_params = tool_calls[0].arguments if tool_calls else {}
                is_memory = (
                    hasattr(agent, "is_memory_tool")
                    and tool_name
                    and agent.is_memory_tool(tool_name)
                )
                mem_result = None
                if is_memory:
                    mem_result = agent.execute_memory_tool(tool_name, tool_params)
                return parse_result, tool_name, tool_params, is_memory, mem_result

            parse_result, tool_name, tool_params, is_memory, result = (
                await asyncio.get_event_loop().run_in_executor(
                    _ENV_EXECUTOR, _sync_parse_action
                )
            )

            if is_memory:
                if tool_name == "CompressExperience":
                    compress_count += 1
                elif tool_name == "ReadExperience":
                    retrieve_count += 1

                if result.success and tool_name == "CompressExperience":
                    # ====================================================
                    # SEGMENT BOUNDARY: save current segment, start new one
                    # (matches Memex mixin.py:355-372 _do_lossless_compress)
                    # ====================================================
                    # The CompressExperience call tokens are already in
                    # response_ids (loss_mask=1), so they're part of this
                    # segment's trainable tokens.

                    # Save current segment (raw token IDs, no re-tokenization)
                    seg_type = "pre_compression" if len(segments) == 0 else "post_compression"
                    segments.append({
                        "prompt_ids": list(prompt_ids),
                        "response_ids": list(response_ids),
                        "loss_masks": list(loss_masks),
                        "rollout_log_probs": list(rollout_log_probs),
                        "segment_type": seg_type,
                        "compressed_at_step": step_idx,
                    })
                    logger.info(
                        f"Segment saved: type={seg_type}, step={step_idx}, "
                        f"prompt_len={len(prompt_ids)}, response_len={len(response_ids)}, "
                        f"compress #{compress_count}"
                    )

                    # Build new prompt from compressed messages
                    new_prompt_text = self.tokenizer.apply_chat_template(
                        agent.chat_completions,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    new_prompt_ids = self.tokenizer.encode(
                        new_prompt_text, add_special_tokens=False
                    )

                    # Reset for new segment
                    prompt_ids = list(new_prompt_ids)
                    response_ids = []
                    loss_masks = []
                    rollout_log_probs = []
                    accumulated_ids = list(new_prompt_ids)

                    # Recalculate budget from compressed accumulated_ids.
                    if env_max_ctx:
                        budget = int(env_max_ctx) - len(accumulated_ids)

                    continue  # Skip env observation — messages already updated

                # ReadExperience or failed CompressExperience: result as observation
                next_obs = result.message
                reward_step, done = 0.0, False
                info_step = {"memory_tool": True, "memory_tool_name": tool_name}
            else:
                # Environment step
                env_action_count += 1
                action = env.format_action(parse_result)
                if action is None:
                    # No valid tool call parsed — pass error observation
                    next_obs = (
                        "[No valid tool call detected. "
                        "Please use the correct tool call format.]"
                    )
                    reward_step, done = 0.0, False
                    info_step = {"parse_error": True}
                else:
                    try:
                        next_obs, reward_step, done, info_step = await asyncio.wait_for(
                            asyncio.get_event_loop().run_in_executor(_ENV_EXECUTOR, env.step, action),
                            timeout=180,  # 3 min guard against Docker hangs
                        )
                    except asyncio.TimeoutError:
                        logger.error(f"env.step() timed out at step {step_idx}")
                        next_obs = "[Environment step timed out after 180 seconds.]"
                        reward_step, done = 0.0, False
                        info_step = {"timeout": True}

            # g. Update agent + compute env delta tokens (offload to thread pool
            #    to avoid blocking the event loop — tokenization is CPU-intensive)
            def _sync_update_and_tokenize():
                agent.update_from_env(
                    observation=next_obs, reward=reward_step, done=done, info=info_step, env=env
                )
                full_text = self.tokenizer.apply_chat_template(
                    agent.chat_completions, tokenize=False, add_generation_prompt=True
                )
                return self.tokenizer.encode(full_text, add_special_tokens=False)

            full_ids = await asyncio.get_event_loop().run_in_executor(
                _ENV_EXECUTOR, _sync_update_and_tokenize
            )

            if len(full_ids) >= len(accumulated_ids):
                env_delta = full_ids[len(accumulated_ids) :]
            else:
                total_bpe_mismatches += 1
                env_delta = full_ids[len(prompt_ids) + len(response_ids) :]
                accumulated_ids = list(full_ids)
                logger.debug(
                    f"BPE mismatch at step {step_idx}: "
                    f"full_ids={len(full_ids)} vs accumulated={len(accumulated_ids)}"
                )

            accumulated_ids.extend(env_delta)
            response_ids.extend(env_delta)
            loss_masks.extend([0] * len(env_delta))
            rollout_log_probs.extend([0.0] * len(env_delta))
            peak_accumulated_len = max(peak_accumulated_len, len(accumulated_ids))
            if budget is not None:
                budget -= len(env_delta)

            # h2. Set context_length on current trajectory step for reward shaping
            #     (matches agent_execution_engine.py:609)
            cur_step = getattr(agent, '_current_step', None) or (
                agent._trajectory.steps[-1] if hasattr(agent, '_trajectory') and agent._trajectory.steps else None
            )
            if cur_step is not None:
                cur_step.context_length = len(accumulated_ids)

            # i. Check termination
            if done:
                status = "completed"
                break

        # 6. Compute final reward
        base_reward = env.compute_final_reward() if hasattr(env, "compute_final_reward") else 0.0
        shaped_reward, penalty_info = self._apply_reward_shaping(base_reward, agent, env)

        # 7. Build metadata (covers full episode, not just final segment)
        metadata = {
            "base_reward": base_reward,
            "shaped_reward": shaped_reward,
            "steps": step_idx + 1,
            "status": status,
            "bpe_mismatches": total_bpe_mismatches,
            # Observability: context & memory metrics
            "peak_accumulated_len": peak_accumulated_len,
            "final_accumulated_len": len(accumulated_ids),
            "compress_count": compress_count,
            "retrieve_count": retrieve_count,
            "env_action_count": env_action_count,
            "num_segments": len(segments) + (1 if segments else 0),
        }

        # Add penalty breakdown when reward shaping is enabled
        if penalty_info:
            metadata["penalty_info"] = penalty_info

        # Add memory stats if available
        if hasattr(agent, "get_memory_stats"):
            metadata["memory_stats"] = agent.get_memory_stats()

        # Log episode summary
        logger.info(
            f"Episode done: status={status}, reward={shaped_reward:.2f}, "
            f"steps={step_idx+1}, env_actions={env_action_count}, "
            f"compress={compress_count}, retrieve={retrieve_count}, "
            f"peak_ctx={peak_accumulated_len}, final_ctx={len(accumulated_ids)}, "
            f"segments={len(segments) + (1 if segments else 0)}, "
            f"response_len={len(response_ids)}, trainable={sum(loss_masks)}"
        )

        # 8. Build response text
        response_text = ""
        for msg in agent.chat_completions:
            if msg.get("role") == "assistant":
                response_text += msg.get("content", "")

        # 9. Return: segmented (list) or single result
        if segments:
            # Finalize: save current response as final segment
            # (matches Memex mixin.py:786-792 finalize_segments)
            if len(response_ids) > 0:
                segments.append({
                    "prompt_ids": list(prompt_ids),
                    "response_ids": list(response_ids),
                    "loss_masks": list(loss_masks),
                    "rollout_log_probs": list(rollout_log_probs),
                    "segment_type": "final",
                })

            # Build one MemexInteractionResult per segment, all sharing episode reward
            results = []
            for seg_idx, seg in enumerate(segments):
                is_last = (seg_idx == len(segments) - 1)
                seg_metadata = {
                    **metadata,
                    "segment_idx": seg_idx,
                    "segment_type": seg["segment_type"],
                    "segment_response_len": len(seg["response_ids"]),
                    "segment_trainable": sum(seg["loss_masks"]),
                }
                results.append(MemexInteractionResult(
                    prompt_token_ids=seg["prompt_ids"],
                    response_token_ids=seg["response_ids"],
                    response_text=response_text if is_last else "",
                    loss_mask=seg["loss_masks"],
                    rollout_log_probs=seg["rollout_log_probs"],
                    reward=shaped_reward,
                    status=status if is_last else "completed",
                    metadata=seg_metadata,
                ))
            return results
        else:
            # No compression happened — single result (baseline compatible)
            metadata["total_response_len"] = len(response_ids)
            metadata["trainable_tokens"] = sum(loss_masks)
            return MemexInteractionResult(
                prompt_token_ids=prompt_ids,
                response_token_ids=response_ids,
                response_text=response_text,
                loss_mask=loss_masks,
                rollout_log_probs=rollout_log_probs,
                reward=shaped_reward,
                status=status,
                metadata=metadata,
            )

    def _setup_token_manager(self, agent):
        """Set up token_manager on agent for context status reporting.

        MemoryAgentMixin._estimate_working_tokens() needs agent.token_manager.
        Without it, context status always shows working_tokens=0.
        """
        try:
            from src.engine.token_stats_manager import TokenStatsManager
            from src.parser import ChatTemplateParser

            chat_parser = ChatTemplateParser.get_parser(self.tokenizer)
            agent.token_manager = TokenStatsManager(
                tokenizer=self.tokenizer,
                chat_parser=chat_parser,
                trajectory_idx=0,
            )
        except ImportError:
            logger.warning(
                "Could not import TokenStatsManager. "
                "Context status will show working_tokens=0."
            )

    def _apply_reward_shaping(self, base_reward: float, agent, env) -> tuple[float, dict]:
        """Apply memory efficiency reward shaping if configured.

        Returns:
            Tuple of (shaped_reward, penalty_info). penalty_info is empty dict when disabled.
        """
        if not self.config.get("reward_shaper_enable", False):
            return base_reward, {}

        try:
            from src.rewards.shapers.memory_efficiency_shaper import MemoryEfficiencyShaper

            shaper_config = {
                "lambda_ctx": self.config.get("reward_lambda_ctx", 0.5),
                "lambda_red": self.config.get("reward_lambda_red", 0.3),
                "lambda_format": self.config.get("reward_lambda_format", 0.2),
                "context_threshold": self.config.get("context_length_threshold", 3000),
            }

            trajectory = getattr(agent, "trajectory", None) or getattr(
                agent, "_trajectory", None
            )
            if trajectory is None:
                logger.warning("Agent has no trajectory for reward shaping")
                return base_reward, {}

            shaper = MemoryEfficiencyShaper(shaper_config)
            shaped_reward, penalty_info = shaper.shape(
                base_reward=base_reward,
                trajectory=trajectory,
                env=env,
            )
            return shaped_reward, penalty_info

        except ImportError:
            logger.warning("Could not import MemoryEfficiencyShaper")
            return base_reward, {}
        except Exception as e:
            logger.warning(f"Reward shaping failed: {e}")
            return base_reward, {}
