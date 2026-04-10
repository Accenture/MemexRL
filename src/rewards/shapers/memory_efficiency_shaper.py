"""Memory efficiency reward shaper with context overflow and redundancy penalties."""

import hashlib
import json
import logging
from typing import Any

from src.agents.agent import Trajectory
from src.environments.base.base_env import BaseEnv
from src.rewards.reward_shaper import RewardShaper

logger = logging.getLogger(__name__)

# Tools to exclude from penalty calculations (memory management tools)
MEMORY_TOOLS = frozenset({'CompressExperience', 'ReadExperience'})

# State-modifying file_editor commands
STATE_MODIFYING_COMMANDS = frozenset({'str_replace', 'insert', 'create'})

# ALFWorld state-modifying action prefixes (these change world/inventory state)
ALFWORLD_STATE_MODIFYING_PREFIXES = (
    'pick up', 'take', 'put', 'open', 'close',
    'use', 'clean', 'heat', 'cool', 'slice', 'toggle'
)


class MemoryEfficiencyShaper(RewardShaper):
    """Reward shaper that penalizes memory inefficiency.

    Applies three normalized penalties (all in [0,1] range):
    1. Context overflow: Penalize when working context exceeds threshold
    2. Redundant tool calls: Penalize duplicate tool calls after state modifications
    3. Format errors: Penalize malformed tool calls (tag mismatches, invalid JSON)

    All penalties are normalized to [0,1] so lambda weights are intuitive:
    - penalty = -λ * ratio, where ratio ∈ [0, 1]
    - When ratio=0: no penalty
    - When ratio=1: full penalty of -λ

    Configuration:
        lambda_ctx (float): Context overflow penalty weight (default: 0.5)
        lambda_red (float): Redundant tool call penalty weight (default: 0.3)
        lambda_format (float): Format error penalty weight (default: 0.2)
        context_threshold (int): Working context threshold in tokens (default: 8000)
        enable_context_penalty (bool): Enable context overflow penalty (default: True)
        enable_redundancy_penalty (bool): Enable redundant tool penalty (default: True)
        enable_format_penalty (bool): Enable format error penalty (default: True)
    """

    def __init__(self, config: dict):
        """Initialize memory efficiency shaper.

        Args:
            config: Configuration dictionary with penalty weights and thresholds.
        """
        super().__init__(config)
        self.lambda_ctx = config.get('lambda_ctx', 0.5)
        self.lambda_red = config.get('lambda_red', 0.3)
        self.lambda_format = config.get('lambda_format', 0.2)
        self.threshold = config.get('context_threshold', 8000)
        self.enable_context_penalty = config.get('enable_context_penalty', True)
        self.enable_redundancy_penalty = config.get('enable_redundancy_penalty', True)
        self.enable_format_penalty = config.get('enable_format_penalty', True)

    def shape(
        self,
        base_reward: float,
        trajectory: Trajectory,
        env: BaseEnv,
        **kwargs: Any
    ) -> tuple[float, dict]:
        """Apply memory efficiency penalties to reward.

        Args:
            base_reward: Original reward from environment.
            trajectory: Completed agent trajectory.
            env: Environment instance.
            **kwargs: Additional context.

        Returns:
            Tuple of (shaped_reward, penalty_info) where penalty_info contains
            detailed breakdown of penalties applied.
        """
        total_penalty = 0.0
        penalty_info = {
            'base_reward': base_reward,
            'penalties': {}
        }

        # 1. Context overflow penalty
        if self.enable_context_penalty:
            ctx_penalty, ctx_info = self._compute_context_overflow_penalty(
                trajectory, self.threshold
            )
            total_penalty += ctx_penalty
            penalty_info['penalties']['context_overflow'] = ctx_info

        # 2. Redundant tool call penalty
        if self.enable_redundancy_penalty:
            red_penalty, red_info = self._compute_redundant_tool_penalty(trajectory)
            total_penalty += red_penalty
            penalty_info['penalties']['redundant_tools'] = red_info

        # 3. Format error penalty
        if self.enable_format_penalty:
            fmt_penalty, fmt_info = self._compute_format_penalty(trajectory)
            total_penalty += fmt_penalty
            penalty_info['penalties']['format_errors'] = fmt_info

        # Compute shaped reward
        shaped_reward = base_reward + total_penalty
        penalty_info['total_penalty'] = total_penalty
        penalty_info['shaped_reward'] = shaped_reward

        return shaped_reward, penalty_info

    def _compute_context_overflow_penalty(
        self,
        trajectory: Trajectory,
        threshold: int
    ) -> tuple[float, dict]:
        """Compute normalized penalty for exceeding context threshold.

        Penalty formula:
            penalty = -λ_ctx * min(1.0, Σ overflow / (threshold * num_steps))

        Where:
            - overflow = max(0, context_size - threshold) per step
            - Normalized by "max reasonable overflow" = threshold * num_steps
            - This gives overflow_ratio ∈ [0, 1]
            - Final penalty = -λ_ctx * overflow_ratio

        Logic:
            - Iterate through trajectory steps
            - Skip steps where compression occurred (agent managing context)
            - Accumulate overflow tokens
            - Normalize to get proportion in [0, 1]
            - Apply lambda weight

        Args:
            trajectory: Agent trajectory with steps
            threshold: Context size threshold in tokens

        Returns:
            Tuple of (penalty, info_dict) where info_dict contains:
                - context_overflow_penalty: Computed penalty value
                - total_overflow_tokens: Sum of all overflow tokens
                - overflow_ratio: Normalized ratio in [0, 1]
                - num_overflow_steps: Number of steps with overflow
                - overflow_steps: List of first 5 overflow steps (for logging)
        """
        total_overflow = 0.0
        overflow_steps = []
        num_steps = len(trajectory.steps)

        for step_idx, step in enumerate(trajectory.steps):
            # Get context size for this step
            context_size = getattr(step, 'context_length', 0)

            # Skip steps where compression occurred (agent is managing context)
            if getattr(step, 'num_compressions_in_step', 0) > 0:
                continue

            # Compute overflow
            overflow = max(0, context_size - threshold)
            if overflow > 0:
                total_overflow += overflow
                overflow_steps.append({
                    'step': step_idx,
                    'context_size': context_size,
                    'overflow': overflow
                })

        # Normalize: divide by "max reasonable overflow" = threshold * num_steps
        # This gives a proportion that naturally stays in [0, ~1] range
        max_overflow = threshold * num_steps if num_steps > 0 else 1
        overflow_ratio = min(1.0, total_overflow / max_overflow) if max_overflow > 0 else 0.0

        # Apply lambda weight (now lambda_ctx represents the penalty weight when ratio=1.0)
        penalty = -self.lambda_ctx * overflow_ratio

        info = {
            'context_overflow_penalty': penalty,
            'total_overflow_tokens': total_overflow,
            'overflow_ratio': overflow_ratio,
            'num_overflow_steps': len(overflow_steps),
            'overflow_steps': overflow_steps[:5]  # Keep first 5 for logging
        }

        return penalty, info

    def _compute_redundant_tool_penalty(self, trajectory: Trajectory) -> tuple[float, dict]:
        """Detect redundant tool calls and compute normalized penalty.

        Uses step.parse_result.tool_calls directly instead of re-parsing.
        """
        call_history: dict[str, int] = {}  # call_hash -> last_seen_step
        redundant_calls: list[dict] = []
        total_tool_calls = 0
        last_state_modify_step = -1

        for step_idx, step in enumerate(trajectory.steps):
            parse_result = step.parse_result
            if parse_result is None or not parse_result.tool_calls:
                continue

            for tc in parse_result.tool_calls:
                if tc.name in MEMORY_TOOLS:
                    continue

                total_tool_calls += 1

                # Hash the call signature
                try:
                    args_str = json.dumps(tc.arguments, sort_keys=True)
                except (TypeError, ValueError):
                    args_str = str(tc.arguments)
                call_hash = hashlib.md5(f"{tc.name}::{args_str}".encode()).hexdigest()

                # Check redundancy (same call after last state modification)
                if call_hash in call_history and call_history[call_hash] > last_state_modify_step:
                    redundant_calls.append({
                        'step': step_idx,
                        'tool': tc.name,
                        'call_hash': call_hash[:8],
                        'last_call_step': call_history[call_hash],
                    })

                call_history[call_hash] = step_idx

                # Track state modifications
                if tc.name == 'file_editor' and isinstance(tc.arguments, dict):
                    if tc.arguments.get('command') in STATE_MODIFYING_COMMANDS:
                        last_state_modify_step = step_idx

                # Track ALFWorld state modifications (execute_action with state-changing verbs)
                if tc.name == 'execute_action' and isinstance(tc.arguments, dict):
                    action = tc.arguments.get('action', '')
                    if isinstance(action, str) and action.lower().startswith(ALFWORLD_STATE_MODIFYING_PREFIXES):
                        last_state_modify_step = step_idx

        # Compute penalty
        ratio = len(redundant_calls) / total_tool_calls if total_tool_calls > 0 else 0.0
        penalty = -self.lambda_red * ratio

        return penalty, {
            'redundant_tool_penalty': penalty,
            'num_redundant_calls': len(redundant_calls),
            'total_tool_calls': total_tool_calls,
            'redundancy_ratio': ratio,
            'redundant_calls': redundant_calls[:5],
        }

    def _compute_format_penalty(self, trajectory: Trajectory) -> tuple[float, dict]:
        """Compute format penalty using pre-parsed results from step.parse_result."""
        steps_with_errors: set[int] = set()
        total_attempts = 0
        total_errors = 0

        for step in trajectory.steps:
            pr = step.parse_result
            if pr is None:
                continue
            if pr.had_tool_attempt:
                total_attempts += 1
            if pr.format_errors:
                steps_with_errors.add(id(step))
                total_errors += len(pr.format_errors)

        ratio = len(steps_with_errors) / total_attempts if total_attempts > 0 else 0.0
        penalty = -self.lambda_format * ratio

        return penalty, {
            'format_penalty': penalty,
            'num_format_errors': total_errors,
            'steps_with_errors': len(steps_with_errors),
            'total_tool_attempts': total_attempts,
            'format_error_ratio': ratio,
        }
