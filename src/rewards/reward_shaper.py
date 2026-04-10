"""Base class for reward shaping strategies."""

from abc import ABC, abstractmethod
from typing import Any

from src.agents.agent import Trajectory
from src.environments.base.base_env import BaseEnv


class RewardShaper(ABC):
    """Base class for reward shaping strategies.

    RewardShaper provides a modular interface for applying reward shaping
    to agent trajectories during training. Subclasses implement specific
    shaping strategies (e.g., memory efficiency penalties).

    Design pattern:
    - Configuration-driven: Enable/disable via config flags
    - Lazy-loaded: Zero overhead when disabled
    - Composable: Multiple shapers can be combined
    """

    def __init__(self, config: dict):
        """Initialize reward shaper with configuration.

        Args:
            config: Configuration dictionary containing shaper-specific parameters.
        """
        self.config = config

    @abstractmethod
    def shape(
        self,
        base_reward: float,
        trajectory: Trajectory,
        env: BaseEnv,
        **kwargs: Any
    ) -> tuple[float, dict]:
        """Shape the reward based on trajectory analysis.

        Args:
            base_reward: Original reward from environment.
            trajectory: Completed agent trajectory containing steps and actions.
            env: Environment instance for context.
            **kwargs: Additional context (e.g., existing penalties, metadata).

        Returns:
            Tuple of (shaped_reward, penalty_info):
                - shaped_reward: Modified reward after applying shaping
                - penalty_info: Dictionary with penalty breakdown and debug info

        Example:
            shaped_reward, info = shaper.shape(
                base_reward=1.0,
                trajectory=agent.trajectory,
                env=env
            )

            # info structure:
            {
                'base_reward': 1.0,
                'shaped_reward': 0.85,
                'total_penalty': -0.15,
                'penalties': {
                    'penalty_type_1': {...},
                    'penalty_type_2': {...}
                }
            }
        """
        pass
