"""
ALFWorld Environment for Memex.

This module provides the ALFWorld environment for household task agents.
ALFWorld is a text-based game environment for embodied AI research.
"""

import json
import os
import tempfile
from contextlib import nullcontext as _nullcontext
from typing import Any, Callable, Optional

from filelock import FileLock
from src.environments.base.base_env import BaseEnv


# Global file lock for TextWorld initialization, reset, and step
# TextWorld's tatsu PDDL parser is not thread-safe and not process-safe
# Using file lock to serialize access across all processes
_TEXTWORLD_LOCK_PATH = os.path.join(tempfile.gettempdir(), "textworld_parser.lock")
_TEXTWORLD_LOCK = FileLock(_TEXTWORLD_LOCK_PATH, timeout=300)


# Tool definitions for ALFWorld agent
ALFWORLD_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_action",
            "description": "Execute an action in the ALFWorld environment. Actions are natural language commands like 'go to desk 1', 'pick up book 1', 'put book 1 in/on desk 1', 'open drawer 1', 'use lamp 1', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "The action to execute in natural language"
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Indicate that you have completed the task or give up.",
            "parameters": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the task was completed successfully"
                    }
                },
                "required": ["success"]
            }
        }
    }
]


def get_alfworld_tools() -> list:
    """Get tool definitions for ALFWorld environment."""
    return ALFWORLD_TOOLS.copy()


class ALFWorldEnv(BaseEnv):
    """
    ALFWorld Environment for Memex.

    This environment wraps ALFWorld/TextWorld for household task agents.
    Supports tasks like pick_and_place, pick_clean_then_place, etc.
    """

    def __init__(
        self,
        task: Optional[dict] = None,
        reward_fn: Optional[Callable] = None,
        max_steps: int = 50,
        use_process_lock: bool = True,
        **kwargs
    ):
        """Initialize the ALFWorld environment.

        Args:
            task: Task dictionary containing game_file, task_type, etc.
            reward_fn: Optional reward function for custom reward computation.
            max_steps: Maximum number of steps before episode ends.
            use_process_lock: Whether to use file lock for TextWorld operations.
                Set to False when running in isolated process (e.g., Ray actor).
        """
        super().__init__(**kwargs)

        self.task = task or {}
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self.use_process_lock = use_process_lock

        # Game file path
        self.game_file = self.task.get("game_file")

        # Lazy initialization
        self._env = None
        self._config = None

        # State variables
        self.current_step = 0
        self.done = False
        self.won = False
        self.admissible_commands = []
        self.interaction_history = []
        self.final_response = None

        # Limit "look" to once per episode (controlled by env var)
        self._limit_look = os.environ.get("ALFWORLD_LIMIT_LOOK", "false").lower() == "true"
        self._look_used = False

    def _lazy_init(self):
        """Lazily initialize the ALFWorld environment using TextWorld directly.

        Uses a global lock to protect TextWorld's tatsu PDDL parser which is not thread-safe.
        The lock can be disabled when running in isolated processes (e.g., Ray actors).
        """
        if self._env is not None:
            return

        try:
            import textworld
            import textworld.gym
        except ImportError as e:
            raise ImportError(
                "TextWorld is not installed. Please install it with:\n"
                "  pip install textworld alfworld"
            ) from e

        # Check game file exists
        if not self.game_file or not os.path.exists(self.game_file):
            raise FileNotFoundError(f"Game file not found: {self.game_file}")

        # Use context manager for optional lock
        lock_ctx = _TEXTWORLD_LOCK if self.use_process_lock else _nullcontext()
        with lock_ctx:
            # Use TextWorld directly with single game file
            # Request useful info from the environment
            request_infos = textworld.EnvInfos(
                won=True,
                admissible_commands=True,
                description=True,
                inventory=True,
            )

            # Register single game with TextWorld Gym
            env_id = textworld.gym.register_games(
                [self.game_file],
                request_infos,
                batch_size=1,
                asynchronous=False,
                max_episode_steps=self.max_steps,
            )

            # Create the environment
            self._env = textworld.gym.make(env_id)

    def reset(self) -> tuple[dict, dict]:
        """Reset the environment for a new episode.

        Returns:
            Tuple of (observation, info) where observation contains the initial
            game state and info contains metadata.
        """
        self._lazy_init()

        # Reset state
        self.current_step = 0
        self.done = False
        self.won = False
        self.interaction_history = []
        self.final_response = None
        self._look_used = False

        # Reset environment with optional lock to protect TextWorld's PDDL parser
        # TextWorld gym returns (obs_tuple, info_dict)
        lock_ctx = _TEXTWORLD_LOCK if self.use_process_lock else _nullcontext()
        with lock_ctx:
            obs_tuple, info = self._env.reset()

        # Extract observation text (batch_size=1)
        obs_text = obs_tuple[0] if isinstance(obs_tuple, (tuple, list)) else obs_tuple

        # Extract admissible commands (list of lists for batch, take first)
        if isinstance(info, dict):
            admissible = info.get("admissible_commands", [[]])
            self.admissible_commands = admissible[0] if admissible else []
        else:
            self.admissible_commands = []

        # Build observation dict
        observation = {
            "observation": obs_text,
            "admissible_commands": self.admissible_commands,
            "task_description": self._extract_task_description(obs_text),
        }

        info_out = {
            "task_id": self.task.get("task_id", ""),
            "task_type": self.task.get("task_type", ""),
            "max_steps": self.max_steps,
            "tools_json": get_alfworld_tools(),
        }

        return observation, info_out

    def _extract_task_description(self, obs: str) -> str:
        """Extract task description from initial observation."""
        # ALFWorld observations typically contain "Your task is to: ..."
        lines = obs.split("\n")
        for line in lines:
            if "task is to" in line.lower():
                return line.strip()
        return self.task.get("task_description", "Complete the household task.")

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Execute an action in the environment.

        Args:
            action: Action dict with format:
                {"function": {"name": "execute_action", "arguments": {"action": "..."}}}

        Returns:
            Tuple of (observation, reward, done, info)
        """
        self.current_step += 1

        # Parse action
        func_info = action.get("function", {}) if isinstance(action, dict) else {}
        tool_name = func_info.get("name", "")
        tool_args = func_info.get("arguments", {})

        # Handle string arguments (JSON parsing)
        if isinstance(tool_args, str):
            try:
                tool_args = json.loads(tool_args)
            except json.JSONDecodeError:
                tool_args = {"action": tool_args}

        # Handle finish action
        if tool_name == "finish":
            self.done = True
            self.final_response = str(tool_args.get("success", False))
            return {"observation": "Task finished."}, 0.0, True, self._get_info()

        # Handle execute_action
        if tool_name == "execute_action":
            action_str = str(tool_args.get("action", "look"))

            # Limit "look" to once per episode when ALFWORLD_LIMIT_LOOK is enabled.
            # Forces model to use ReadExperience instead of re-looking after compression.
            if action_str.strip().lower() == "look" and getattr(self, '_limit_look', False):
                if getattr(self, '_look_used', False):
                    return {
                        "observation": "[You have already used 'look' in this episode. Use ReadExperience to retrieve stored information.]",
                        "admissible_commands": self.admissible_commands,
                    }, 0.0, False, self._get_info()
                self._look_used = True

            # Execute action with optional lock (TextWorld's PDDL parser is not thread-safe)
            # TextWorld gym step returns (obs_tuple, reward_tuple, done_tuple, info_dict)
            lock_ctx = _TEXTWORLD_LOCK if self.use_process_lock else _nullcontext()
            with lock_ctx:
                obs_tuple, rewards, dones, info = self._env.step([action_str])

            # Extract results (batch_size=1)
            obs_text = obs_tuple[0] if isinstance(obs_tuple, (tuple, list)) else obs_tuple
            done = dones[0] if isinstance(dones, (tuple, list)) else dones

            # Update state
            if isinstance(info, dict):
                won_list = info.get("won", [False])
                self.won = won_list[0] if isinstance(won_list, (tuple, list)) else won_list
                admissible = info.get("admissible_commands", [[]])
                self.admissible_commands = admissible[0] if admissible else []
            else:
                self.won = False
                self.admissible_commands = []

            # Record interaction
            self.interaction_history.append({
                "step": self.current_step,
                "action": action_str,
                "observation": obs_text,
            })

            # Check if done
            self.done = done or self.current_step >= self.max_steps

            observation = {
                "observation": obs_text,
                "admissible_commands": self.admissible_commands,
            }

            return observation, 0.0, self.done, self._get_info()

        # Unknown tool - no reward for invalid actions
        return {
            "observation": f"Error: Unknown tool '{tool_name}'. Use 'execute_action' or 'finish'."
        }, 0.0, False, self._get_info()

    def _get_info(self) -> dict:
        """Get current environment info."""
        return {
            "task_id": self.task.get("task_id", ""),
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "won": self.won,
            "done": self.done,
        }

    def compute_final_reward(self) -> float:
        """Compute final reward for the episode.

        Returns:
            1.0 if task was completed successfully, 0.0 otherwise.
            (EPO uses 10.0, we divide by 10 for normalized rewards)
        """
        if self.reward_fn is not None:
            task_info = self.task.copy()
            task_info["interaction_history"] = self.interaction_history
            task_info["won"] = self.won
            try:
                result = self.reward_fn(task_info=task_info, action=self.final_response or "")
                if hasattr(result, "reward"):
                    return result.reward
                return float(result)
            except Exception:
                pass

        return 1.0 if self.won else 0.0

    def get_tools(self) -> list[dict]:
        """Get tool definitions for the agent."""
        return get_alfworld_tools()

    @staticmethod
    def from_dict(info: dict) -> "ALFWorldEnv":
        """Create environment from dictionary.

        Args:
            info: Dictionary containing task configuration.

        Returns:
            ALFWorldEnv instance.
        """
        task = {
            "task_id": info.get("task_id", ""),
            "task_type": info.get("task_type", ""),
            "game_file": info.get("game_file", ""),
            "task_description": info.get("task_description", ""),
            "data_source": info.get("data_source", "alfworld"),
        }

        return ALFWorldEnv(
            task=task,
            reward_fn=info.get("reward_fn"),
            max_steps=info.get("max_steps", 50),
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        """Return True as required by Memex ExecutionEngine."""
        return True

    def close(self):
        """Close the environment and release resources."""
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    def __del__(self):
        """Destructor to ensure resources are released."""
        self.close()
