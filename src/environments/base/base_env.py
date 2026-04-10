from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from src.parser.tool_parser import ParseResult


class BaseEnv(ABC):
    @property
    def idx(self) -> Any:
        """The index or identifier of the environment, often used within a batch.

        Returns:
            The assigned index or identifier, or None if not set.
        """
        # Return the stored _idx value if it exists, otherwise return None.
        return getattr(self, "_idx", None)

    @idx.setter
    def idx(self, value: Any):
        """Set the environment index or identifier.

        This allows assigning an index or identifier (e.g., its position in a batch)
        to the environment instance after it has been created.

        Example:
            env = MyEnvSubclass()  # Assuming MyEnvSubclass inherits from BaseEnv
            env.idx = 5            # Set the index externally

        Args:
            value: The index or identifier to set for this environment.
        """
        self._idx = value

    @abstractmethod
    def reset(self) -> tuple[dict, dict]:
        """Standard Gym reset method. Resets the environment to an initial state.

        Returns:
            A tuple typically containing the initial observation and auxiliary info.
        """
        pass

    @abstractmethod
    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Standard Gym step method. Executes one time step within the environment.

        Args:
            action: An action provided by the agent.

        Returns:
            A tuple containing (observation, reward, done, info).
        """
        pass

    def format_action(self, parse_result: "ParseResult") -> Any:
        """Convert ParseResult to environment-specific action format.

        Subclasses should override this for different formats.
        Default returns OpenAI-style format used by BrowseComp/UltraHorizon.

        Args:
            parse_result: Unified ParseResult from agent

        Returns:
            Environment-specific action format
        """
        if not parse_result or not parse_result.tool_calls:
            return None
        tc = parse_result.tool_calls[0]
        return {"function": {"name": tc.name, "arguments": tc.arguments}}

    def close(self):
        """Standard Gym close method. Performs any necessary cleanup."""
        return

    @staticmethod
    @abstractmethod
    def from_dict(info: dict) -> "BaseEnv":
        """Creates an environment instance from a dictionary.

        This method should be implemented by concrete subclasses to handle
        environment-specific initialization from serialized data.

        Args:
            info: A dictionary containing the necessary information to initialize the environment.

        Returns:
            An instance of the specific BaseEnv subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        # BaseEnv is abstract, subclasses must implement this factory method.
        raise NotImplementedError("Subclasses must implement the 'from_dict' static method.")

    @staticmethod
    def is_multithread_safe() -> bool:
        return True

    # ========== Reward Computation Support ==========

    def compute_final_reward(self) -> float:
        """
        Compute final reward using the full interaction history.

        This method is called by the execution engine after the trajectory completes.
        Subclasses should either:
        1. Override this method for custom reward logic
        2. Set self.reward_fn to use the default implementation

        The default implementation:
        - Calls self.reward_fn with task_info and final_response
        - Includes lossless_stats (num_retrievals, num_compressions, context_penalty) if available
        - Includes interaction_history if available

        Returns:
            float: The final reward score
        """
        import random

        # Get reward function
        reward_fn = getattr(self, 'reward_fn', None)
        task = getattr(self, 'task', None) or getattr(self, 'task_info', None)

        if not reward_fn or not task:
            return 0.0

        # Build task info with all available context
        task_info = task.copy() if isinstance(task, dict) else {}

        # Add interaction history if available
        if hasattr(self, 'interaction_history'):
            task_info["interaction_history"] = self.interaction_history

        # Add compression records for gradient flow (if compression was used)
        if hasattr(self, "get_processor_log_probs"):
            compression_records = self.get_processor_log_probs()
            if compression_records:
                task_info["compression_records"] = compression_records

        # Add lossless compression statistics and context penalty (for reward shaping)
        # These are set by agent_execution_engine before calling compute_final_reward
        if hasattr(self, 'lossless_stats') and self.lossless_stats:
            task_info["num_retrievals"] = self.lossless_stats.get("num_retrievals", 0)
            task_info["num_compressions"] = self.lossless_stats.get("num_compressions", 0)
            task_info["context_penalty"] = self.lossless_stats.get("context_penalty", 0.0)
            if random.random() < 0.1:  # Log 10% of the time
                print(f"[LOSSLESS_STATS] Episode completed with {task_info['num_retrievals']} retrievals, "
                      f"{task_info['num_compressions']} compressions, context_penalty={task_info['context_penalty']:.4f}")

        # Fallback: Direct access to agent (if lossless_stats not set)
        elif hasattr(self, 'agent') and self.agent and hasattr(self.agent, 'trajectory') and self.agent.trajectory:
            num_retrievals = 0
            num_compressions = 0
            for step in self.agent.trajectory.steps:
                num_retrievals += getattr(step, 'num_retrievals_in_step', 0)
                num_compressions += getattr(step, 'num_compressions_in_step', 0)

            if num_retrievals > 0 or num_compressions > 0:
                task_info["num_retrievals"] = num_retrievals
                task_info["num_compressions"] = num_compressions
                if random.random() < 0.1:
                    print(f"[LOSSLESS_STATS] Episode completed with {num_retrievals} retrievals, {num_compressions} compressions")

        # Add final_result from environment's built-in judge (for UltraHorizon etc.)
        # This contains the judge's evaluation result from commit_final_result()
        final_result = getattr(self, 'final_result', None)
        if final_result:
            task_info["final_result"] = final_result

        # Get final response
        final_response = getattr(self, 'final_response', None) or getattr(self, 'final_answer', None) or ""

        try:
            reward_output = reward_fn(task_info=task_info, action=final_response)

            # Store task_info for later retrieval (e.g., for trajectory logging)
            self.task_info_with_judge = task_info

            # Handle RewardOutput or float
            if hasattr(reward_output, 'reward'):
                return reward_output.reward
            return float(reward_output)
        except Exception as e:
            print(f"[BASE_ENV_ERROR] Error in compute_final_reward: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
