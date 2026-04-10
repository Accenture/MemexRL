"""
Unified Tool Agent - Base class for all tool-calling agents.

This module provides:
1. ToolAgent - Base class without memory support
2. ToolAgentWithMemory - Base class with memory support (uses MemoryAgentMixin)

Usage:
    # Simple usage (no memory)
    agent = ToolAgent(
        system_prompt="You are a helpful assistant.",
        tool_call_format="xml",
    )

    # With memory enabled - use ToolAgentWithMemory
    agent = ToolAgentWithMemory(
        system_prompt="You are a helpful assistant.",
        compression_mode="lossless_db",
    )

    # For environment-specific agents, inherit and override hooks
    class MyEnvAgent(ToolAgent):
        def _format_first_observation(self, obs: str) -> str:
            return f"Task: {obs}"
"""

from typing import Any, Callable, Optional

from src.agents.agent import BaseAgent, Step, Trajectory
from src.agents.memory import (
    MemoryAgentMixin,
    get_memory_tools_prompt_full,
    get_memory_tools_prompt_compress_only,
    get_memory_tools_prompt_rag,
)
from src.parser.tool_parser import ParseResult
from src.parser.tool_parser_qwen import QwenToolParser
from src.parser.tool_parser_xml import XMLToolParser


class ToolAgent(BaseAgent):
    """
    Base class for tool-calling agents (without memory support).

    Features:
    - Message history management
    - Trajectory recording
    - Response parsing (XML/Qwen formats)
    - Hookable methods for customization

    Hookable Methods (override in subclasses):
    - _format_observation(): Format raw observation
    - _format_first_observation(): Special handling for first observation
    - _get_extra_observation_info(): Inject extra info (remaining steps, etc.)
    """

    def __init__(
        self,
        system_prompt: str,
        tool_call_format: str = "xml",
        agent_name: str = "agent",
        model_name: Optional[str] = None,
        observation_formatter: Optional[Callable[[Any], str]] = None,
    ):
        """
        Initialize the agent.

        Args:
            system_prompt: System prompt for the agent
            tool_call_format: "xml" or "qwen"
            agent_name: Name for trajectory logging
            model_name: Model name (for auto-detecting format)
            observation_formatter: Optional custom observation formatter
        """
        self._base_system_prompt = system_prompt
        self.system_prompt = system_prompt
        self.tool_call_format = tool_call_format
        self.agent_name = agent_name
        self.model_name = model_name
        self._observation_formatter = observation_formatter

        # Auto-detect format from model name
        if model_name and "qwen" in model_name.lower():
            self.tool_call_format = "qwen"

        # Create parser instance for parse_with_errors
        self._tool_parser = QwenToolParser() if self.tool_call_format == "qwen" else XMLToolParser()

        # State
        self.messages: list[dict] = []
        self._trajectory = Trajectory(name=agent_name)
        self._current_step: Optional[Step] = None
        self.step = 0

        self.reset()

    # =========================================================================
    # Core Interface (BaseAgent implementation)
    # =========================================================================

    def reset(self):
        """Reset agent state for a new episode."""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._trajectory = Trajectory(name=self.agent_name)
        self._current_step = None
        self.step = 0

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        """Get current message history."""
        return self.messages

    @property
    def trajectory(self) -> Trajectory:
        """Get current trajectory."""
        return self._trajectory

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """
        Update agent state after environment step.

        Handles:
        - First observation formatting
        - Extra info injection (remaining steps, etc.)
        - Trajectory recording
        """
        # Format observation
        obs_text = self._format_observation(observation)

        # First observation special handling
        # For first observation (task description): use plain content
        # For subsequent observations (tool outputs): wrap in <tool_response> tags
        # This preserves <think> content in history via Qwen3 chat_template's last_query_index logic
        # (Qwen3 template skips last_query_index update for user messages wrapped in <tool_response>)
        is_first_observation = not self._trajectory.steps and self._current_step is None

        # Inject extra info (remaining steps, token warnings, etc.) BEFORE wrapping
        # This ensures the message ends with </tool_response> for Qwen3 chat_template compatibility
        extra_info = self._get_extra_observation_info(info)
        if extra_info:
            obs_text = obs_text + extra_info

        if is_first_observation:
            obs_text = self._format_first_observation(obs_text)
        else:
            # Wrap tool output in <tool_response> tags for Qwen3 compatibility
            # IMPORTANT: extra_info must be inside the tags, so message endswith('</tool_response>')
            # Qwen3's chat_template checks startswith/endswith to identify tool responses
            # and preserve <think> content in history via last_query_index logic
            obs_text = f"<tool_response>\n{obs_text}\n</tool_response>"

        # Add to messages (always use "user" role for OpenAI/Azure API compatibility)
        self.messages.append({"role": "user", "content": obs_text})

        # Update current step if exists
        if self._current_step:
            self._current_step.observation = obs_text
            self._current_step.reward = reward
            self._current_step.done = done
            self._current_step.chat_completions = self.messages.copy()
            self._trajectory.steps.append(self._current_step)
            self._current_step = None

    def update_from_model(self, response: str, **kwargs) -> ParseResult:
        """Update agent state after model response.

        Returns:
            ParseResult containing tool_calls and format_errors.
            Engine will call env.format_action(parse_result) to get action.
        """
        # Single parse pass - get ParseResult with tool_calls and errors
        parse_result = self._parse_response_with_errors(response)

        # Add assistant message
        self.messages.append({"role": "assistant", "content": response})

        # Create step with parse_result (action will be set by engine)
        # Note: thought is not stored separately - it can be derived from model_response if needed
        self._current_step = Step(
            model_response=response,
            parse_result=parse_result,
        )

        self.step += 1
        return parse_result

    def get_current_state(self) -> Optional[Step]:
        """Get current step state."""
        if self._trajectory.steps:
            return self._trajectory.steps[-1]
        return self._current_step

    # =========================================================================
    # Hookable Methods (override in subclasses)
    # =========================================================================

    def _parse_response_with_errors(self, response: str) -> ParseResult:
        """
        Parse model response and collect format errors.

        This method uses the parser's parse_with_errors() to get both
        tool calls and format errors in a single pass. The result is
        stored in Step.parse_result for use by reward shapers.

        Args:
            response: Raw model response

        Returns:
            ParseResult containing tool_calls, format_errors, and metadata
        """
        return self._tool_parser.parse_with_errors(response)

    def _format_observation(self, observation: Any) -> str:
        """
        Format raw observation for the agent.

        Override this for environment-specific formatting.

        Args:
            observation: Raw observation from environment

        Returns:
            Formatted observation string
        """
        # Use custom formatter if provided
        if self._observation_formatter is not None:
            return self._observation_formatter(observation)

        # Default: convert to string
        if isinstance(observation, str):
            return observation
        return str(observation)

    def _format_first_observation(self, observation: str) -> str:
        """
        Format the first observation (task description).

        Override this for environment-specific first observation handling.
        Example: SWE agent wraps with user_prompt_template.

        Args:
            observation: Formatted observation string

        Returns:
            Formatted first observation
        """
        return observation

    def _get_extra_observation_info(self, info: dict) -> str:
        """
        Get extra info to append to observations.

        Override this to inject remaining steps, token warnings, etc.

        Args:
            info: Info dict from environment

        Returns:
            Extra info string to append (empty string if none)
        """
        return ""


class ToolAgentWithMemory(MemoryAgentMixin, ToolAgent):
    """
    Base class for tool-calling agents WITH memory support.

    Uses multiple inheritance from MemoryAgentMixin for memory functionality.
    All memory methods (is_memory_tool, execute_memory_tool, etc.) are
    inherited from MemoryAgentMixin.

    Usage:
        agent = ToolAgentWithMemory(
            system_prompt="You are a helpful assistant.",
            compression_mode="lossless_db",
            context_length_threshold=8000,
        )
    """

    def __init__(
        self,
        system_prompt: str,
        tool_call_format: str = "xml",
        agent_name: str = "agent",
        model_name: Optional[str] = None,
        observation_formatter: Optional[Callable[[Any], str]] = None,
        # Memory parameters
        compression_mode: str = "lossless_db",
        context_db: Any = None,
        db_path: str = "experience.sqlite",
        context_length_threshold: int = 16000,
        auto_compress_prompt: bool = True,
        disable_retrieve: bool = False,
        max_summary_tokens: int = 0,
        # Tool format suffix (appended after memory tools)
        tool_format_suffix: Optional[str] = None,
    ):
        """
        Initialize the agent with memory support.

        Args:
            system_prompt: Base system prompt (memory tools will be appended)
            tool_call_format: "xml" or "qwen"
            agent_name: Name for trajectory logging
            model_name: Model name (for auto-detecting format)
            observation_formatter: Optional custom observation formatter
            compression_mode: "lossless_db", "lossy", "rag", "none"
            context_db: Database instance (optional)
            db_path: Path for database file
            context_length_threshold: Token threshold for compression warning
            auto_compress_prompt: Whether to inject compression warnings
            disable_retrieve: If True, only compress (no retrieve)
            tool_format_suffix: Tool format instructions (appended after memory tools)
        """
        # Store base prompt
        self._base_system_prompt = system_prompt

        # Auto-detect format from model name
        if model_name and "qwen" in model_name.lower():
            tool_call_format = "qwen"

        # Initialize memory BEFORE calling super().__init__()
        # This sets up compression_mode, context_db, etc.
        self.init_memory(
            compression_mode=compression_mode,
            context_db=context_db,
            db_path=db_path,
            context_length_threshold=context_length_threshold,
            auto_compress_prompt=auto_compress_prompt,
            disable_retrieve=disable_retrieve,
            max_summary_tokens=max_summary_tokens,
        )

        # Build system prompt: base + memory_tools + tool_format_suffix
        if compression_mode in ("lossless_db", "lossy", "rag"):
            if compression_mode == "rag":
                memory_prompt = get_memory_tools_prompt_rag(tool_call_format)
            elif disable_retrieve:
                memory_prompt = get_memory_tools_prompt_compress_only(tool_call_format)
            else:
                memory_prompt = get_memory_tools_prompt_full(tool_call_format)
            full_system_prompt = system_prompt + memory_prompt
            if tool_format_suffix:
                full_system_prompt += tool_format_suffix
        else:
            full_system_prompt = system_prompt
            if tool_format_suffix:
                full_system_prompt += tool_format_suffix

        # Initialize base agent
        super().__init__(
            system_prompt=full_system_prompt,
            tool_call_format=tool_call_format,
            agent_name=agent_name,
            model_name=model_name,
            observation_formatter=observation_formatter,
        )

    def reset(self):
        """Reset agent state for a new episode."""
        super().reset()
        self.reset_memory_stats()

    # NOTE: update_from_env is inherited from MemoryAgentMixin, which:
    # 1. Injects context status into observation
    # 2. Calls super().update_from_env() (ToolAgent's version)
    # 3. Handles finalize_segments on done
