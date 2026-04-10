"""
Base class and factory for tool parsers.

Tool parsers extract tool calls from model responses in various formats:
- XML: <function=tool_name><parameter=key>value</parameter></function>
- Qwen: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from src.tools.tool_base import ToolCall

if TYPE_CHECKING:
    from .tool_parser_xml import XMLToolParser
    from .tool_parser_qwen import QwenToolParser


@dataclass
class FormatError:
    """Represents a format error detected during parsing.

    Attributes:
        error_type: Type of error (e.g., 'tag_mismatch', 'invalid_json', 'not_dict', 'missing_name')
        detail: Human-readable description of the error
        step_idx: Optional step index where error occurred (set by shaper)
    """
    error_type: str
    detail: str
    step_idx: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "error_type": self.error_type,
            "detail": self.detail,
            "step_idx": self.step_idx,
        }


@dataclass
class ParseResult:
    """Result of parsing a model response for tool calls.

    Contains both successfully parsed tool calls and any format errors
    detected during parsing. This allows downstream code (e.g., reward shapers)
    to use the errors directly without re-parsing.

    Note: thought and raw_response are NOT stored here to avoid redundancy.
    The full response is available in Step.model_response.

    Attributes:
        tool_calls: List of successfully parsed ToolCall objects
        format_errors: List of FormatError objects detected during parsing
        had_tool_attempt: Whether the response attempted to call a tool
                         (used for computing error ratios)
    """
    tool_calls: list[ToolCall] = field(default_factory=list)
    format_errors: list[FormatError] = field(default_factory=list)
    had_tool_attempt: bool = False

    def to_dict(self) -> dict:
        return {
            "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in self.tool_calls],
            "format_errors": [e.to_dict() for e in self.format_errors],
            "had_tool_attempt": self.had_tool_attempt,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ParseResult":
        return cls(
            tool_calls=[ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in data.get("tool_calls", [])],
            format_errors=[FormatError(**e) for e in data.get("format_errors", [])],
            had_tool_attempt=data.get("had_tool_attempt", False),
        )

    @property
    def has_errors(self) -> bool:
        """Check if any format errors were detected."""
        return len(self.format_errors) > 0

    @property
    def first_tool_call(self) -> Optional[ToolCall]:
        """Get the first tool call, or None if no tool calls parsed."""
        return self.tool_calls[0] if self.tool_calls else None


class ToolParser(ABC):
    """Base class for tool parsers."""

    @abstractmethod
    def parse_with_errors(self, model_response: str) -> ParseResult:
        """Parse tool calls and collect format errors.

        Args:
            model_response: The raw model response text

        Returns:
            ParseResult containing tool_calls, format_errors, and had_tool_attempt
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_tool_prompt(self, tools_schema: str) -> str:
        """Get the tool prompt for the model.

        Args:
            tools_schema: JSON schema of available tools

        Returns:
            Prompt string explaining the tool call format
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_tool_call_example(self) -> str:
        """Get an example of the tool call format.

        Used to provide hints when no tool call is detected in agent response.

        Returns:
            Example string showing the correct tool call format
        """
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def get_parser(cls, tokenizer) -> "ToolParser":
        """Factory method to get the appropriate tool parser based on tokenizer.

        Args:
            tokenizer: The tokenizer to use with the parser

        Returns:
            ToolParser: An instance of the requested parser

        Raises:
            ValueError: If no suitable parser found for the tokenizer
        """
        # Import here to avoid circular imports
        from .tool_parser_qwen import QwenToolParser

        # Determine parser type based on tokenizer name or path
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()

            if "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                return QwenToolParser()

        raise ValueError(f"No tool parser found for {tokenizer.name_or_path}")

    @classmethod
    def get_parser_by_name(cls, parser_name: str) -> "ToolParser":
        """Get parser by name.

        Args:
            parser_name: One of 'xml', 'qwen'

        Returns:
            ToolParser instance

        Raises:
            ValueError: If parser name not recognized
        """
        from .tool_parser_xml import XMLToolParser
        from .tool_parser_qwen import QwenToolParser

        parsers = {
            "xml": XMLToolParser,
            "qwen": QwenToolParser,
        }

        if parser_name not in parsers:
            raise ValueError(f"Unknown parser: {parser_name}. Available: {list(parsers.keys())}")

        return parsers[parser_name]()


def parse_tool_call(response_text: str) -> tuple[Optional[str], Optional[dict]]:
    """Extract the first tool call from a response, auto-detecting format.

    This is a convenience function that tries all formats.

    Args:
        response_text: The raw model response

    Returns:
        Tuple of (tool_name, parameters) or (None, None) if no tool call found
    """
    from .tool_parser_xml import parse_xml_tool_call
    from .tool_parser_qwen import parse_qwen_tool_call

    # Try XML format first (most common in agent environments)
    tool_name, params = parse_xml_tool_call(response_text)
    if tool_name:
        return tool_name, params

    # Try Qwen format
    tool_name, params = parse_qwen_tool_call(response_text)
    if tool_name:
        return tool_name, params

    return None, None
