"""
Parser module for chat templates and tool call parsing.

Chat Template Parsers:
    Convert message lists to model input format.

Tool Parsers:
    Extract tool calls from model responses.
    Supported formats: XML, Qwen

Usage:
    # Chat template parsing
    from src.parser import ChatTemplateParser
    parser = ChatTemplateParser.get_parser(tokenizer)
    prompt = parser.parse(messages)

    # Tool call parsing
    from src.parser import parse_tool_call, ToolParser
    tool_name, params = parse_tool_call(response)  # Auto-detect format

    # Or use specific parsers
    from src.parser import XMLToolParser, QwenToolParser
    parser = XMLToolParser()
    result = parser.parse_with_errors(response)
    tool_calls = result.tool_calls
"""

# Chat template parsers
from src.parser.chat_template_parser import (
    ChatTemplateParser,
    LlamaChatTemplateParser,
    QwenChatTemplateParser,
)

# Tool parser base class and factory
from src.parser.tool_parser import ToolParser, parse_tool_call, FormatError, ParseResult

# XML format tool parser
from src.parser.tool_parser_xml import (
    XMLToolParser,
    parse_xml_tool_call,
)

# Qwen format tool parser
from src.parser.tool_parser_qwen import (
    QwenToolParser,
    parse_qwen_tool_call,
)

# Utility functions
from src.parser.utils import (
    fix_json_control_chars,
    load_tool_call_json,
    validate_tool_call_braces,
    PARSER_TEST_MESSAGES,
)


__all__ = [
    # Chat template parsers
    "ChatTemplateParser",
    "QwenChatTemplateParser",
    "LlamaChatTemplateParser",
    # Tool parser base
    "ToolParser",
    "parse_tool_call",
    "FormatError",
    "ParseResult",
    # XML format
    "XMLToolParser",
    "parse_xml_tool_call",
    # Qwen format
    "QwenToolParser",
    "parse_qwen_tool_call",
    # Utils
    "fix_json_control_chars",
    "load_tool_call_json",
    "validate_tool_call_braces",
    "PARSER_TEST_MESSAGES",
]


# Registry for backward compatibility
PARSER_REGISTRY = {
    "xml": XMLToolParser,
    "qwen": QwenToolParser,
}


def get_tool_parser(parser_name: str) -> type[ToolParser]:
    """Get tool parser class by name.

    Args:
        parser_name: One of 'xml', 'qwen'

    Returns:
        ToolParser class
    """
    assert parser_name in PARSER_REGISTRY, f"Tool parser {parser_name} not found in {list(PARSER_REGISTRY.keys())}"
    return PARSER_REGISTRY[parser_name]
