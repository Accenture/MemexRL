"""
Shared utilities for tool parsing.
"""

import json


def fix_json_control_chars(json_str: str) -> str:
    """Fix common JSON formatting issues in tool calls.

    Handles unescaped control characters that break JSON parsing.
    """
    try:
        json.loads(json_str)
        return json_str
    except (json.JSONDecodeError, ValueError):
        pass

    placeholders = {
        '\\n': '\x00NEWLINE\x00',
        '\\t': '\x00TAB\x00',
        '\\r': '\x00CR\x00',
        '\\b': '\x00BS\x00',
        '\\f': '\x00FF\x00',
        '\\"': '\x00QUOTE\x00',
        '\\\\': '\x00BACKSLASH\x00',
    }

    result = json_str
    for escaped, placeholder in placeholders.items():
        result = result.replace(escaped, placeholder)

    result = result.replace('\n', '\\n')
    result = result.replace('\t', '\\t')
    result = result.replace('\r', '\\r')
    result = result.replace('\b', '\\b')
    result = result.replace('\f', '\\f')

    for escaped, placeholder in placeholders.items():
        result = result.replace(placeholder, escaped)

    return result


def load_tool_call_json(json_text: str) -> dict | None:
    """Load JSON from tool call, handling common formatting issues.

    Extracts JSON object from text, handles extra braces, etc.
    """
    candidate = json_text.strip()
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]

    # Handle extra closing braces
    open_count = candidate.count("{")
    close_count = candidate.count("}")
    while close_count > open_count and candidate.endswith("}"):
        candidate = candidate[:-1].rstrip()
        close_count -= 1

    try:
        data = json.loads(candidate)
    except Exception:
        return None

    return data if isinstance(data, dict) else None


def validate_tool_call_braces(json_text: str) -> tuple[bool, str | None]:
    """Validate JSON structure for a tool_call JSON blob.

    Returns:
        Tuple of (ok, error_message)
    """
    candidate = json_text.strip()
    if "{" in candidate and "}" in candidate:
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]

    try:
        data = json.loads(candidate)
        if not isinstance(data, dict):
            return False, "JSON is not a dictionary object"
        return True, None
    except json.JSONDecodeError as e:
        return False, f"JSON parse error: {str(e)}"
    except Exception as e:
        return False, f"validation error: {str(e)}"


# Test messages for parser testing
PARSER_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Search for information about Python."},
    {"role": "assistant", "content": "I'll search for that.", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Python programming"}'}}]},
    {"role": "user", "content": "What about Java?"},
    {"role": "assistant", "content": "Let me search for Java information.", "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Java programming"}'}}]},
]
