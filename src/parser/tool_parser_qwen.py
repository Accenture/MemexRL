"""
Qwen format tool parser.

Format:
    <tool_call>
    {"name": "tool_name", "arguments": {"param1": "value1", "param2": "value2"}}
    </tool_call>
"""

import json
from typing import Optional

from src.tools.tool_base import ToolCall

from .tool_parser import FormatError, ParseResult, ToolParser
from .utils import fix_json_control_chars, load_tool_call_json, validate_tool_call_braces


class QwenToolParser(ToolParser):
    """Parser for Qwen format tool calls.

    Format:
        <tool_call>
        {"name": "tool_name", "arguments": {"param1": "value1"}}
        </tool_call>
    """

    def __init__(self):
        self.tool_call_begin = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_output_begin = "<tool_response>"
        self.tool_output_end = "</tool_response>"

    def _strip_think_block(self, text: str) -> str:
        """Remove content before </think> to avoid parsing tool_call mentions in thinking.

        This prevents false positives when the model mentions <tool_call> inside <think> block.
        """
        think_end_tag = "</think>"
        think_end_idx = text.find(think_end_tag)
        if think_end_idx != -1:
            return text[think_end_idx + len(think_end_tag):]
        return text

    def parse_with_errors(self, model_response: str) -> ParseResult:
        """Parse tool calls and collect format errors in a single pass.

        This is the preferred method for parsing when you need both:
        - Successfully parsed tool calls
        - Format errors for penalty calculation

        The method:
        1. Strips <think> block to avoid false positives
        2. Checks tag matching (<tool_call> vs </tool_call>)
        3. Parses each tool call JSON, recording errors along the way
        4. Returns ParseResult with both tool_calls and format_errors

        Args:
            model_response: The raw model response text

        Returns:
            ParseResult containing tool_calls, format_errors, and metadata
        """
        format_errors: list[FormatError] = []
        tool_calls: list[ToolCall] = []

        # Skip <think> block to avoid false positives when model mentions <tool_call> in thinking
        search_region = self._strip_think_block(model_response)

        # Check if there's any tool call attempt
        had_tool_attempt = self.tool_call_begin in search_region

        if not had_tool_attempt:
            return ParseResult(
                tool_calls=[],
                format_errors=[],
                had_tool_attempt=False,
            )

        # Check tag matching
        opens = search_region.count(self.tool_call_begin)
        closes = search_region.count(self.tool_call_end)
        if opens != closes:
            format_errors.append(FormatError(
                error_type="tag_mismatch",
                detail=f"{opens} <tool_call> vs {closes} </tool_call>"
            ))

        # Parse each tool call block
        remaining = search_region
        while self.tool_call_begin in remaining:
            start = remaining.find(self.tool_call_begin) + len(self.tool_call_begin)
            end = remaining.find(self.tool_call_end)

            if end == -1:
                # Unclosed tag - already captured in tag_mismatch
                break

            json_content = remaining[start:end].strip()
            json_content = fix_json_control_chars(json_content)

            # Validate brace matching
            ok, brace_error = validate_tool_call_braces(json_content)
            if not ok:
                format_errors.append(FormatError(
                    error_type="invalid_json",
                    detail=f"Brace mismatch: {brace_error}"
                ))
                remaining = remaining[end + len(self.tool_call_end):]
                continue

            # Try to parse JSON
            try:
                call_data = json.loads(json_content)
            except json.JSONDecodeError as e:
                # Try robust parsing
                call_data = load_tool_call_json(json_content)
                if call_data is None:
                    format_errors.append(FormatError(
                        error_type="invalid_json",
                        detail=str(e)[:100]
                    ))
                    remaining = remaining[end + len(self.tool_call_end):]
                    continue

            # Validate it's a dict
            if not isinstance(call_data, dict):
                format_errors.append(FormatError(
                    error_type="not_dict",
                    detail=f"Expected dict, got {type(call_data).__name__}"
                ))
                remaining = remaining[end + len(self.tool_call_end):]
                continue

            # Check required fields
            if "name" not in call_data:
                format_errors.append(FormatError(
                    error_type="missing_name",
                    detail="JSON missing 'name' field"
                ))

            if "arguments" not in call_data:
                format_errors.append(FormatError(
                    error_type="missing_arguments",
                    detail="JSON missing 'arguments' field"
                ))

            # Extract name and arguments
            name = call_data.get("name", "")
            arguments = call_data.get("arguments", {})

            # Ensure arguments is a dict
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (json.JSONDecodeError, ValueError):
                    arguments = {}

            if not isinstance(arguments, dict):
                arguments = {}

            # Add successfully parsed tool call
            if name and arguments:  # Only add if we have both name and non-empty arguments
                tool_calls.append(ToolCall(name=name, arguments=arguments))

            remaining = remaining[end + len(self.tool_call_end):]

        return ParseResult(
            tool_calls=tool_calls,
            format_errors=format_errors,
            had_tool_attempt=had_tool_attempt,
        )

    def get_tool_prompt(self, tools_schema: str) -> str:
        """Get the tool prompt for Qwen format.

        Args:
            tools_schema: JSON schema of available tools

        Returns:
            Prompt string explaining the tool call format
        """
        return f"""

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
""".rstrip()

    def get_tool_call_example(self) -> str:
        """Get an example of the Qwen tool call format."""
        return '''<tool_call>
{"name": "example_function_name", "arguments": {"example_parameter_1": "value_1", "example_parameter_2": "This is the value for the second parameter that can span multiple lines"}}
</tool_call>'''


# Convenience function
def parse_qwen_tool_call(response_text: str) -> tuple[Optional[str], Optional[dict]]:
    """Extract the first Qwen format tool call from response.

    Args:
        response_text: The raw model response

    Returns:
        Tuple of (tool_name, parameters) or (None, None) if not found
    """
    parser = QwenToolParser()
    result = parser.parse_with_errors(response_text)
    if result.tool_calls:
        tc = result.tool_calls[0]
        return tc.name, tc.arguments
    return None, None
