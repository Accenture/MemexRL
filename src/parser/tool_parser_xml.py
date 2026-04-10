"""
XML format tool parser.

Format:
    <function=tool_name>
    <parameter=param1>value1</parameter>
    <parameter=param2>value2</parameter>
    </function>
"""

import json
import re
from typing import Optional

from src.tools.tool_base import ToolCall

from .tool_parser import FormatError, ParseResult, ToolParser


class XMLToolParser(ToolParser):
    """Parser for XML format tool calls.

    Format:
        <function=tool_name>
        <parameter=param1>value1</parameter>
        <parameter=param2>value2</parameter>
        </function>
    """

    def __init__(self):
        self.tool_call_begin = "<function="
        self.tool_call_end = "</function>"

    def parse_with_errors(self, model_response: str) -> ParseResult:
        """Parse tool calls and collect format errors in a single pass.

        Args:
            model_response: The raw model response text

        Returns:
            ParseResult containing tool_calls, format_errors, and metadata
        """
        format_errors: list[FormatError] = []
        tool_calls: list[ToolCall] = []

        # Check if there's any tool call attempt
        had_tool_attempt = self.tool_call_begin in model_response

        if not had_tool_attempt:
            return ParseResult(
                tool_calls=[],
                format_errors=[],
                had_tool_attempt=False,
            )

        # Check tag matching
        opens = model_response.count(self.tool_call_begin)
        closes = model_response.count(self.tool_call_end)
        if opens != closes:
            format_errors.append(FormatError(
                error_type="tag_mismatch",
                detail=f"{opens} <function= vs {closes} </function>"
            ))

        # Parse all tool calls
        pattern = re.compile(r"<function=(\w+)>(.*?)</function>", re.DOTALL)

        for match in pattern.finditer(model_response):
            function_name = match.group(1)
            block_content = match.group(2)

            # Extract parameters
            params = {}
            param_pattern = re.compile(r'<parameter=(\w+)>(.*?)</parameter>', re.DOTALL)
            for param_match in param_pattern.finditer(block_content):
                param_name = param_match.group(1)
                param_value = param_match.group(2).strip()
                # Try to parse as JSON (for numbers, lists, etc.)
                try:
                    params[param_name] = json.loads(param_value)
                except (json.JSONDecodeError, ValueError):
                    params[param_name] = param_value

            if function_name:
                tool_calls.append(ToolCall(name=function_name, arguments=params))

        # Check for unclosed <function= tags (opened but not properly closed)
        # This catches cases where regex didn't match due to malformed XML
        unclosed_pattern = re.compile(r"<function=(\w+)>(?:(?!</function>).)*$", re.DOTALL)
        if unclosed_pattern.search(model_response):
            format_errors.append(FormatError(
                error_type="unclosed_tag",
                detail="<function=...> tag opened but not closed"
            ))

        return ParseResult(
            tool_calls=tool_calls,
            format_errors=format_errors,
            had_tool_attempt=had_tool_attempt,
        )

    def get_tool_prompt(self, tools_schema: str) -> str:
        """Get the tool prompt for XML format.

        Args:
            tools_schema: JSON schema of available tools

        Returns:
            Prompt string explaining the tool call format
        """
        return f"""
# Tools

You may call one or more functions to assist with the user query.

Available tools:
{tools_schema}

# Tool Call Format
Use the following XML format to call tools:

<function=tool_name>
<parameter=param_name>param_value</parameter>
</function>

Example:
<function=search>
<parameter=query>Python programming</parameter>
</function>
"""

    def get_tool_call_example(self) -> str:
        """Get an example of the XML tool call format."""
        return '''<function=example_function_name>
<parameter=example_parameter_1>value_1</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>'''


# Convenience function
def parse_xml_tool_call(response_text: str) -> tuple[Optional[str], Optional[dict]]:
    """Extract the first XML format tool call from response.

    Args:
        response_text: The raw model response

    Returns:
        Tuple of (tool_name, parameters) or (None, None) if not found
    """
    parser = XMLToolParser()
    result = parser.parse_with_errors(response_text)
    if result.tool_calls:
        tc = result.tool_calls[0]
        return tc.name, tc.arguments
    return None, None
