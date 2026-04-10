"""
ALFWorld Agent for Memex.

This module provides the agent class for interacting with ALFWorld environments.
Supports both baseline mode and memory-enhanced mode.
"""

import json
from typing import Any, Optional

from src.agents.tool_agent import ToolAgent, ToolAgentWithMemory


REASONING_INSTRUCTION = """
<IMPORTANT>
- You MUST provide consise reasoning BEFORE every action.
</IMPORTANT>
"""

# System prompt for ALFWorld agent
ALFWORLD_SYSTEM_PROMPT_BASE = """You are an intelligent agent in a household environment (ALFWorld).

# Your Goal
Complete household tasks by navigating rooms, interacting with objects, and manipulating them appropriately.

# Available Tools

1. **execute_action** - Execute an action in the environment
   - Parameter: action (string) - The action to execute in natural language
   - Returns: Observation of the result

2. **finish** - Indicate task completion
   - Parameter: success (boolean) - Whether the task was completed successfully

# Action Format
Actions are natural language commands. Common actions include:
- Navigation: "go to desk 1", "go to drawer 2", "go to fridge 1"
- Picking up: "pick up book 1", "pick up apple 1"
- Placing: "put book 1 in/on desk 1", "put apple 1 in fridge 1"
- Opening/Closing: "open drawer 1", "close fridge 1"
- Using: "use lamp 1" (turn on), "cool apple 1 with fridge 1", "heat potato 1 with microwave 1"
- Looking: "examine desk 1", "look"

# Tips
1. Pay attention to the task description - it tells you exactly what to do
2. Use "go to [location]" to navigate to objects
3. You must be at a location before you can interact with it
4. Some containers (drawers, fridges) need to be opened before you can see/use their contents
5. The "admissible_commands" show valid actions - use them as hints
6. Be systematic: find the object, pick it up, find the destination, put it down

##############################################################################
#                         MANDATORY REQUIREMENTS                              #
##############################################################################

>>> REQUIREMENT 1: YOU MUST CALL A TOOL IN EVERY RESPONSE <<<
- Every response MUST contain a tool call
- Plain text responses WITHOUT a tool call will be REJECTED

>>> REQUIREMENT 2: USE execute_action FOR ALL INTERACTIONS <<<
- Use execute_action to perform actions in the environment
- Only use finish when you have completed the task

""" + REASONING_INSTRUCTION

ALFWORLD_TOOL_CALL_FORMAT_XML = """# Tool Call Format
Use the following XML format to call tools:

<function=tool_name>
<parameter=param_name>param_value</parameter>
</function>

Examples:
<function=execute_action>
<parameter=action>go to desk 1</parameter>
</function>

<function=execute_action>
<parameter=action>pick up book 1</parameter>
</function>

<function=finish>
<parameter=success>true</parameter>
</function>
"""

ALFWORLD_TOOL_CALL_FORMAT_QWEN = """# Tool Call Format
Use the following JSON format inside <tool_call> tags:

<tool_call>
{"name": "tool_name", "arguments": {"param1": "value1"}}
</tool_call>

Examples:
<tool_call>
{"name": "execute_action", "arguments": {"action": "go to desk 1"}}
</tool_call>

<tool_call>
{"name": "execute_action", "arguments": {"action": "pick up book 1"}}
</tool_call>

<tool_call>
{"name": "finish", "arguments": {"success": true}}
</tool_call>
"""

# =============================================================================
# ALFWorld-Specific Memory Guidance
# =============================================================================

ALFWORLD_MEMORY_GUIDANCE = """
##############################################################################
#                    MEMORY MANAGEMENT FOR ALFWORLD                          #
##############################################################################

CRITICAL: How to use CompressExperience effectively in ALFWorld:

1. **summary** must contain ONLY short descriptions and index map. NEVER put raw IDs in summary!
   - BAD: "ctx_cabinet_001 - cabinet_bar__minus_00_dot_36_bar__plus_00_dot_38..."
   - BAD: "Locations: countertop_bar__minus_00_dot_28_bar__plus_00_dot_79..."
   - GOOD: "ctx_locations - All 20 location IDs\nctx_progress - Task progress"
   Summary is truncated to save space. Any IDs in summary WILL BE LOST.

2. **db_blocks** is the ONLY safe place to store exact IDs:
   - Store ALL location IDs in db_blocks (e.g., db_index="ctx_locations")
   - Store object IDs you've found (e.g., db_index="ctx_objects")
   - These IDs are required to interact with objects in future actions

3. **After compression**, you MUST call ReadExperience(db_index) to get IDs back:
   - Summary does NOT contain IDs (they are truncated)
   - The ONLY way to get exact IDs is ReadExperience

3. **After compression**, call ReadExperience(db_index) to retrieve precise IDs
   - Don't try to remember IDs from memory - they are deleted after compression
   - Always retrieve before taking actions that need specific location names

ALFWORLD-SPECIFIC EXAMPLE:

Step 1 - Compress (store IDs in db_blocks, NOT in summary):
<tool_call>
{"name": "CompressExperience", "arguments": {
  "summary": "Index map:\\n- ctx_locations - All room location IDs\\n- ctx_progress - Task status\\nStatus: Found butterknife, need to clean and place on table",
  "db_blocks": [
    {"db_index": "ctx_locations", "db_content": "countertop_bar__minus_00_dot_28_bar__plus_00_dot_79_bar__plus_01_dot_93\\ndrawer_bar__minus_00_dot_33_bar__plus_00_dot_32_bar__plus_01_dot_72\\nsinkbasin_bar__plus_01_dot_13_bar__plus_00_dot_00_bar__minus_01_dot_33\\ndiningtable_bar__plus_01_dot_02_bar__plus_00_dot_00_bar__plus_01_dot_61"},
    {"db_index": "ctx_progress", "db_content": "Task: put clean butterknife on diningtable\\nFound: butterknife at countertop\\nInventory: butterknife_bar__minus_00_dot_77_bar__plus_00_dot_90_bar__minus_01_dot_68\\nNext: go to sinkbasin to clean, then to diningtable"}
  ]
}}
</tool_call>

Step 2 - After compression, retrieve IDs before navigating:
<tool_call>
{"name": "ReadExperience", "arguments": {"db_index": "ctx_locations"}}
</tool_call>
→ Returns: "countertop_bar__...\ndrawer_bar__...\nsinkbasin_bar__...\ndiningtable_bar__..."
Now you can use these IDs: execute_action("go to sinkbasin_bar__plus_01_dot_13...")

This returns the exact cabinet IDs so you don't re-check the same locations.
"""

ALFWORLD_MEMORY_GUIDANCE_XML = """
##############################################################################
#                    MEMORY MANAGEMENT FOR ALFWORLD                          #
##############################################################################

CRITICAL: How to use CompressExperience effectively in ALFWorld:

1. **summary** should contain SHORT descriptions only:
   - BAD: "ctx_cabinet_001 - Checked cabinet_bar__minus_00_dot_36_bar__plus_00_dot_38..."
   - GOOD: "ctx_search_001 - List of 3 checked locations (all empty)"

2. **db_blocks** should store the EXACT object/location IDs you'll need later:
   - Store precise IDs like "cabinet_bar__minus_00_dot_36_bar__plus_00_dot_38_bar__plus_00_dot_27"
   - These IDs are required to interact with objects in future actions

3. **After compression**, call ReadExperience(db_index) to retrieve precise IDs
   - Don't try to remember IDs from memory - they are deleted after compression
   - Always retrieve before taking actions that need specific location names

ALFWORLD-SPECIFIC EXAMPLE:

When you've searched multiple cabinets looking for an object:

<function=CompressExperience>
<parameter=summary>Index map:
- ctx_search_001 - List of checked locations (3 cabinets, all empty)
- ctx_task_001 - Task progress: looking for butterknife
Status: Need to check more cabinets or try drawers</parameter>
<parameter=db_blocks>[
  {"db_index": "ctx_search_001", "db_content": "Checked locations (all empty):\\n1. cabinet_bar__minus_00_dot_36_bar__plus_00_dot_38_bar__plus_00_dot_27\\n2. cabinet_bar__minus_00_dot_36_bar__plus_00_dot_38_bar__plus_01_dot_71\\n3. cabinet_bar__minus_00_dot_36_bar__plus_00_dot_38_bar__plus_02_dot_63"},
  {"db_index": "ctx_task_001", "db_content": "Task: put clean butterknife on diningtable\\nProgress: Searching for butterknife\\nInventory: empty"}
]</parameter>
</function>

Later, when you need to know which cabinets you already checked:

<function=ReadExperience>
<parameter=db_index>ctx_search_001</parameter>
</function>

This returns the exact cabinet IDs so you don't re-check the same locations.
"""


def _get_alfworld_memory_guidance(tool_call_format: str) -> str:
    """Get ALFWorld-specific memory guidance based on tool call format."""
    if tool_call_format == "qwen":
        return ALFWORLD_MEMORY_GUIDANCE
    else:
        return ALFWORLD_MEMORY_GUIDANCE_XML


def _get_alfworld_system_prompt(tool_call_format: str) -> str:
    """Get the full system prompt for ALFWorld agent."""
    if tool_call_format == "qwen":
        return ALFWORLD_SYSTEM_PROMPT_BASE + ALFWORLD_TOOL_CALL_FORMAT_QWEN
    else:
        return ALFWORLD_SYSTEM_PROMPT_BASE + ALFWORLD_TOOL_CALL_FORMAT_XML


def _get_tool_format_suffix(tool_call_format: str) -> str:
    """Get the tool format suffix for memory agents."""
    if tool_call_format == "qwen":
        return ALFWORLD_TOOL_CALL_FORMAT_QWEN
    else:
        return ALFWORLD_TOOL_CALL_FORMAT_XML


def _format_alfworld_observation(
    observation: Any,
    hide_actions: bool = False,
    hide_initial_obs: bool = False,
) -> str:
    """Format ALFWorld observation for the agent.

    Args:
        observation: Observation dict from environment.
        hide_actions: If True, omit admissible_commands from output.
        hide_initial_obs: If True, strip room description from the initial observation
            so location IDs only appear via "look" action (in conversation history,
            not in messages[1]). This makes compression lose the IDs, forcing
            ReadExperience to retrieve them.
    """
    if isinstance(observation, dict):
        parts = []

        # Task description (only in initial observation)
        if "task_description" in observation:
            parts.append(f"Task: {observation['task_description']}")
            if hide_initial_obs:
                # Don't include room description — model must call "look" to see it.
                parts.append('\n[Use execute_action("look") to see your surroundings]')
                parts.append('[IMPORTANT: "look" can only be used ONCE. After that, store location IDs via CompressExperience and use ReadExperience to retrieve them.]')
                # Skip observation text (room description with location IDs)
                # and admissible_commands for the initial step
                return "\n".join(parts)

        # Current observation
        if "observation" in observation:
            parts.append(f"\nObservation:\n{observation['observation']}")

        # Admissible commands (hints) — skip when hide_actions=True
        if not hide_actions:
            if "admissible_commands" in observation and observation["admissible_commands"]:
                commands = observation["admissible_commands"]
                commands_str = ", ".join(commands)
                parts.append(f"\nAvailable actions: {commands_str}")

        if parts:
            return "\n".join(parts)
        else:
            return json.dumps(observation, ensure_ascii=False, indent=2)

    return str(observation)


class ALFWorldAgent(ToolAgent):
    """Agent for ALFWorld environments."""

    def __init__(
        self,
        tool_call_format: str = "xml",
        model_name: Optional[str] = None,
        hide_admissible_commands: bool = False,
        hide_initial_obs: bool = False,
    ):
        """Initialize the ALFWorld agent."""
        # Auto-detect format from model name
        if model_name and "qwen" in model_name.lower():
            tool_call_format = "qwen"

        system_prompt = _get_alfworld_system_prompt(tool_call_format)

        _ha = hide_admissible_commands
        _hi = hide_initial_obs
        formatter = lambda obs: _format_alfworld_observation(obs, hide_actions=_ha, hide_initial_obs=_hi)

        super().__init__(
            system_prompt=system_prompt,
            tool_call_format=tool_call_format,
            agent_name="alfworld_agent",
            model_name=model_name,
            observation_formatter=formatter,
        )


class ALFWorldAgentWithMemory(ToolAgentWithMemory):
    """ALFWorld Agent with Memory support (CompressExperience/ReadExperience)."""

    def __init__(
        self,
        tool_call_format: str = "xml",
        model_name: Optional[str] = None,
        # Memory parameters
        compression_mode: str = "lossless_db",
        context_db: Any = None,
        db_path: str = "alfworld_experience.sqlite",
        context_length_threshold: int = 16000,
        auto_compress_prompt: bool = True,
        disable_retrieve: bool = False,
        hide_admissible_commands: bool = False,
        hide_initial_obs: bool = False,
        max_summary_tokens: int = 0,
    ):
        """Initialize ALFWorld agent with memory support.

        Args:
            hide_admissible_commands: If True, don't show admissible_commands.
            hide_initial_obs: If True, strip room description from initial observation.
                Model must call "look" to see locations. After compression, location IDs
                are lost and must be retrieved via ReadExperience.
            max_summary_tokens: If > 0, truncate summary to force ReadExperience usage.
        """
        # Auto-detect format from model name
        if model_name and "qwen" in model_name.lower():
            tool_call_format = "qwen"

        _ha = hide_admissible_commands
        _hi = hide_initial_obs
        formatter = lambda obs: _format_alfworld_observation(obs, hide_actions=_ha, hide_initial_obs=_hi)

        # Build tool format suffix, only include memory guidance when memory is enabled
        tool_format_suffix = _get_tool_format_suffix(tool_call_format)
        if compression_mode in ("lossless_db", "lossy", "rag"):
            memory_guidance = _get_alfworld_memory_guidance(tool_call_format)
            full_tool_format_suffix = memory_guidance + tool_format_suffix
        else:
            full_tool_format_suffix = tool_format_suffix

        super().__init__(
            system_prompt=ALFWORLD_SYSTEM_PROMPT_BASE,
            tool_call_format=tool_call_format,
            agent_name="alfworld_agent",
            model_name=model_name,
            observation_formatter=formatter,
            # Memory parameters
            compression_mode=compression_mode,
            context_db=context_db,
            db_path=db_path,
            context_length_threshold=context_length_threshold,
            auto_compress_prompt=auto_compress_prompt,
            disable_retrieve=disable_retrieve,
            max_summary_tokens=max_summary_tokens,
            tool_format_suffix=full_tool_format_suffix,
        )
