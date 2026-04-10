"""
Memory Tool Prompts - Prompt templates for memory compression and retrieval tools.

This module contains all prompt templates for injecting memory tool descriptions
into agent system prompts, supporting both XML and Qwen tool call formats.
"""

# =============================================================================
# Lossless DB Mode Prompts (Full memory with compress + retrieve)
# =============================================================================

# Memory system introduction (shared across formats)
_MEMORY_SYSTEM_INTRO = '''
=== CRITICAL: THREE OBJECTIVES ===

You have THREE equally important goals:
1. SOLVE THE TASK correctly
2. KEEP working context UNDER the threshold (shown in [Context Status] after EVERY observation)
3. NEVER make redundant tool calls (same tool + same arguments without file/status changes in between)

SEVERE PENALTIES (can nullify your task success reward):
- Context overflow: If working > threshold, you receive a SEVERE PENALTY that can completely offset solving the task
- Redundant tool calls: Calling the SAME tool with IDENTICAL arguments twice (without modifying files/status in between) results in a SEVERE PENALTY
- These penalties are AS IMPORTANT as solving the task - poor memory management can make a solved task worth ZERO

MANDATORY PRACTICES:
- Monitor [Context Status: working tokens=X, threshold=Z] after EVERY observation
- Compress BEFORE working exceeds threshold (don't wait until it's too late!)
- When compressing, store BROAD coverage in db_blocks - include everything you might need later

CRITICAL - After compression:
- Compressed messages are DELETED from context. The ONLY way to access them is ReadExperience.
- If you need past information, you MUST call ReadExperience(db_index) - re-running tools without file/status changes is forbidden and penalized.

'''

# Base description for CompressExperience (shared across formats)
_COMPRESS_EXPERIENCE_DESC = '''–– BEGIN FUNCTION: CompressExperience ––
Description:
Compress working context to database for later retrieval. Replaces all messages
(except system prompt and task description) with your summary.

Usage:
  • Check [Context Status: working tokens=X, threshold=Z] at the end of each observation
  • Strongly recommended when working > 0.8 * threshold
  • Exceeding threshold will result in penalty
  • After compression, use ReadExperience to get saved content instead of re-running tools
  • When compressing multiple times: include ALL previous indices in your new summary (copy them over), then add new ones

Parameters:
  1. summary (string, required)
     Index map listing ALL stored indices (both old and new). Format:
     - <db_index> - <what it contains>
     - <db_index> - <what it contains>
     Include current status and next steps at the end.

  2. db_blocks (array, required)
     List of content blocks to store. Two options:

     Option A - Write content yourself:
       • db_index (string): Unique key, e.g. "ctx_code_001"
       • db_content (string): Content you write/summarize

     Option B - System auto-extracts from current conversation:
       The system finds text between your anchors and saves it automatically.
       • db_index (string): Unique key
       • start_anchor (string): REQUIRED - exact text where extraction STARTS
       • mid_anchor (string): REQUIRED - exact text that MUST appear in the middle
       • end_anchor (string): REQUIRED - exact text where extraction ENDS
       ALL THREE anchors are REQUIRED. Missing any anchor = failure.

       IMPORTANT for anchors:
       - Choose your own anchors that uniquely identify the content boundaries
       - start_anchor: unique text at the START of what you want to extract
       - mid_anchor: unique text somewhere in the MIDDLE (for verification)
       - end_anchor: unique text at the END of what you want to extract
       - Keep anchors SHORT (20-100 chars), NOT entire code blocks
       - Good: "def _check_required", "raise ValueError", "return result"
       - Bad: copying 10+ lines of code (whitespace errors cause failures)

Tip: Use Option A for summaries. Use Option B only for large verbatim outputs (test results, stack traces) where you want exact copy.
'''

# XML format examples - Generic template
_COMPRESS_EXAMPLE_XML = '''
Example:
<function=CompressExperience>
<parameter=summary>Index map:
- ctx_data_001 - Brief description of what's stored
- ctx_data_002 - Brief description of what's stored
Status: Current progress and next steps</parameter>
<parameter=db_blocks>[
  {"db_index": "ctx_data_001", "db_content": "Precise details, exact IDs, full content..."},
  {"db_index": "ctx_data_002", "db_content": "More precise details..."}
]</parameter>
</function>

IMPORTANT:
- summary: Keep descriptions SHORT (what type of data, not the data itself)
- db_blocks: Store PRECISE details you'll need later (exact IDs, full content, specific values)
- After compression, use ReadExperience(db_index) to retrieve precise details

–– END FUNCTION ––
'''

# Qwen format examples - Generic template
_COMPRESS_EXAMPLE_QWEN = '''
Example:
<tool_call>
{"name": "CompressExperience", "arguments": {"summary": "Index map:\\n- ctx_data_001 - Brief description of what's stored\\n- ctx_data_002 - Brief description of what's stored\\nStatus: Current progress and next steps", "db_blocks": [{"db_index": "ctx_data_001", "db_content": "Precise details, exact IDs, full content..."}, {"db_index": "ctx_data_002", "db_content": "More precise details..."}]}}
</tool_call>

IMPORTANT:
- summary: Keep descriptions SHORT (what type of data, not the data itself)
- db_blocks: Store PRECISE details you'll need later (exact IDs, full content, specific values)
- After compression, use ReadExperience(db_index) to retrieve precise details

–– END FUNCTION ––
'''

# ReadExperience description (shared)
_READ_EXPERIENCE_DESC = '''
–– BEGIN FUNCTION: ReadExperience ––
Description:
Retrieve previously compressed content by index.

Usage:
  • Use when you need exact details stored during compression
  • Available indices shown in [Context Status] and your summary's index map
  • Always retrieve instead of re-running tools for same information

Parameters:
  1. db_index (string, required)
     The index to retrieve. Must match exactly.
'''

_READ_EXAMPLE_XML = '''
Example:
<function=ReadExperience>
<parameter=db_index>ctx_code_001</parameter>
</function>

–– END FUNCTION ––
'''

_READ_EXAMPLE_QWEN = '''
Example:
<tool_call>
{"name": "ReadExperience", "arguments": {"db_index": "ctx_code_001"}}
</tool_call>

–– END FUNCTION ––
'''

# =============================================================================
# Compress-Only Mode Prompts (Lossy compression, no retrieval)
# =============================================================================

# Compress-only system intro (two objectives, no retrieve)
_COMPRESS_ONLY_SYSTEM_INTRO = '''
=== CRITICAL: TWO OBJECTIVES ===

You have TWO equally important goals:
1. SOLVE THE TASK correctly
2. KEEP working context UNDER the threshold (shown in [Context Status] after EVERY observation)

SEVERE PENALTIES (can nullify your task success reward):
- Context overflow: If working > threshold, you receive a SEVERE PENALTY that can completely offset solving the task
- This penalty is AS IMPORTANT as solving the task - poor memory management can make a solved task worth ZERO

MANDATORY PRACTICES:
- Monitor [Context Status: working tokens=X, threshold=Z] after EVERY observation
- Compress BEFORE working exceeds threshold (don't wait until it's too late!)
- Make summary comprehensive - once compressed, original content CANNOT be retrieved

'''

# Compress-only description
_COMPRESS_ONLY_DESC = '''–– BEGIN FUNCTION: CompressExperience ––
Description:
Compress working context into a summary. Replaces all messages (except system
prompt and task description) with your summary.

Usage:
  • Strongly recommended when [Context Status] shows working > 0.8 * threshold
  • Exceeding threshold will result in penalty
  • Once compressed, original content CANNOT be retrieved - make summary comprehensive

Parameters:
  1. summary (string, required)
     Comprehensive summary of everything you've learned and done. Include:
     - Key findings and evidence
     - File paths and code locations
     - Current status and next steps
'''

_COMPRESS_ONLY_EXAMPLE_XML = '''
Example:
<function=CompressExperience>
<parameter=summary>Index map:
- ctx_code_001 - utils.py handler() function analysis
- ctx_test_001 - test results summary
Status: identified root cause, next step is to implement fix</parameter>
</function>

–– END FUNCTION ––
'''

_COMPRESS_ONLY_EXAMPLE_QWEN = '''
Example:
<tool_call>
{"name": "CompressExperience", "arguments": {"summary": "Index map:\\n- ctx_code_001 - utils.py handler() function analysis\\n- ctx_test_001 - test results summary\\nStatus: identified root cause, next step is to implement fix"}}
</tool_call>

–– END FUNCTION ––
'''

# =============================================================================
# RAG Mode Prompts (BM25-based retrieval)
# =============================================================================

# RAG mode system intro
_RAG_SYSTEM_INTRO = '''
=== CRITICAL: THREE OBJECTIVES ===

You have THREE equally important goals:
1. SOLVE THE TASK correctly
2. KEEP working context UNDER the threshold (shown in [Context Status])
3. NEVER make redundant tool calls (same tool + same arguments without file/status changes in between)

SEVERE PENALTIES (can nullify your task success reward):
- Context overflow: If working > threshold, you receive a SEVERE PENALTY that can completely offset solving the task
- Redundant tool calls: Calling the SAME tool with IDENTICAL arguments twice results in a SEVERE PENALTY
- These penalties are AS IMPORTANT as solving the task - poor memory management can make a solved task worth ZERO

MANDATORY PRACTICES:
- Monitor [Context Status: working tokens=X, threshold=Z] after EVERY observation
- Compress BEFORE working exceeds threshold (don't wait until it's too late!)
- After compression, use ReadExperience(query='...') to search for past content

CRITICAL - After compression:
- Compressed messages are stored and searchable. Use ReadExperience with a query to find relevant content.
- Re-running tools without file/status changes is forbidden and penalized.

'''

_RAG_COMPRESS_DESC = '''–– BEGIN FUNCTION: CompressExperience ––
Description:
Compress working context into a summary. Replaces all messages (except system
prompt and task description) with your summary. Original content is stored for search.

Usage:
  • Strongly recommended when [Context Status] shows working > 0.8 * threshold
  • Exceeding threshold will result in penalty
  • After compression, use ReadExperience to search for past content

Parameters:
  1. summary (string, required)
     Summary of what you've learned and done. Include:
     - Key findings and current status
     - File paths and code locations you worked with
     - Next steps
'''

_RAG_COMPRESS_EXAMPLE_XML = '''
Example:
<function=CompressExperience>
<parameter=summary>Explored repo structure, found issue in core.py line 150.
Modified _check_required_columns() to fix error message.
Next: create reproduce_issue.py and verify fix.</parameter>
</function>

–– END FUNCTION ––
'''

_RAG_COMPRESS_EXAMPLE_QWEN = '''
Example:
<tool_call>
{"name": "CompressExperience", "arguments": {"summary": "Explored repo structure, found issue in core.py line 150.\\nModified _check_required_columns() to fix error message.\\nNext: create reproduce_issue.py and verify fix."}}
</tool_call>

–– END FUNCTION ––
'''

_RAG_READ_DESC = '''–– BEGIN FUNCTION: ReadExperience ––
Description:
Search and retrieve content from your compressed history using a natural language query.

Usage:
  • Use when you need details from past actions (file contents, test results, etc.)
  • Provide a descriptive query to find relevant content
  • Returns top matching chunks from your compressed history

Parameters:
  1. query (string, required)
     What you're looking for. Examples:
     - "test output" - find test results
     - "core.py code" - find code you viewed
     - "error message" - find error outputs
'''

_RAG_READ_EXAMPLE_XML = '''
Example:
<function=ReadExperience>
<parameter=query>test results for pytest</parameter>
</function>

–– END FUNCTION ––
'''

_RAG_READ_EXAMPLE_QWEN = '''
Example:
<tool_call>
{"name": "ReadExperience", "arguments": {"query": "test results for pytest"}}
</tool_call>

–– END FUNCTION ––
'''


# =============================================================================
# Prompt Getter Functions
# =============================================================================

def get_memory_tools_prompt_rag(tool_call_format: str | None = None) -> str:
    """Get RAG mode memory tools prompt with format-specific examples."""
    if tool_call_format == "qwen":
        return _RAG_SYSTEM_INTRO + _RAG_COMPRESS_DESC + _RAG_COMPRESS_EXAMPLE_QWEN + _RAG_READ_DESC + _RAG_READ_EXAMPLE_QWEN
    else:
        # Default to XML format
        return _RAG_SYSTEM_INTRO + _RAG_COMPRESS_DESC + _RAG_COMPRESS_EXAMPLE_XML + _RAG_READ_DESC + _RAG_READ_EXAMPLE_XML


def get_memory_tools_prompt_full(tool_call_format: str | None = None) -> str:
    """Get full memory tools prompt (compress + retrieve) with format-specific examples."""
    if tool_call_format == "qwen":
        return _MEMORY_SYSTEM_INTRO + _COMPRESS_EXPERIENCE_DESC + _COMPRESS_EXAMPLE_QWEN + _READ_EXPERIENCE_DESC + _READ_EXAMPLE_QWEN
    else:
        # Default to XML format
        return _MEMORY_SYSTEM_INTRO + _COMPRESS_EXPERIENCE_DESC + _COMPRESS_EXAMPLE_XML + _READ_EXPERIENCE_DESC + _READ_EXAMPLE_XML


def get_memory_tools_prompt_compress_only(tool_call_format: str | None = None) -> str:
    """Get compress-only memory tools prompt with format-specific examples."""
    if tool_call_format == "qwen":
        return _COMPRESS_ONLY_SYSTEM_INTRO + _COMPRESS_ONLY_DESC + _COMPRESS_ONLY_EXAMPLE_QWEN
    else:
        # Default to XML format
        return _COMPRESS_ONLY_SYSTEM_INTRO + _COMPRESS_ONLY_DESC + _COMPRESS_ONLY_EXAMPLE_XML
