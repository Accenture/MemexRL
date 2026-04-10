"""
Memory Agent Module - Reusable memory/compression capabilities for agents.

This module provides:
- MemoryAgentMixin: A mixin class that adds compression and retrieval to agents
- MemoryToolResult: Data class for memory tool results
- Prompt getter functions for injecting memory tools into agent system prompts

Usage:
    from src.agents.memory import MemoryAgentMixin, MemoryToolResult
    from src.agents.memory import get_memory_tools_prompt_full

    class MyAgentWithMemory(MemoryAgentMixin, MyBaseAgent):
        def __init__(self, ...):
            super().__init__(...)
            self.init_memory(compression_mode="lossless_db", ...)
"""

from .mixin import MemoryAgentMixin
from .types import MemoryToolResult
from .prompts import (
    get_memory_tools_prompt_full,
    get_memory_tools_prompt_compress_only,
    get_memory_tools_prompt_rag,
)

__all__ = [
    "MemoryAgentMixin",
    "MemoryToolResult",
    "get_memory_tools_prompt_full",
    "get_memory_tools_prompt_compress_only",
    "get_memory_tools_prompt_rag",
]
