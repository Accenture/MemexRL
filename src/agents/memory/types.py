"""
Memory Tool Types - Data classes for memory tool results.
"""
from dataclasses import dataclass


@dataclass
class MemoryToolResult:
    """Result from executing a memory tool."""
    success: bool
    message: str
    tool_name: str
    indices: list[str] | None = None  # For compress: created indices
