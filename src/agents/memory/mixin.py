"""
Memory Agent Mixin - Reusable memory/compression capabilities for agents.

This mixin provides lossless compression and retrieval functionality that can be
added to any agent. Implementation based on SWEAgentWithMemory.

Usage:
    class MyAgentWithMemory(MemoryAgentMixin, MyBaseAgent):
        def __init__(self, ...):
            super().__init__(...)
            # Initialize memory after base init
            self.init_memory(compression_mode="lossless_db", ...)

Note:
    Debug output printing ([AGENT → ENV], [ENV → AGENT]) is handled centrally
    by the engine (AgentExecutionEngine._print_agent_response/_print_env_observation).
    This mixin only handles memory tool logic and context status injection.
"""
import copy
import json
import logging
import re
from typing import Any

from .types import MemoryToolResult


logger = logging.getLogger(__name__)


class MemoryAgentMixin:
    """
    Mixin providing memory tools that work like regular environment tools.

    Design: parse_response() -> action -> environment -> execute_memory_tool() -> result

    Usage:
        class MyAgent(MemoryAgentMixin, BaseAgent):
            def __init__(self, ...):
                super().__init__(...)
                self.init_memory(compression_mode="lossless_db", ...)

    Environment should call:
        if agent.is_memory_tool(tool_name):
            result = agent.execute_memory_tool(tool_name, params)
    """

    # =========================================================================
    # Initialization
    # =========================================================================

    def init_memory(
        self,
        compression_mode: str = "lossless_db",
        context_db: Any = None,
        db_path: str = "experience.sqlite",
        context_length_threshold: int = 16000,
        auto_compress_prompt: bool = True,
        disable_retrieve: bool = False,
        max_summary_tokens: int = 0,
    ) -> None:
        """Initialize memory capabilities.

        Args:
            max_summary_tokens: If > 0, truncate summary to this many tokens (approx).
                Forces model to store details in db_blocks and use ReadExperience.
        """
        self.compression_mode = compression_mode
        self.context_length_threshold = context_length_threshold
        self.auto_compress_prompt = auto_compress_prompt
        self.disable_retrieve = disable_retrieve
        self.max_summary_tokens = max_summary_tokens

        # Store original system prompt (before memory tools injection) for reset
        # This will be set by subclass after super().__init__() completes
        self._base_system_prompt = None

        # Statistics
        self.compression_count = 0
        self.retrieval_count = 0
        self.total_chars_compressed = 0
        self.db_indices: list[str] = []

        # SFT data segments - each compression saves messages here
        self.sft_segments: list[dict] = []

        # Initialize database
        # Use in-memory database by default for isolation between parallel agents
        if compression_mode == "lossless_db":
            if context_db is not None:
                self.context_db = context_db
            else:
                from src.database.context_database import create_context_database
                self.context_db = create_context_database(backend="memory")
        else:
            self.context_db = None

        # Initialize RAG storage for rag mode
        if compression_mode == "rag":
            self.rag_chunks: list[str] = []  # Store all message chunks
            self.rag_bm25 = None  # BM25 index, built lazily
        else:
            self.rag_chunks = None
            self.rag_bm25 = None

    # =========================================================================
    # Main Entry Point - Called by Environment
    # =========================================================================

    def is_memory_tool(self, tool_name: str) -> bool:
        """Check if a tool name is a memory tool."""
        return tool_name in ("CompressExperience", "ReadExperience")

    def execute_memory_tool(self, tool_name: str, params: dict) -> MemoryToolResult:
        """
        Execute a memory tool. Called by environment like any other tool.

        Args:
            tool_name: "CompressExperience" or "ReadExperience"
            params: Tool parameters dict

        Returns:
            MemoryToolResult with success status and message
        """
        if tool_name == "CompressExperience":
            return self._execute_compress(params)
        elif tool_name == "ReadExperience":
            return self._execute_retrieve(params)
        else:
            return MemoryToolResult(
                success=False,
                message=f"Unknown memory tool: {tool_name}",
                tool_name=tool_name
            )

    # =========================================================================
    # CompressExperience Implementation
    # =========================================================================

    def _execute_compress(self, params: dict) -> MemoryToolResult:
        """
        Execute CompressExperience with input validation.

        Expected params:
            summary: str (required)
            db_blocks: list[dict] or JSON string (required for lossless_db)
        """
        # === Validate summary ===
        summary = params.get("summary", "")
        if not isinstance(summary, str) or not summary.strip():
            return MemoryToolResult(
                success=False,
                message="Error: CompressExperience requires 'summary' (non-empty string)",
                tool_name="CompressExperience"
            )
        summary = summary.strip()

        # === Check messages ===
        messages = getattr(self, 'messages', [])
        if len(messages) <= 2:
            return MemoryToolResult(
                success=False,
                message="Not enough messages to compress. Continue with your task.",
                tool_name="CompressExperience"
            )

        # === Lossy mode: no db_blocks needed ===
        if self.compression_mode == "lossy":
            return self._do_lossy_compress(summary)

        # === RAG mode: store messages in BM25, no db_blocks needed ===
        if self.compression_mode == "rag":
            return self._do_rag_compress(summary)

        # === Lossless mode: validate db_blocks ===
        db_blocks_raw = params.get("db_blocks")
        if db_blocks_raw is None:
            return MemoryToolResult(
                success=False,
                message="Error: CompressExperience requires 'db_blocks'",
                tool_name="CompressExperience"
            )

        # Parse if string (JSON or XML format)
        if isinstance(db_blocks_raw, str):
            db_blocks_str = db_blocks_raw.strip()
            # Try JSON first
            if db_blocks_str.startswith('['):
                try:
                    db_blocks = json.loads(db_blocks_str)
                except json.JSONDecodeError as e:
                    return MemoryToolResult(
                        success=False,
                        message=f"Error: db_blocks JSON parse failed. {e}",
                        tool_name="CompressExperience"
                    )
            # Try XML format (e.g., <db_block>...</db_block>)
            elif '<db_block>' in db_blocks_str or '<db_index>' in db_blocks_str:
                db_blocks = self._parse_xml_db_blocks(db_blocks_str)
                if db_blocks is None:
                    return MemoryToolResult(
                        success=False,
                        message="Error: db_blocks XML parse failed. Expected <db_block><db_index>...</db_index><db_content>...</db_content></db_block>",
                        tool_name="CompressExperience"
                    )
            else:
                return MemoryToolResult(
                    success=False,
                    message="Error: db_blocks must be JSON array or XML <db_block> elements",
                    tool_name="CompressExperience"
                )
        else:
            db_blocks = db_blocks_raw

        if not isinstance(db_blocks, list) or len(db_blocks) == 0:
            return MemoryToolResult(
                success=False,
                message="Error: db_blocks must be a non-empty list",
                tool_name="CompressExperience"
            )

        # === Validate and resolve each block ===
        validated_blocks: list[tuple[str, str]] = []
        seen_indices: set[str] = set()

        # Build anchor source for extraction
        messages_to_compress = messages[2:]
        anchor_source = self._build_anchor_source(messages_to_compress)

        for i, block in enumerate(db_blocks):
            if not isinstance(block, dict):
                return MemoryToolResult(
                    success=False,
                    message=f"Error: db_blocks[{i}] must be object with 'db_index' and 'db_content'",
                    tool_name="CompressExperience"
                )

            # Validate db_index
            db_index = block.get("db_index", "")
            if not isinstance(db_index, str) or not db_index.strip():
                return MemoryToolResult(
                    success=False,
                    message=f"Error: db_blocks[{i}] missing 'db_index'",
                    tool_name="CompressExperience"
                )
            db_index = db_index.strip()

            # Validate index format
            if len(db_index) > 64 or not re.fullmatch(r"[A-Za-z0-9_-]+", db_index):
                return MemoryToolResult(
                    success=False,
                    message=f"Error: db_index '{db_index}' invalid. Use 1-64 chars: letters/numbers/underscore/dash",
                    tool_name="CompressExperience"
                )

            # Check duplicates within this compression call only (allow updating existing indices)
            if db_index in seen_indices:
                return MemoryToolResult(
                    success=False,
                    message=f"Error: db_index '{db_index}' appears multiple times in db_blocks",
                    tool_name="CompressExperience"
                )
            seen_indices.add(db_index)

            # Get db_content (may be empty if using anchors)
            db_content = block.get("db_content", "")
            if db_content is None:
                db_content = ""
            if not isinstance(db_content, str):
                return MemoryToolResult(
                    success=False,
                    message=f"Error: db_blocks[{i}] db_content must be string",
                    tool_name="CompressExperience"
                )

            # Check for anchors
            start_anchor = block.get("start_anchor", "")
            mid_anchor = block.get("mid_anchor", "")
            end_anchor = block.get("end_anchor", "")

            has_anchors = bool(start_anchor or mid_anchor or end_anchor)
            if has_anchors:
                if not (start_anchor and mid_anchor and end_anchor):
                    return MemoryToolResult(
                        success=False,
                        message=f"Error: db_blocks[{i}] anchors require all three: start_anchor, mid_anchor, end_anchor",
                        tool_name="CompressExperience"
                    )
                # Extract content using anchors
                extracted = self._extract_by_anchors(anchor_source, start_anchor, mid_anchor, end_anchor)
                if extracted is None:
                    return MemoryToolResult(
                        success=False,
                        message=f"Error: db_blocks[{i}] anchor extraction failed. Check anchors are exact substrings.",
                        tool_name="CompressExperience"
                    )
                # Combine db_content with extracted
                if db_content:
                    if "{{EXTRACT}}" in db_content:
                        final_content = db_content.replace("{{EXTRACT}}", extracted)
                    else:
                        final_content = f"{db_content}\n\n{extracted}"
                else:
                    final_content = extracted
                validated_blocks.append((db_index, final_content))
            else:
                # No anchors, must have db_content
                if not db_content.strip():
                    return MemoryToolResult(
                        success=False,
                        message=f"Error: db_blocks[{i}] needs 'db_content' or anchors",
                        tool_name="CompressExperience"
                    )
                validated_blocks.append((db_index, db_content))

        # === Execute compression ===
        return self._do_lossless_compress(summary, validated_blocks)

    def _do_lossless_compress(
        self,
        summary: str,
        blocks: list[tuple[str, str]],
    ) -> MemoryToolResult:
        """Execute lossless compression with validated blocks."""
        messages = getattr(self, 'messages', [])
        messages_to_compress = messages[2:]
        original_chars = sum(len(str(m.get('content', ''))) for m in messages_to_compress)

        # Store in database
        if self.context_db is not None:
            for db_index, db_content in blocks:
                self.context_db.store(db_index, {
                    'db_content': db_content,
                    'summary': summary,
                    'original_chars': original_chars,
                    'message_count': len(messages_to_compress),
                })

        # Update statistics
        self.compression_count += 1
        self.total_chars_compressed += original_chars
        created_indices = [idx for idx, _ in blocks]
        self.db_indices.extend(created_indices)

        # Enforce max summary length (in tokens, approx 4 chars/token).
        # Forces model to store details in db_blocks and use ReadExperience.
        max_summary_tokens = getattr(self, 'max_summary_tokens', 0)
        if max_summary_tokens > 0:
            max_chars = max_summary_tokens * 4
            if len(summary) > max_chars:
                summary = summary[:max_chars] + "\n[TRUNCATED - use ReadExperience(db_index) to retrieve details]"

        # Build summary message with compression stats
        ratio = 1 - (len(summary) / original_chars) if original_chars > 0 else 0
        compression_header = (
            f"=== SUMMARY OF YOUR PREVIOUS CONTEXT ===\n"
            f"[Compressed {len(messages_to_compress)} messages ({original_chars} chars -> {len(summary)} chars, {ratio:.0%} reduction)]\n\n"
            f"MANDATORY WORKFLOW - Before ANY tool call:\n"
            f"1. Read the index map below to see what information you already have\n"
            f"2. If the information you need is stored → call ReadExperience(db_index) to retrieve it\n"
            f"3. Only call other tools (file_editor, execute_bash, etc.) if the information is NOT in your stored indices\n"
            f"VIOLATION: Re-running tools to get information you already stored = SEVERE PENALTY\n\n"
        )
        summary_message = {
            'role': 'user',
            'content': f"{compression_header}{summary}",
            'metadata': {'compression': True, 'db_indices': created_indices}
        }

        # Save current messages as SFT segment BEFORE compression
        # (compress_call is already in self.messages via update_from_model)
        if not hasattr(self, 'sft_segments'):
            self.sft_segments = []

        segment_type = 'pre_compression' if len(self.sft_segments) == 0 else 'post_compression'
        pre_compress_messages = copy.deepcopy(messages)

        self.sft_segments.append({
            'messages': pre_compress_messages,
            'segment_type': segment_type,
            'num_messages': len(pre_compress_messages),
            'compressed_at_step': getattr(self, 'step', -1),
        })

        # Replace messages with compressed version
        compressed_messages = [messages[0], messages[1], summary_message]
        self.messages = compressed_messages

        # Log segment creation for training consistency
        logger.info(
            f"Lossless segment created: type={segment_type}, "
            f"pre_compress_msgs={len(pre_compress_messages)}, "
            f"post_compress_msgs={len(compressed_messages)}, "
            f"total_segments={len(self.sft_segments)}"
        )

        return MemoryToolResult(
            success=True,
            message="",
            tool_name="CompressExperience",
            indices=created_indices
        )

    def _do_lossy_compress(self, summary: str) -> MemoryToolResult:
        """Execute lossy compression (no DB storage)."""
        messages = getattr(self, 'messages', [])
        messages_to_compress = messages[2:]
        original_chars = sum(len(str(m.get('content', ''))) for m in messages_to_compress)

        self.compression_count += 1
        self.total_chars_compressed += original_chars

        # Build summary message with compression stats
        ratio = 1 - (len(summary) / original_chars) if original_chars > 0 else 0
        compression_header = (
            f"=== SUMMARY OF YOUR PREVIOUS CONTEXT ===\n"
            f"[Compressed {len(messages_to_compress)} messages ({original_chars} chars -> {len(summary)} chars, {ratio:.0%} reduction)]\n\n"
            f"IMPORTANT: This summary is your ONLY memory of previous work. Original content cannot be retrieved.\n"
            f"Use the information in this summary to continue your work. Do NOT re-run tools to get information already described here.\n\n"
        )
        summary_message = {
            'role': 'user',
            'content': f"{compression_header}{summary}",
            'metadata': {'compression': True, 'compression_mode': 'lossy'}
        }

        # Save current messages as SFT segment BEFORE compression
        if not hasattr(self, 'sft_segments'):
            self.sft_segments = []

        segment_type = 'pre_compression' if len(self.sft_segments) == 0 else 'post_compression'
        pre_compress_messages = copy.deepcopy(messages)

        self.sft_segments.append({
            'messages': pre_compress_messages,
            'segment_type': segment_type,
            'num_messages': len(pre_compress_messages),
            'compressed_at_step': getattr(self, 'step', -1),
        })

        # Replace messages with compressed version
        compressed_messages = [messages[0], messages[1], summary_message]
        self.messages = compressed_messages

        # Log segment creation for training consistency
        logger.info(
            f"Lossy segment created: type={segment_type}, "
            f"pre_compress_msgs={len(pre_compress_messages)}, "
            f"post_compress_msgs={len(compressed_messages)}, "
            f"total_segments={len(self.sft_segments)}"
        )

        return MemoryToolResult(
            success=True,
            message="",
            tool_name="CompressExperience"
        )

    def _do_rag_compress(self, summary: str) -> MemoryToolResult:
        """Execute RAG compression (store messages in BM25 for retrieval)."""
        messages = getattr(self, 'messages', [])
        messages_to_compress = messages[2:]
        original_chars = sum(len(str(m.get('content', ''))) for m in messages_to_compress)

        self.compression_count += 1
        self.total_chars_compressed += original_chars

        # Store each message as a chunk for RAG retrieval
        for msg in messages_to_compress:
            role = msg.get('role', 'unknown')
            content = str(msg.get('content', ''))
            if content.strip():
                chunk = f"[{role}] {content}"
                self.rag_chunks.append(chunk)

        # Rebuild BM25 index with all chunks
        if self.rag_chunks:
            from rank_bm25 import BM25Okapi
            tokenized_chunks = [chunk.split() for chunk in self.rag_chunks]
            self.rag_bm25 = BM25Okapi(tokenized_chunks)

        # Build summary message with compression stats
        ratio = 1 - (len(summary) / original_chars) if original_chars > 0 else 0
        compression_header = (
            f"=== SUMMARY OF YOUR PREVIOUS CONTEXT ===\n"
            f"[Compressed {len(messages_to_compress)} messages ({original_chars} chars -> {len(summary)} chars, {ratio:.0%} reduction)]\n\n"
            f"MANDATORY WORKFLOW - Before ANY tool call:\n"
            f"1. Read this summary to see what information you already have\n"
            f"2. If you need details from previous work → call ReadExperience(query='...') to search\n"
            f"3. Only call other tools (file_editor, execute_bash, etc.) if the information is NOT retrievable\n"
            f"VIOLATION: Re-running tools to get information you already have = SEVERE PENALTY\n\n"
        )
        summary_message = {
            'role': 'user',
            'content': f"{compression_header}{summary}",
            'metadata': {'compression': True, 'compression_mode': 'rag', 'num_chunks': len(self.rag_chunks)}
        }

        # Save current messages as SFT segment BEFORE compression
        if not hasattr(self, 'sft_segments'):
            self.sft_segments = []

        segment_type = 'pre_compression' if len(self.sft_segments) == 0 else 'post_compression'
        pre_compress_messages = copy.deepcopy(messages)

        self.sft_segments.append({
            'messages': pre_compress_messages,
            'segment_type': segment_type,
            'num_messages': len(pre_compress_messages),
            'compressed_at_step': getattr(self, 'step', -1),
        })

        # Replace messages with compressed version
        compressed_messages = [messages[0], messages[1], summary_message]
        self.messages = compressed_messages

        logger.info(
            f"RAG segment created: type={segment_type}, "
            f"pre_compress_msgs={len(pre_compress_messages)}, "
            f"total_chunks={len(self.rag_chunks)}"
        )

        return MemoryToolResult(
            success=True,
            message="",
            tool_name="CompressExperience"
        )

    # =========================================================================
    # ReadExperience Implementation
    # =========================================================================

    def _execute_retrieve(self, params: dict) -> MemoryToolResult:
        """
        Execute ReadExperience with input validation.

        Expected params:
            db_index: str (required for lossless_db mode)
            query: str (required for rag mode)
        """
        if self.disable_retrieve:
            return MemoryToolResult(
                success=False,
                message="Error: ReadExperience disabled in compress-only mode",
                tool_name="ReadExperience"
            )

        # === RAG mode: search by query ===
        if self.compression_mode == "rag":
            return self._do_rag_retrieve(params)

        # === Lossless mode: lookup by db_index ===
        # Validate db_index
        db_index = params.get("db_index", "")
        if not isinstance(db_index, str) or not db_index.strip():
            return MemoryToolResult(
                success=False,
                message="Error: ReadExperience requires 'db_index'",
                tool_name="ReadExperience"
            )
        db_index = db_index.strip()

        # Check database
        if self.context_db is None:
            return MemoryToolResult(
                success=False,
                message="Error: Context database not initialized",
                tool_name="ReadExperience"
            )

        # Retrieve
        try:
            entry = self.context_db.retrieve(db_index)
        except KeyError:
            entry = None
        except Exception as e:
            return MemoryToolResult(
                success=False,
                message=f"Error: Failed to retrieve '{db_index}': {e}",
                tool_name="ReadExperience"
            )

        if entry is None:
            available = ', '.join(self.db_indices) if self.db_indices else 'none'
            return MemoryToolResult(
                success=False,
                message=f"Error: Index '{db_index}' not found. Available: {available}",
                tool_name="ReadExperience"
            )

        self.retrieval_count += 1
        db_content = entry.get('db_content', '')

        return MemoryToolResult(
            success=True,
            message=f"Retrieved [{db_index}] ({len(db_content)} chars):\n\n{db_content}",
            tool_name="ReadExperience"
        )

    def _do_rag_retrieve(self, params: dict, top_k: int = 1) -> MemoryToolResult:
        """Execute RAG retrieval using BM25 search."""
        # Validate query
        query = params.get("query", "")
        if not isinstance(query, str) or not query.strip():
            return MemoryToolResult(
                success=False,
                message="Error: ReadExperience requires 'query' in RAG mode",
                tool_name="ReadExperience"
            )
        query = query.strip()

        # Check if BM25 index exists
        if self.rag_bm25 is None or not self.rag_chunks:
            return MemoryToolResult(
                success=False,
                message="Error: No compressed content available for retrieval. Compress first.",
                tool_name="ReadExperience"
            )

        # Search with BM25
        tokenized_query = query.split()
        scores = self.rag_bm25.get_scores(tokenized_query)

        # Get top-k results
        import numpy as np
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include if there's some relevance
                results.append(f"[Score: {scores[idx]:.2f}]\n{self.rag_chunks[idx]}")

        if not results:
            return MemoryToolResult(
                success=False,
                message=f"No relevant content found for query: '{query}'",
                tool_name="ReadExperience"
            )

        self.retrieval_count += 1
        result_text = "\n\n---\n\n".join(results)

        return MemoryToolResult(
            success=True,
            message=f"Found {len(results)} relevant chunks for '{query}':\n\n{result_text}",
            tool_name="ReadExperience"
        )

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _build_anchor_source(messages: list[dict]) -> str:
        """Build source text from messages for anchor extraction."""
        parts = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            parts.append(f"[{role}]\n{content}")
        return "\n\n".join(parts)

    @staticmethod
    def _normalize_whitespace(text: str) -> str:
        """Normalize whitespace: collapse multiple spaces/tabs/newlines into single space."""
        import re
        return re.sub(r'\s+', ' ', text.strip())

    @staticmethod
    def _extract_by_anchors(source: str, start: str, mid: str, end: str) -> str | None:
        """Extract span using three anchors. Tries exact match first, then normalized whitespace."""
        start, mid, end = start.strip(), mid.strip(), end.strip()
        if not (start and mid and end):
            return None

        # Try exact match first
        start_pos = source.find(start)
        while start_pos != -1:
            mid_pos = source.find(mid, start_pos + len(start))
            while mid_pos != -1:
                end_pos = source.find(end, mid_pos + len(mid))
                if end_pos != -1:
                    return source[start_pos:end_pos + len(end)]
                mid_pos = source.find(mid, mid_pos + 1)
            start_pos = source.find(start, start_pos + 1)

        # Fallback: try with normalized whitespace
        import re
        norm_source = re.sub(r'\s+', ' ', source)
        norm_start = re.sub(r'\s+', ' ', start)
        norm_mid = re.sub(r'\s+', ' ', mid)
        norm_end = re.sub(r'\s+', ' ', end)

        start_pos = norm_source.find(norm_start)
        while start_pos != -1:
            mid_pos = norm_source.find(norm_mid, start_pos + len(norm_start))
            while mid_pos != -1:
                end_pos = norm_source.find(norm_end, mid_pos + len(norm_mid))
                if end_pos != -1:
                    # Return from normalized source (whitespace collapsed)
                    return norm_source[start_pos:end_pos + len(norm_end)]
                mid_pos = norm_source.find(norm_mid, mid_pos + 1)
            start_pos = norm_source.find(norm_start, start_pos + 1)

        return None

    @staticmethod
    def _parse_xml_db_blocks(xml_str: str) -> list[dict] | None:
        """Parse XML format db_blocks into list of dicts.

        Supports format like:
            <db_block>
                <db_index>ctx_001</db_index>
                <db_content>content here</db_content>
            </db_block>

        Also supports anchor-based extraction:
            <db_block>
                <db_index>ctx_001</db_index>
                <start_anchor>...</start_anchor>
                <mid_anchor>...</mid_anchor>
                <end_anchor>...</end_anchor>
            </db_block>
        """
        blocks = []

        # Find all <db_block>...</db_block> sections
        block_pattern = re.compile(r'<db_block>(.*?)</db_block>', re.DOTALL)
        block_matches = block_pattern.findall(xml_str)

        # If no <db_block> tags, try parsing as single block with just <db_index> and <db_content>
        if not block_matches:
            block_matches = [xml_str]

        for block_content in block_matches:
            block_dict = {}

            # Extract db_index
            idx_match = re.search(r'<db_index>(.*?)</db_index>', block_content, re.DOTALL)
            if idx_match:
                block_dict['db_index'] = idx_match.group(1).strip()

            # Extract db_content
            content_match = re.search(r'<db_content>(.*?)</db_content>', block_content, re.DOTALL)
            if content_match:
                block_dict['db_content'] = content_match.group(1)

            # Extract anchors if present
            start_match = re.search(r'<start_anchor>(.*?)</start_anchor>', block_content, re.DOTALL)
            if start_match:
                block_dict['start_anchor'] = start_match.group(1)

            mid_match = re.search(r'<mid_anchor>(.*?)</mid_anchor>', block_content, re.DOTALL)
            if mid_match:
                block_dict['mid_anchor'] = mid_match.group(1)

            end_match = re.search(r'<end_anchor>(.*?)</end_anchor>', block_content, re.DOTALL)
            if end_match:
                block_dict['end_anchor'] = end_match.group(1)

            # Only add if we got at least db_index
            if 'db_index' in block_dict:
                blocks.append(block_dict)

        return blocks if blocks else None

    # =========================================================================
    # Context Status Injection
    # =========================================================================

    def update_from_env(self, observation, reward, done, info, **kwargs):
        """Update agent state and inject context status into observation.

        Note: Debug output printing ([ENV → AGENT]) is handled by engine.
        This method only injects context status for memory-enabled agents.
        """
        # Inject context status BEFORE calling super() so trajectory records it
        compression_mode = getattr(self, 'compression_mode', 'none')
        if not done and compression_mode in ("lossless_db", "lossy", "rag"):
            context_status = self.get_context_status()
            observation = str(observation) + context_status

        # Call super() with modified observation
        super().update_from_env(observation, reward, done, info, **kwargs)

        # Save final messages when trajectory ends
        if done:
            self.finalize_segments()

    def finalize_segments(self) -> None:
        """Create final segment after trajectory completes (for segmented RL training).

        Only creates final segment if there are existing segments from compression.
        This ensures that trajectories without compression are not segmented.
        """
        messages = getattr(self, 'messages', [])
        if not hasattr(self, 'sft_segments'):
            self.sft_segments = []

        # Only create final segment if there are existing segments
        if len(self.sft_segments) > 0:
            # Check if final segment already exists
            if not (self.sft_segments and self.sft_segments[-1].get('segment_type') == 'final'):
                self.sft_segments.append({
                    'messages': copy.deepcopy(messages),
                    'segment_type': 'final',
                })

        # Also save sft_segments to trajectory info for serialization
        trajectory = getattr(self, '_trajectory', None)
        if trajectory is not None and hasattr(trajectory, 'info'):
            trajectory.info['sft_segments'] = copy.deepcopy(self.sft_segments)

    def get_context_status(self) -> str:
        """Generate context status string to append to observations."""
        messages = getattr(self, 'messages', [])
        working = self._estimate_working_tokens()
        threshold = self.context_length_threshold

        status = f"\n\n[Context Status: working tokens={working}, threshold={threshold}]"
        status += f"\n[Message Count: {len(messages)} messages]"

        # Show available indices (reminds agent what can be retrieved)
        if self.db_indices:
            status += f"\n[Available indices: {', '.join(self.db_indices)}]"
            status += f"\n[Read your summary's index map and call ReadExperience(db_index) to retrieve what you need]"

        # Add compression warnings if auto_compress_prompt is enabled
        if getattr(self, 'auto_compress_prompt', True):
            if working > threshold:
                status += f"\n CONSIDER COMPRESS NOW: working ({working}) > threshold ({threshold})"
            elif working > 0.8 * threshold:
                status += f"\n[Warning: approaching threshold, consider compressing]"

        return status

    # =========================================================================
    # Statistics
    # =========================================================================

    def _estimate_working_tokens(self) -> int:
        """Estimate working context tokens (excludes system + task)."""
        messages = getattr(self, 'messages', [])
        token_manager = getattr(self, 'token_manager', None)
        if token_manager is None:
            return 0  # Agent not fully initialized (e.g., early Docker failure)
        return token_manager.get_working_tokens(messages)

    def _estimate_total_tokens(self) -> int:
        """Estimate total context tokens."""
        messages = getattr(self, 'messages', [])
        token_manager = getattr(self, 'token_manager', None)
        if token_manager is None:
            return 0  # Agent not fully initialized (e.g., early Docker failure)
        return token_manager.get_total_tokens(messages)

    def reset_memory_stats(self) -> None:
        """Reset memory statistics for new trajectory."""
        self.compression_count = 0
        self.retrieval_count = 0
        self.total_chars_compressed = 0
        self.db_indices = []
        self.sft_segments = []

    def get_memory_stats(self) -> dict:
        """Get memory usage statistics."""
        return {
            'compression_mode': self.compression_mode,
            'compression_count': self.compression_count,
            'retrieval_count': self.retrieval_count,
            'total_chars_compressed': self.total_chars_compressed,
            'db_indices': self.db_indices.copy(),
            'working_tokens': self._estimate_working_tokens(),
            'total_tokens': self._estimate_total_tokens(),
            'threshold': self.context_length_threshold,
        }
