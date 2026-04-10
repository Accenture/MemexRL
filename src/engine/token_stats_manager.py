"""
Token statistics manager for tracking and estimating token usage across trajectories.

Token Counting Strategy (in priority order):
1. **API Usage** (highest priority): Use actual token counts from API response (e.g., usage.prompt_tokens)
2. **Tokenizer**: Use local tokenizer to count tokens
3. **Char-based fallback** (lowest priority): Estimate as chars/4

All token counting logic is centralized in this class.
"""

import hashlib
from transformers import PreTrainedTokenizerBase
from src.parser import ChatTemplateParser


class TokenStatsManager:
    """
    Centralized manager for token counting and statistics.

    This is the SINGLE SOURCE OF TRUTH for all token-related calculations.

    Responsibilities:
    - Token estimation (with multiple fallback levels)
    - Tracking cumulative input/output tokens from API calls
    - Computing peak working context tokens (excluding system prompt)
    - Managing one-time warnings for estimation fallbacks
    """

    # Fallback estimation constant
    CHARS_PER_TOKEN = 4

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase | None = None,
        chat_parser: ChatTemplateParser | None = None,
        trajectory_idx: int | None = None,
        output_callback=None,
    ):
        """
        Initialize token stats manager.

        Args:
            tokenizer: Tokenizer for accurate token counting (optional)
            chat_parser: Parser for converting messages to prompts (optional)
            trajectory_idx: Trajectory index for logging (optional)
            output_callback: Callback for logging warnings, signature: callback(idx, message)
        """
        self.tokenizer = tokenizer
        self.chat_parser = chat_parser
        self.trajectory_idx = trajectory_idx
        self.output_callback = output_callback

        # Token statistics
        self.prefix_tokens: int | None = None  # Tokens in first 2 messages (system + initial user)
        self.peak_context_tokens: int = 0  # Peak working tokens across trajectory
        self.total_input_tokens: int = 0  # Cumulative input tokens from API
        self.total_output_tokens: int = 0  # Cumulative output tokens from API

        # Debug: track per-step values for comparison
        self._debug_steps: list[dict] = []  # Each step: {api_prompt, api_working, context_status_working}

        # Warning state (one-time warnings per trajectory)
        self._warned = {
            "tokenizer_missing": False,
            "usage_missing": False,
        }

    def _warn(self, key: str, message: str):
        """Log a warning once per trajectory."""
        if not self._warned.get(key, False) and self.output_callback:
            self.output_callback(self.trajectory_idx, f"[TOKEN_STATS_FALLBACK] {message}")
            self._warned[key] = True

    def _estimate_by_chars(self, text: str) -> int:
        """Fallback: estimate tokens as chars/4."""
        return len(text) // self.CHARS_PER_TOKEN

    def _estimate_messages_by_chars(self, messages: list[dict]) -> int:
        """Fallback: estimate message tokens by summing content chars/4."""
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        total_chars += len(item["text"])
        return total_chars // self.CHARS_PER_TOKEN

    def _tokenize_text(self, text: str) -> int:
        """
        Estimate tokens for text.
        Priority: tokenizer > chars/4
        """
        if self.tokenizer is not None:
            try:
                return len(self.tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                pass

        # Fallback
        self._warn("tokenizer_missing", "Tokenizer unavailable; using chars/4 estimation.")
        return self._estimate_by_chars(text)

    def _tokenize_messages(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
        is_first_msg: bool = True,
    ) -> int:
        """
        Estimate tokens for messages.
        Priority: tokenizer + parser > chars/4
        """
        if self.tokenizer is not None and self.chat_parser is not None:
            try:
                prompt = self.chat_parser.parse(
                    messages,
                    add_generation_prompt=add_generation_prompt,
                    is_first_msg=is_first_msg
                )
                tokenizer_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
                # chars_div4 = self._estimate_messages_by_chars(messages)
                # print(f"[TOKEN_COMPARE] tokenizer={tokenizer_len}, chars/4={chars_div4}, ratio={tokenizer_len/chars_div4:.2f}x" if chars_div4 > 0 else f"[TOKEN_COMPARE] tokenizer={tokenizer_len}, chars/4=0")
                return tokenizer_len
            except Exception:
                pass

        # Fallback
        self._warn("tokenizer_missing", "Tokenizer unavailable; using chars/4 estimation.")
        return self._estimate_messages_by_chars(messages)

    def update_from_api_call(
        self,
        prompt_len: int | None,
        completion_len: int | None,
        prompt_messages: list,
        response: str,
    ):
        """
        Update statistics after an LLM API call.

        Token counting priority:
        1. Use API-provided counts (prompt_len, completion_len) if available ← PREFERRED
        2. Fall back to tokenizer estimation if API doesn't provide
        3. Fall back to chars/4 if tokenizer unavailable

        This method also:
        - Accumulates total_input_tokens and total_output_tokens
        - Updates peak_context_tokens (max working tokens across trajectory)

        Args:
            prompt_len: Actual prompt tokens from API (e.g., usage.prompt_tokens from OpenAI)
                       Pass None if unavailable
            completion_len: Actual completion tokens from API (e.g., usage.completion_tokens)
                           Pass None if unavailable
            prompt_messages: The messages sent to the API
            response: The response text from the API
        """
        # ========== Priority 1: Use API-provided token counts ==========
        if isinstance(prompt_len, int) and prompt_len > 0:
            actual_prompt_tokens = prompt_len
            prompt_source = "API"
            # DEBUG: Compare API vs tokenizer
            if self.tokenizer and self.chat_parser:
                tokenizer_est = self._tokenize_messages(prompt_messages)
                diff = prompt_len - tokenizer_est
                # msg_chars = sum(len(str(m.get('content', ''))) for m in prompt_messages)
                # msg_hash = hashlib.md5(str([(m.get('role'), len(str(m.get('content', '')))) for m in prompt_messages]).encode()).hexdigest()[:8]
                # step_num = len(self._debug_steps)
                # print(f"[API_VS_TOKENIZER] traj={self.trajectory_idx} step={step_num} API={prompt_len}, tokenizer={tokenizer_est}, diff={diff:+d}, num_msgs={len(prompt_messages)}, chars={msg_chars}, hash={msg_hash}")
        else:
            # ========== Priority 2/3: Estimate (tokenizer or chars/4) ==========
            self._warn("usage_missing", "API did not return prompt_length; estimating tokens.")
            actual_prompt_tokens = self._tokenize_messages(prompt_messages)
            prompt_source = "tokenizer" if (self.tokenizer and self.chat_parser) else "chars/4"

        if isinstance(completion_len, int) and completion_len > 0:
            actual_completion_tokens = completion_len
            completion_source = "API"
        else:
            self._warn("usage_missing", "API did not return completion_length; estimating tokens.")
            actual_completion_tokens = self._tokenize_text(response)
            completion_source = "tokenizer" if self.tokenizer else "chars/4"

        # Log token counting method
        if self.output_callback:
            self.output_callback(
                self.trajectory_idx,
                f"[TOKEN_COUNT] prompt={actual_prompt_tokens} (source: {prompt_source}), "
                f"completion={actual_completion_tokens} (source: {completion_source})"
            )

        # Accumulate totals (use ACTUAL counts from API when available)
        self.total_input_tokens += actual_prompt_tokens
        self.total_output_tokens += actual_completion_tokens

        # Calculate working tokens (prompt - prefix) and update peak
        if self.prefix_tokens is None and len(prompt_messages) >= 2:
            # Calculate prefix once (first 2 messages: system + initial user)
            self.prefix_tokens = self._tokenize_messages(
                prompt_messages[:2],
                add_generation_prompt=False,
                is_first_msg=True,
            )

        if self.prefix_tokens is not None:
            working_tokens = max(0, actual_prompt_tokens - self.prefix_tokens)
        else:
            working_tokens = actual_prompt_tokens

        self.peak_context_tokens = max(self.peak_context_tokens, working_tokens)

        # Debug: record this step's values
        self._debug_steps.append({
            "step": len(self._debug_steps),
            "api_prompt": actual_prompt_tokens,
            "api_working": working_tokens,
            "prefix_used": self.prefix_tokens,
        })

    def get_working_tokens(self, messages: list) -> int:
        """
        Get current working tokens (total - prefix).
        Used by agents for context status display.
        """
        if len(messages) <= 2:
            return 0

        total_tokens = self._tokenize_messages(messages)
        prefix_tokens = self._tokenize_messages(
            messages[:2],
            add_generation_prompt=False,
            is_first_msg=True
        )
        working = max(0, total_tokens - prefix_tokens)
        # DEBUG: Show CS calculation details
        # msg_chars = sum(len(str(m.get('content', ''))) for m in messages)
        # msg_hash = hashlib.md5(str([(m.get('role'), len(str(m.get('content', '')))) for m in messages]).encode()).hexdigest()[:8]
        # step_num = len(self._debug_steps) - 1 if self._debug_steps else -1
        # print(f"[CS_CALC] traj={self.trajectory_idx} step={step_num} total={total_tokens}, prefix={prefix_tokens}, working={working}, num_msgs={len(messages)}, chars={msg_chars}, hash={msg_hash}")
        return working

    def record_context_status(self, working_tokens: int):
        """Record context_status working tokens for the current step (for debug comparison)."""
        if self._debug_steps:
            # Add to the last recorded step
            self._debug_steps[-1]["context_status_working"] = working_tokens
            self._debug_steps[-1]["context_status_total"] = working_tokens + (self.prefix_tokens or 0)

    def print_debug_comparison(self):
        """Print per-step comparison of API vs Context Status values."""
        print(f"\n[DEBUG] Token comparison for trajectory {self.trajectory_idx}:")
        print(f"  prefix_tokens (tokenizer, no gen prompt) = {self.prefix_tokens}")
        print(f"  Peak Context (max api_working) = {self.peak_context_tokens}")
        print(f"  {'Step':<5} | {'API_prompt':<10} | {'API_working':<12} | {'CS_working':<12} | {'Diff':<8}")
        print(f"  {'-'*5} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*8}")

        max_api_working = 0
        max_cs_working = 0
        for d in self._debug_steps:
            step = d.get("step", "?")
            api_prompt = d.get("api_prompt", 0)
            api_working = d.get("api_working", 0)
            cs_working = d.get("context_status_working", "N/A")

            if isinstance(cs_working, int):
                diff = cs_working - api_working
                diff_str = f"{diff:+d}"
                max_cs_working = max(max_cs_working, cs_working)
            else:
                diff_str = "N/A"

            max_api_working = max(max_api_working, api_working)
            print(f"  {step:<5} | {api_prompt:<10} | {api_working:<12} | {str(cs_working):<12} | {diff_str:<8}")

        print(f"  {'-'*5} | {'-'*10} | {'-'*12} | {'-'*12} | {'-'*8}")
        print(f"  MAX:  | {'':<10} | {max_api_working:<12} | {max_cs_working:<12} | {max_cs_working - max_api_working:+d}")
        print()

    def get_total_tokens(self, messages: list) -> int:
        """Get total tokens for messages."""
        return self._tokenize_messages(messages)

    def get_stats(self) -> dict:
        """
        Get all statistics.

        Returns:
            Dict with:
            - peak_context_tokens: Max working tokens across trajectory
            - total_input_tokens: Cumulative prompt tokens (from API when available)
            - total_output_tokens: Cumulative completion tokens (from API when available)
        """
        return {
            "peak_context_tokens": self.peak_context_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
