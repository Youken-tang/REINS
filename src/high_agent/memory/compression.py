"""Deterministic context compression for interactive sessions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CompressionResult:
    compressed: bool
    messages: list[dict[str, Any]]
    summary: str = ""
    original_count: int = 0
    retained_count: int = 0
    created_at: float = field(default_factory=time.time)


class ContextCompressor:
    """Small local compressor used before model-backed compression exists.

    The compressor intentionally stores a compact factual summary instead of
    preserving full tool transcripts. It is deterministic so tests can assert
    exact behavior without calling a model.
    """

    def __init__(self, *, keep_recent: int = 12, max_item_chars: int = 500) -> None:
        self.keep_recent = max(2, int(keep_recent))
        self.max_item_chars = max(80, int(max_item_chars))

    def maybe_compress(self, messages: list[dict[str, Any]], budget: int = 24_000) -> CompressionResult:
        total_chars = sum(len(str(item.get("content") or "")) for item in messages)
        if total_chars <= budget and len(messages) <= self.keep_recent * 2:
            return CompressionResult(
                compressed=False,
                messages=list(messages),
                original_count=len(messages),
                retained_count=len(messages),
            )

        recent = list(messages[-self.keep_recent:])
        older = list(messages[:-self.keep_recent])
        bullets = []
        for item in older:
            role = str(item.get("role") or "unknown")
            if role not in {"user", "assistant", "runtime", "system"}:
                continue
            content = " ".join(str(item.get("content") or "").split())
            if not content:
                continue
            bullets.append(f"- {role}: {content[:self.max_item_chars]}")
        summary = "Context compressed from earlier session messages:\n" + "\n".join(bullets[-40:])
        compressed_messages = [{"role": "system", "content": summary}, *recent]
        return CompressionResult(
            compressed=True,
            messages=compressed_messages,
            summary=summary,
            original_count=len(messages),
            retained_count=len(compressed_messages),
        )
