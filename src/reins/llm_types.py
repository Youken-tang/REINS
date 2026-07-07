"""Provider-neutral LLM types."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ToolCall:
    id: str | None
    name: str
    arguments: str
    provider_data: dict[str, Any] | None = field(default=None, repr=False)

    def args_dict(self) -> dict[str, Any]:
        if not self.arguments:
            return {}
        data = json.loads(self.arguments)
        if not isinstance(data, dict):
            raise ValueError("tool call arguments must decode to an object")
        return data


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(slots=True)
class NormalizedResponse:
    content: str | None
    tool_calls: list[ToolCall] | None
    finish_reason: str
    usage: Usage | None = None
    provider_data: dict[str, Any] | None = field(default=None, repr=False)
