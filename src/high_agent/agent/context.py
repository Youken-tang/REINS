"""Two-level context primitives."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TotalContext:
    """Model-visible run context.

    This stores only user objective, completed summaries, and compact runtime
    digests. Worker transcripts and scheduler internals stay in RuntimeLedger.
    """

    objective: str
    completed_summaries: list[str] = field(default_factory=list)
    runtime_digests: list[str] = field(default_factory=list)

    def add_delivery(self, summary: str, digest: str | None = None) -> None:
        if summary:
            self.completed_summaries.append(summary)
        if digest:
            self.runtime_digests.append(digest)

    def render(self, *, max_chars: int = 12_000) -> str:
        parts = [f"Objective: {self.objective}"]
        if self.completed_summaries:
            completed = "\n".join(f"- {item}" for item in self.completed_summaries[-24:])
            parts.append("Completed:\n" + completed)
        if self.runtime_digests:
            parts.append("Runtime:\n" + self.runtime_digests[-1])
        rendered = "\n\n".join(parts)
        if len(rendered) <= max_chars:
            return rendered
        omitted = len(rendered) - max_chars
        return f"{rendered[:max_chars]}\n[omitted {omitted} chars]"
