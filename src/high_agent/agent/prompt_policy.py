"""Hermes-style generic tool execution discipline."""

from __future__ import annotations

from typing import Any


def workspace_context_snippet(workspace_root: str | None) -> str:
    """One-liner injected into agent system prompts so the model knows cwd.

    F1 fix: previously every system prompt (main / worker / agent_loop) was
    silent about ``runtime.workspace_root``. The model could only guess cwd
    from the user prompt, which led to scaffold tasks writing root-level
    config files (``pyproject.toml`` / ``.gitignore`` / ``Dockerfile`` / ...)
    into the high-agent workspace itself when the user asked for a sibling
    project. Telling the model the resolved root + how to interpret a target
    directory in the user request closes that gap.
    """
    if not workspace_root:
        return ""
    return (
        "# Workspace\n"
        f"Your current workspace root is `{workspace_root}`. Relative paths "
        "in tool calls resolve under this root. Treat this directory as "
        "read/edit scope for the active project — do NOT scaffold a new "
        "project, overwrite root-level config files (pyproject.toml, "
        ".gitignore, Dockerfile, README.md, etc.), or create unrelated trees "
        "directly under it.\n"
        "When the user asks you to build a NEW project at some target "
        "location (e.g. \"在 /path/to/foo 下构建 ...\", \"create a project "
        "in ~/bar\"), put every file of that new project under that target "
        "directory using absolute paths. If the target is ambiguous (e.g. "
        "\"home/webtest\"), prefer `$HOME/<name>` (an absolute path under "
        "the user's home) and never fall back to the workspace root."
    )


TOOL_USE_ENFORCEMENT_MODELS = ("gpt", "codex", "gemini", "gemma", "grok")

TOOL_USE_ENFORCEMENT_GUIDANCE = (
    "# Tool-use enforcement\n"
    "Use tools to take action instead of describing future intent. If you say "
    "you will inspect, create, edit, run, or verify something, make the tool call "
    "in the same response. Keep working until the task is actually complete or "
    "blocked, and do not end with a promise of future work."
)

OPENAI_MODEL_EXECUTION_GUIDANCE = (
    "# Execution discipline\n"
    "<tool_persistence>\n"
    "- Use tools whenever they improve correctness or completeness.\n"
    "- Do not stop early when another tool call would materially improve the result.\n"
    "- Keep calling tools until the task is complete and verified.\n"
    "</tool_persistence>\n\n"
    "<prerequisite_checks>\n"
    "- Check required project structure, files, dependencies, and command outputs "
    "before making assumptions.\n"
    "</prerequisite_checks>\n\n"
    "<verification>\n"
    "- Before finalizing, verify that the requested work is actually done.\n"
    "- If tests or commands fail, continue fixing when possible instead of claiming success.\n"
    "</verification>"
)

GOOGLE_MODEL_OPERATIONAL_GUIDANCE = (
    "# Google model operational directives\n"
    "- Use absolute paths for file operations when available.\n"
    "- Inspect files and dependencies before editing.\n"
    "- Make independent tool calls in parallel when possible.\n"
    "- Use non-interactive command flags where appropriate.\n"
    "- Continue until the task is fully resolved or clearly blocked."
)


class PromptPolicy:
    def __init__(self, enforcement: Any = "auto") -> None:
        self.enforcement = enforcement

    def build(
        self,
        *,
        base_prompt: str,
        model_name: str = "",
        has_tools: bool = True,
        workspace_root: str | None = None,
    ) -> str:
        parts = [base_prompt]
        ws = workspace_context_snippet(workspace_root)
        if ws:
            parts.append(ws)
        if not has_tools or not self._should_inject(model_name):
            return "\n\n".join(parts)
        parts.append(TOOL_USE_ENFORCEMENT_GUIDANCE)
        model_lower = model_name.lower()
        if "gpt" in model_lower or "codex" in model_lower:
            parts.append(OPENAI_MODEL_EXECUTION_GUIDANCE)
        if "gemini" in model_lower or "gemma" in model_lower:
            parts.append(GOOGLE_MODEL_OPERATIONAL_GUIDANCE)
        return "\n\n".join(parts)

    def _should_inject(self, model_name: str) -> bool:
        value = self.enforcement
        if value is True:
            return True
        if value is False:
            return False
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "always", "yes", "on"}:
                return True
            if lowered in {"false", "never", "no", "off"}:
                return False
            if lowered == "auto":
                return any(token in model_name.lower() for token in TOOL_USE_ENFORCEMENT_MODELS)
            return lowered in model_name.lower()
        if isinstance(value, list):
            return any(str(token).lower() in model_name.lower() for token in value)
        return any(token in model_name.lower() for token in TOOL_USE_ENFORCEMENT_MODELS)
