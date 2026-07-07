"""Provider catalog and runtime model resolution.

The catalog is intentionally small compared with Hermes, but it preserves the
same shape: provider identity, default endpoint, API mode, and credential env
vars are resolved in one place.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlparse

from high_agent.config import get_nested

ApiMode = Literal["chat_completions", "anthropic_messages", "codex_responses"]

VALID_API_MODES: set[str] = {"chat_completions", "anthropic_messages", "codex_responses"}


@dataclass(frozen=True)
class ProviderDef:
    id: str
    name: str
    base_url: str
    api_mode: ApiMode
    api_key_env_vars: tuple[str, ...]
    auth_type: str = "api_key"
    note: str = ""

    @property
    def supported(self) -> bool:
        return self.auth_type == "api_key"


@dataclass(frozen=True)
class ModelSettings:
    provider: str
    model: str
    base_url: str
    api_mode: ApiMode
    api_key: str = ""
    # Optional pool of API keys (e.g. 4 NewAPI keys) the ModelClient can
    # round-robin across to spread concurrent multi-planner requests.
    # Empty / single-element tuple → behave like the single api_key field.
    api_keys: tuple[str, ...] = ()
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ResolvedModelConfig:
    settings: ModelSettings
    provider_def: ProviderDef
    source: dict[str, str] = field(default_factory=dict)


PROVIDERS: dict[str, ProviderDef] = {
    "custom": ProviderDef("custom", "Custom OpenAI-compatible", "", "chat_completions", ("HIGH_AGENT_API_KEY",)),
    "openai": ProviderDef("openai", "OpenAI", "https://api.openai.com/v1", "codex_responses", ("OPENAI_API_KEY",)),
    "openrouter": ProviderDef("openrouter", "OpenRouter", "https://openrouter.ai/api/v1", "chat_completions", ("OPENROUTER_API_KEY", "OPENAI_API_KEY")),
    "anthropic": ProviderDef("anthropic", "Anthropic", "https://api.anthropic.com", "anthropic_messages", ("ANTHROPIC_API_KEY", "ANTHROPIC_TOKEN")),
    "xai": ProviderDef("xai", "xAI", "https://api.x.ai/v1", "codex_responses", ("XAI_API_KEY",)),
    "deepseek": ProviderDef("deepseek", "DeepSeek", "https://api.deepseek.com/v1", "chat_completions", ("DEEPSEEK_API_KEY",)),
    "alibaba": ProviderDef("alibaba", "Alibaba DashScope", "https://dashscope.aliyuncs.com/compatible-mode/v1", "chat_completions", ("DASHSCOPE_API_KEY", "ALIBABA_API_KEY")),
    "zai": ProviderDef("zai", "Z.ai / GLM", "https://open.bigmodel.cn/api/paas/v4", "chat_completions", ("ZAI_API_KEY", "GLM_API_KEY", "Z_AI_API_KEY")),
    "kimi": ProviderDef("kimi", "Moonshot Kimi", "https://api.moonshot.cn/v1", "chat_completions", ("MOONSHOT_API_KEY", "KIMI_API_KEY")),
    "kimi-coding": ProviderDef("kimi-coding", "Kimi Coding", "https://api.kimi.com/coding", "anthropic_messages", ("KIMI_API_KEY",)),
    "minimax": ProviderDef("minimax", "MiniMax", "https://api.minimax.io/anthropic", "anthropic_messages", ("MINIMAX_API_KEY",)),
    "lmstudio": ProviderDef("lmstudio", "LM Studio", "http://127.0.0.1:1234/v1", "chat_completions", ("LM_API_KEY",)),
    "ollama": ProviderDef("ollama", "Ollama", "http://127.0.0.1:11434/v1", "chat_completions", ("OLLAMA_API_KEY",)),
    "bedrock": ProviderDef("bedrock", "AWS Bedrock", "", "anthropic_messages", ("AWS_ACCESS_KEY_ID",), auth_type="aws_sdk", note="v1 不直接支持 Bedrock SDK，请使用兼容网关或自定义 endpoint。"),
    "copilot-acp": ProviderDef("copilot-acp", "GitHub Copilot ACP", "acp://copilot", "codex_responses", ("COPILOT_GITHUB_TOKEN",), auth_type="external_process", note="v1 不启动外部 ACP 进程。"),
}

ALIASES = {
    "openai-chat": "openai",
    "openai-codex": "openai",
    "openrouter.ai": "openrouter",
    "claude": "anthropic",
    "anthropic-messages": "anthropic",
    "x.ai": "xai",
    "x-ai": "xai",
    "deepseek-chat": "deepseek",
    "dashscope": "alibaba",
    "glm": "zai",
    "z-ai": "zai",
    "moonshot": "kimi",
    "kimi-for-coding": "kimi-coding",
    "local": "lmstudio",
}


def provider_id(value: str | None) -> str:
    raw = (value or "custom").strip().lower()
    return ALIASES.get(raw, raw)


def get_provider(value: str | None) -> ProviderDef:
    resolved = provider_id(value)
    if resolved not in PROVIDERS:
        return ProviderDef(resolved, resolved, "", "chat_completions", ("HIGH_AGENT_API_KEY",))
    return PROVIDERS[resolved]


def list_supported_providers() -> list[ProviderDef]:
    return sorted(PROVIDERS.values(), key=lambda item: item.id)


def detect_api_mode(base_url: str, provider: str | None = None, configured: str | None = None) -> ApiMode:
    parsed = _parse_api_mode(configured)
    if parsed:
        return parsed
    normalized = (base_url or "").strip().lower().rstrip("/")
    hostname = urlparse(normalized).hostname or ""
    if hostname == "api.x.ai":
        return "codex_responses"
    if hostname == "api.openai.com":
        return "codex_responses"
    if normalized.endswith("/anthropic"):
        return "anthropic_messages"
    if hostname == "api.kimi.com" and "/coding" in normalized:
        return "anthropic_messages"
    return get_provider(provider).api_mode


def resolve_model_config(
    config: dict[str, Any],
    secrets: dict[str, Any],
    *,
    cli_overrides: dict[str, str | None] | None = None,
    env: dict[str, str] | None = None,
) -> ResolvedModelConfig:
    values = env or os.environ
    overrides = cli_overrides or {}
    model_cfg = config.get("model") if isinstance(config.get("model"), dict) else {}
    runtime_provider = (
        _first(overrides.get("provider"), values.get("HIGH_AGENT_PROVIDER"), get_nested(config, ("model", "provider")))
        or "custom"
    )
    provider = get_provider(runtime_provider)
    base_url = _first(overrides.get("base_url"), values.get("HIGH_AGENT_BASE_URL"), get_nested(config, ("model", "base_url")), provider.base_url)
    model = _first(overrides.get("model"), values.get("HIGH_AGENT_MODEL"), model_cfg.get("model"), model_cfg.get("default"))
    api_mode = detect_api_mode(
        base_url or "",
        provider.id,
        _first(overrides.get("api_mode"), values.get("HIGH_AGENT_API_MODE"), model_cfg.get("api_mode")),
    )
    api_key = _first(overrides.get("api_key"), values.get("HIGH_AGENT_API_KEY"))
    key_source = "cli/env" if api_key else ""
    if not api_key:
        for env_name in provider.api_key_env_vars:
            api_key = values.get(env_name, "").strip()
            if api_key:
                key_source = env_name
                break
    if not api_key:
        api_key = str(get_nested(secrets, ("providers", provider.id, "api_key"), "") or "").strip()
        key_source = "secrets" if api_key else ""
    return ResolvedModelConfig(
        settings=ModelSettings(
            provider=provider.id,
            model=str(model or ""),
            base_url=str(base_url or ""),
            api_mode=api_mode,
            api_key=str(api_key or ""),
        ),
        provider_def=provider,
        source={
            "provider": str(runtime_provider),
            "base_url": "resolved",
            "api_key": key_source,
        },
    )


def missing_config_reason(resolved: ResolvedModelConfig) -> str | None:
    if not resolved.provider_def.supported:
        return resolved.provider_def.note or f"provider {resolved.provider_def.id} needs unsupported auth: {resolved.provider_def.auth_type}"
    if not resolved.settings.model:
        return "missing model"
    if not resolved.settings.base_url:
        return "missing base_url"
    if not resolved.settings.api_key and resolved.settings.provider not in {"lmstudio", "ollama"}:
        return "missing api_key"
    return None


def _parse_api_mode(raw: Any) -> ApiMode | None:
    value = str(raw or "").strip()
    if value in VALID_API_MODES:
        return value  # type: ignore[return-value]
    return None


def _first(*values: Any) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""
