"""Interactive model setup for high-agent."""

from __future__ import annotations

import getpass
import sys
from collections.abc import Callable
from typing import TextIO

try:
    from rich.console import Console
except ModuleNotFoundError:  # pragma: no cover - minimal env guidance path
    class Console:  # type: ignore[no-redef]
        def print(self, *values: object, **kwargs: object) -> None:
            text = " ".join(str(value) for value in values)
            for token in ("[bold]", "[/bold]", "[red]", "[/red]", "[yellow]", "[/yellow]", "[green]", "[/green]"):
                text = text.replace(token, "")
            print(text)

from high_agent.config import ConfigPaths, get_config_paths, load_config, load_secrets, save_config, save_secrets, set_nested
from high_agent.llm.probe import probe_models
from high_agent.llm.providers import (
    ModelSettings,
    detect_api_mode,
    get_provider,
    list_supported_providers,
)


def print_noninteractive_setup_guidance(*, file: TextIO | None = None) -> None:
    out = file or sys.stderr
    paths = get_config_paths()
    print("high-agent 还没有可用模型配置。", file=out)
    print("", file=out)
    print("交互式配置：", file=out)
    print("  high-agent setup", file=out)
    print("", file=out)
    print("非交互式环境变量：", file=out)
    print("  HIGH_AGENT_PROVIDER=openai", file=out)
    print("  HIGH_AGENT_MODEL=gpt-5.1", file=out)
    print("  HIGH_AGENT_API_KEY=...", file=out)
    print("  HIGH_AGENT_BASE_URL=https://api.openai.com/v1", file=out)
    print("  HIGH_AGENT_API_MODE=codex_responses", file=out)
    print("", file=out)
    print(f"配置目录：{paths.home}", file=out)


def run_setup(
    *,
    non_interactive: bool = False,
    paths: ConfigPaths | None = None,
    input_func: Callable[[str], str] = input,
    getpass_func: Callable[[str], str] = getpass.getpass,
) -> int:
    if non_interactive:
        print_noninteractive_setup_guidance(file=sys.stdout)
        return 0

    paths = paths or get_config_paths()
    console = Console()
    config = load_config(paths)
    secrets = load_secrets(paths)
    providers = list_supported_providers()

    console.print("[bold]high-agent 模型设置[/bold]")
    for index, provider in enumerate(providers, start=1):
        suffix = "" if provider.supported else f" ({provider.auth_type}, 暂不直接支持)"
        console.print(f"  {index}. {provider.id} - {provider.name}{suffix}")

    current_model = config.get("model") if isinstance(config.get("model"), dict) else {}
    default_provider = str(current_model.get("provider") or "custom")
    provider_raw = _ask(input_func, f"Provider [{default_provider}]: ", default_provider)
    if provider_raw.isdigit() and 1 <= int(provider_raw) <= len(providers):
        provider_raw = providers[int(provider_raw) - 1].id
    provider = get_provider(provider_raw)
    if not provider.supported:
        console.print(f"[red]该 provider 当前不直接支持：{provider.note or provider.auth_type}[/red]")
        return 2

    default_base = str(current_model.get("base_url") or provider.base_url)
    base_url = _ask(input_func, f"Base URL [{default_base}]: ", default_base)
    default_mode = detect_api_mode(base_url, provider.id, current_model.get("api_mode"))
    api_mode = _ask(input_func, f"API mode [{default_mode}]: ", default_mode)

    secret_cfg = secrets.get("providers", {}).get(provider.id, {}) if isinstance(secrets.get("providers"), dict) else {}
    has_existing_key = bool(secret_cfg.get("api_key"))
    needs_key = provider.id not in {"lmstudio", "ollama"}
    api_key = ""
    if needs_key:
        prompt = "API key [保持已有]: " if has_existing_key else "API key: "
        api_key = getpass_func(prompt).strip()

    probe_settings = ModelSettings(
        provider=provider.id,
        model="",
        base_url=base_url,
        api_mode=detect_api_mode(base_url, provider.id, api_mode),
        api_key=api_key or str(secret_cfg.get("api_key") or ""),
    )
    probe = probe_models(probe_settings)
    if probe.models:
        console.print("可用模型：")
        for index, model in enumerate(probe.models[:20], start=1):
            console.print(f"  {index}. {model}")
    elif probe.error:
        console.print(f"[yellow]模型列表探测失败：{probe.error}。可以继续手填模型名。[/yellow]")

    default_model = str(current_model.get("model") or current_model.get("default") or (probe.models[0] if len(probe.models) == 1 else ""))
    model = _ask(input_func, f"Model [{default_model}]: ", default_model)
    if model.isdigit() and probe.models and 1 <= int(model) <= len(probe.models):
        model = probe.models[int(model) - 1]
    if not model:
        console.print("[red]必须配置 model。[/red]")
        return 2

    set_nested(config, ("model", "provider"), provider.id)
    set_nested(config, ("model", "model"), model)
    set_nested(config, ("model", "base_url"), base_url)
    set_nested(config, ("model", "api_mode"), detect_api_mode(base_url, provider.id, api_mode))
    config.setdefault("runtime", {})
    config["runtime"].setdefault("max_workers", 8)
    config["runtime"].setdefault("max_planner_requests", 4)

    if api_key:
        set_nested(secrets, ("providers", provider.id, "api_key"), api_key)

    save_config(config, paths)
    save_secrets(secrets, paths)
    console.print(f"[green]已保存配置：{paths.config_path}[/green]")
    return 0


def _ask(input_func: Callable[[str], str], prompt: str, default: str) -> str:
    value = input_func(prompt).strip()
    return value or default
