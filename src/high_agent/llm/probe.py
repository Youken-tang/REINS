"""Model endpoint probing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    import httpx
except ModuleNotFoundError:  # pragma: no cover - minimal env guidance path
    httpx = None

from high_agent.llm.providers import ModelSettings


@dataclass(frozen=True)
class ModelProbeResult:
    ok: bool
    models: list[str] = field(default_factory=list)
    error: str = ""


def probe_models(settings: ModelSettings, *, timeout: float = 8.0,
                 http_client: Any = None) -> ModelProbeResult:
    if httpx is None and http_client is None:
        return ModelProbeResult(ok=False, error="httpx is not installed")
    if not settings.base_url:
        return ModelProbeResult(ok=False, error="missing base_url")
    url = _models_url(settings.base_url)
    headers = {"accept": "application/json"}
    if settings.api_key:
        if settings.api_mode == "anthropic_messages":
            headers["x-api-key"] = settings.api_key
            headers["anthropic-version"] = "2023-06-01"
        else:
            headers["authorization"] = f"Bearer {settings.api_key}"
    client = http_client or httpx.Client(timeout=timeout)
    close_client = http_client is None
    try:
        response = client.get(url, headers=headers)
        if response.status_code >= 400:
            return ModelProbeResult(ok=False, error=f"GET {url} -> {response.status_code}")
        payload = response.json()
        models = _extract_models(payload)
        return ModelProbeResult(ok=True, models=models)
    except Exception as exc:
        return ModelProbeResult(ok=False, error=str(exc))
    finally:
        if close_client:
            client.close()


def _models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/models"):
        return base
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _extract_models(payload: object) -> list[str]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get("data") or payload.get("models") or []
    out: list[str] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict) and item.get("id"):
                out.append(str(item["id"]))
            elif isinstance(item, str):
                out.append(item)
    return sorted(set(out))
