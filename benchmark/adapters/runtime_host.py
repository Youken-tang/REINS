"""Runtime host for the REINS-into-hermes bridge/W2 deliverable).

This is the ``runtime_host.py`` half of the bridge described in
```` and ````
 The matching :mod:`benchmark.adapters.hermes_reins_loop` module is the
``loop.py`` half (drop-in for ``_execute_tool_calls``).

Responsibilities
----------------

1. Construct one process-wide :class:`CausalRuntime` with the right
   knobs for hermes (``strict_nogil=False`` so the bridge stays
   GIL-tolerant for artifact reproducibility, generous ``max_workers``,
   modest ``delivery_debounce``). Note: in the actual sweep ran
   under the project-wide ``.venv/`` (3.13t noGIL) — the v1 plan to
   put hermes on its own ``.venv-gil/`` was retired in v2 (see
   ```` "v1 → v2 偏差记录"). The
   ``strict_nogil=False`` flag remains because (a) it lets the bridge
   reproduce on a stock CPython 3.13 if a reviewer needs it, and (b)
 wall-clock decomposition predicts ≤5–10% wall difference
   either way.
2. Read environment-flag overrides so the sweep can flip
   ``hermes-vanilla`` / ``hermes-REINS`` and tune ``max_workers``
   without code changes:

   * ``HERMES_REINS=0`` — bridge disabled, vanilla path
     (delegated to :mod:`benchmark.adapters.hermes_reins_loop`).
   * ``HERMES_REINS_MAX_WORKERS`` — overrides ``max_workers``.
   * ``HERMES_REINS_DEBOUNCE`` — overrides ``delivery_debounce``.
   * ``HERMES_REINS_TRACE_PATH`` — writes runtime trace to this path
     (consumed by the W3 sweep harness via
     ``benchmark/scripts/parse_trace.py``).
   * ``HERMES_REINS_TIMEOUT`` — per-batch dispatch timeout.

3. Expose a single-call ``dispatch_tool_calls`` shim that the
   call-site (``_execute_tool_calls`` swap target at hermes
   ``run_agent.py:9098``) invokes with no setup other than handing
   over hermes' own ``invoke_tool`` callback. The host owns the
   runtime lifetime, not the call site.

This module is exactly what would land at
``hermes-agent/reins_bridge/runtime_host.py`` once the bridge is
wired in-tree. It lives here so the benchmark's test surface can
exercise it without modifying ``hermes-agent/`` (CLAUDE.md rule).
"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from reins import CausalRuntime

from benchmark.adapters.hermes_reins_loop import execute_tool_calls

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Env-flag plumbing
# ---------------------------------------------------------------------------

_DEFAULT_MAX_WORKERS = 8
# hermes is 1-batch-per-call (the planner makes a
# model call, gets a tool batch, dispatches it, waits, then makes the next
# model call). Debounce coalesces *deliveries* into batches for the
# planner — but the planner only consumes one batch at a time and has
# already moved on by the time the next would arrive. The 0.05s default
# was inherited from high_agent's controller-driven loop where it makes
# sense; on hermes it's pure wall-time tax (10 ms × N tools per cell).
_DEFAULT_DELIVERY_DEBOUNCE = 0.0
_DEFAULT_TIMEOUT_SECONDS = 600.0

ENV_MAX_WORKERS = "HERMES_REINS_MAX_WORKERS"
ENV_DEBOUNCE = "HERMES_REINS_DEBOUNCE"
ENV_TRACE_PATH = "HERMES_REINS_TRACE_PATH"
ENV_TIMEOUT = "HERMES_REINS_TIMEOUT"


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        logger.warning("invalid %s=%r — using default %d", name, raw, default)
        return default
    if value <= 0:
        logger.warning("%s=%d must be positive — using default %d", name, value, default)
        return default
    return value


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError:
        logger.warning("invalid %s=%r — using default %g", name, raw, default)
        return default
    if value < 0:
        logger.warning("%s=%g must be ≥ 0 — using default %g", name, value, default)
        return default
    return value


# ---------------------------------------------------------------------------
# Host singleton
# ---------------------------------------------------------------------------


class HermesReinsHost:
    """Process-wide REINS runtime host for hermes.

    Hermes' agent loop wants to construct the runtime once at startup
    and reuse it across many ``_execute_tool_calls`` calls. The host
    owns the :class:`CausalRuntime`'s lifetime so the call-site code
    stays one-line:

    .. code-block:: python

        host = HermesReinsHost(workspace_root=cwd)
        host.start()
        ...
        # in _execute_tool_calls:
        host.dispatch(assistant_message, messages, invoke_tool=self._invoke_tool)
        ...
        host.shutdown()

    Construction is cheap (no threads); ``start()`` boots the
    underlying executor; ``shutdown()`` is idempotent so atexit hooks
    don't crash the agent on second pass.

    The host does *not* own hermes' tool registry — hermes still
    decides what each tool does. The host only owns scheduling.
    """

    def __init__(
        self,
        *,
        workspace_root: str | Path | None = None,
        max_workers: int | None = None,
        delivery_debounce: float | None = None,
        trace_path: str | Path | None = None,
        timeout_seconds: float | None = None,
        strict_nogil: bool = False,  # bridge stays GIL-tolerant; sweep nonetheless ran on 3.13t (CLAUDE.md project-wide constraint)
    ) -> None:
        self.workspace_root: str | None = (
            str(workspace_root) if workspace_root is not None else None
        )
        self.max_workers = max_workers or _env_int(ENV_MAX_WORKERS, _DEFAULT_MAX_WORKERS)
        self.delivery_debounce = (
            delivery_debounce if delivery_debounce is not None
            else _env_float(ENV_DEBOUNCE, _DEFAULT_DELIVERY_DEBOUNCE)
        )
        env_trace = os.environ.get(ENV_TRACE_PATH)
        self.trace_path: str | Path | None = trace_path or env_trace
        self.timeout_seconds = (
            timeout_seconds if timeout_seconds is not None
            else _env_float(ENV_TIMEOUT, _DEFAULT_TIMEOUT_SECONDS)
        )
        self.strict_nogil = strict_nogil

        self._runtime: CausalRuntime | None = None
        self._lock = threading.Lock()
        self._started = False
        self._shutdown = False

    # ---- lifecycle ------------------------------------------------

    def start(self) -> CausalRuntime:
        """Construct + start the underlying runtime. Idempotent.

        Returns the live :class:`CausalRuntime` so callers that need
        direct access (custom ``submit`` paths, trace introspection)
        don't have to dig through ``host._runtime``.
        """
        with self._lock:
            if self._shutdown:
                raise RuntimeError("HermesReinsHost: cannot start after shutdown")
            if self._started:
                assert self._runtime is not None
                return self._runtime
            self._runtime = CausalRuntime(
                max_workers=self.max_workers,
                workspace_root=self.workspace_root or ".",
                delivery_debounce=self.delivery_debounce,
                trace_path=self.trace_path,
                strict_nogil=self.strict_nogil,
            )
            self._started = True
            logger.info(
                "HermesReinsHost started: max_workers=%d debounce=%.3f trace=%s",
                self.max_workers, self.delivery_debounce, self.trace_path,
            )
            return self._runtime

    def shutdown(self) -> None:
        """Tear down the runtime. Idempotent — safe in atexit hooks."""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
            runtime = self._runtime
            self._runtime = None
        if runtime is not None:
            try:
                runtime.shutdown()
            except Exception:  # noqa: BLE001 — atexit must not raise
                logger.exception("HermesReinsHost: runtime.shutdown() raised")

    @property
    def runtime(self) -> CausalRuntime | None:
        """The underlying :class:`CausalRuntime`, or None before start.

        Most callers should use :meth:`dispatch` instead — exposed
        only for the sweep harness which needs trace flushing.
        """
        return self._runtime

    @property
    def started(self) -> bool:
        return self._started and not self._shutdown

    # ---- main entry point -----------------------------------------

    def dispatch(
        self,
        assistant_message: Any,
        messages: list,
        *,
        invoke_tool: Callable[[str, dict[str, Any]], Any],
        timeout_seconds: float | None = None,
    ) -> None:
        """Drop-in body for hermes' ``_execute_tool_calls``.

        Auto-starts the runtime on first call. ``HERMES_REINS=0``
        falls back to sequential dispatch in
        :func:`benchmark.adapters.hermes_reins_loop.execute_tool_calls`
        without consulting the runtime, so vanilla runs incur no
        runtime-startup cost.

        ``timeout_seconds`` overrides the host-level default for this
        call; mostly useful in tests or for tools the caller knows
        will run a long time (long ``terminal`` builds, etc.).
        """
        if not self._started and not self._shutdown:
            self.start()
        runtime = self._runtime  # may be None when HERMES_REINS=0
        execute_tool_calls(
            assistant_message,
            messages,
            invoke_tool=invoke_tool,
            runtime=runtime,
            root=self.workspace_root,
            timeout_seconds=timeout_seconds or self.timeout_seconds,
        )

    # ---- context-manager sugar ------------------------------------

    def __enter__(self) -> "HermesReinsHost":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.shutdown()


# ---------------------------------------------------------------------------
# Module-level convenience: a default host for the «one process, one runtime»
# call site at hermes ``run_agent.py:9098``. Hermes constructs an AIAgent
# instance per session, so the host is keyed off the workspace root.
# ---------------------------------------------------------------------------


_default_host: HermesReinsHost | None = None
_default_host_lock = threading.Lock()


def get_default_host(
    *,
    workspace_root: str | Path | None = None,
) -> HermesReinsHost:
    """Return (and lazily build) the default process-wide host.

    Hermes' main loop is single-process; one host per process keeps
    the runtime alive across the agent's many tool batches. The
    workspace root is captured on first call — subsequent calls
    ignore the parameter, matching hermes' own «cwd is fixed at
    startup» behaviour.
    """
    global _default_host
    with _default_host_lock:
        if _default_host is None:
            _default_host = HermesReinsHost(workspace_root=workspace_root)
    return _default_host


def reset_default_host() -> None:
    """Tear down + clear the singleton. Used by tests; safe in
    production atexit hooks too.
    """
    global _default_host
    with _default_host_lock:
        host = _default_host
        _default_host = None
    if host is not None:
        host.shutdown()


__all__ = [
    "ENV_DEBOUNCE",
    "ENV_MAX_WORKERS",
    "ENV_TIMEOUT",
    "ENV_TRACE_PATH",
    "HermesReinsHost",
    "get_default_host",
    "reset_default_host",
]
