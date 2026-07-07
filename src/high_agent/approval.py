"""Approval policy primitives for side-effectful actions."""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

ApprovalPolicy = Literal["ask", "auto", "deny"]
ApprovalScope = Literal["once", "session", "always"]


@dataclass(frozen=True)
class ApprovalRequest:
    action: str
    resource: str
    reason: str = ""
    risk: str = "medium"
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.action}:{self.resource}"


@dataclass(frozen=True)
class ApprovalDecision:
    approved: bool
    scope: ApprovalScope = "once"
    reason: str = ""


DecisionCallback = Callable[[ApprovalRequest], ApprovalDecision]


class ApprovalManager:
    def __init__(
        self,
        *,
        policy: ApprovalPolicy = "ask",
        callback: DecisionCallback | None = None,
        input_func: Callable[[str], str] = input,
        output_func: Callable[[str], None] = print,
        interactive: bool | None = None,
    ) -> None:
        self.policy = policy
        self.callback = callback
        self.input_func = input_func
        self.output_func = output_func
        self.interactive = sys.stdin.isatty() if interactive is None else interactive
        self._session: dict[str, ApprovalDecision] = {}
        self._always: dict[str, ApprovalDecision] = {}

    def request(self, request: ApprovalRequest) -> ApprovalDecision:
        if request.key in self._always:
            return self._always[request.key]
        if request.key in self._session:
            return self._session[request.key]
        if self.policy == "auto":
            return self._remember(request, ApprovalDecision(True, "session", "auto policy"))
        if self.policy == "deny":
            return ApprovalDecision(False, "once", "deny policy")
        if self.callback is not None:
            return self._remember(request, self.callback(request))
        if not self.interactive:
            return ApprovalDecision(False, "once", "approval required in non-interactive mode")
        self.output_func(f"Approval required: {request.action} {request.resource}")
        if request.reason:
            self.output_func(f"Reason: {request.reason}")
        answer = self.input_func("Allow? [y/N/s=允许本会话/a=永久允许] ").strip().lower()
        if answer in {"a", "always"}:
            return self._remember(request, ApprovalDecision(True, "always", "user approved always"))
        if answer in {"s", "session"}:
            return self._remember(request, ApprovalDecision(True, "session", "user approved session"))
        if answer in {"y", "yes"}:
            return ApprovalDecision(True, "once", "user approved once")
        return ApprovalDecision(False, "once", "user denied")

    def _remember(self, request: ApprovalRequest, decision: ApprovalDecision) -> ApprovalDecision:
        if decision.scope == "always":
            self._always[request.key] = decision
        elif decision.scope == "session":
            self._session[request.key] = decision
        return decision
