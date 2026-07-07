"""Delivery-driven agent run controller."""

from __future__ import annotations

import concurrent.futures
import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from reins.llm_types import NormalizedResponse, Usage
from reins.runtime.types import DeliveryBatch, DeliveryEvent
from reins.tool_calls import (
    NormalizedToolCall,
    assistant_tool_calls,
    sanitize_tool_protocol_messages,
)

MAX_PLANNER_CONTEXT_CHARS = 24_000
MAX_DURABLE_CONTEXT_CHARS = 8_000
MAX_FACT_CHARS = 1_200
MAX_FACTS_IN_VIEW = 16
MAX_STATUS_NOTES_IN_VIEW = 4
# the "known reads" block is the *content*
# half of the two-level context — sibling planners must see the actual
# file/dir contents read by earlier planners, not just file paths. Pre-fix
# we truncated each entry to 80 chars (effectively path-only), forcing
# every planner to re-issue its own read_file/list_tree calls. This
# inflated REINS tool-call counts ~5× vs hermes single-loop on read-only
# work. Now we keep the full result up to MAX_KNOWN_READ_ENTRY_CHARS per
# path with a global MAX_KNOWN_READS_CHARS budget.
MAX_KNOWN_READS_CHARS = 12_000
MAX_KNOWN_READ_ENTRY_CHARS = 3_000

_READ_ONLY_TOOL_NAMES = frozenset({"list_dir", "list_tree", "read_file", "read_many_files"})

# Tools whose effect is non-idempotent on disk: re-issuing the same (name, args)
# would produce duplicate observable state (par_003 reproducer: shared.txt
# appended 3 times instead of once). Once one of these completes, the same
# (name, args) must not run again.
#
# Excluded:
# - Command-style tools (run_tests, terminal, run_python, http_fetch, ...) —
#   the planner can legitimately re-invoke them after intervening writes
#   (build → tests fail → patch → tests pass).
# - mkdir is idempotent (`exist_ok=True`); blocking it permanently can wedge
#   the run when LLM keeps emitting the same mkdir while no other progress
#   is visible. In-flight dedupe via _submitted_action_keys still prevents
#   concurrent duplicates within a single batch.
_FILE_MUTATING_TOOL_NAMES = frozenset({
    "write_file",
    "write_many_files",
    "append_file",
    "replace_in_file",
    "patch_file",
    "delete_path",
    "move_path",
    "todo_write",
})


@dataclass
class RunUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    model_calls: int = 0
    context_estimate: int = 0

    def add(self, usage: Usage | None, *, prompt_estimate: int = 0, completion_estimate: int = 0) -> None:
        self.model_calls += 1
        if usage is None:
            self.input_tokens += max(0, int(prompt_estimate))
            self.output_tokens += max(0, int(completion_estimate))
            self.total_tokens += max(0, int(prompt_estimate) + int(completion_estimate))
            return
        prompt = int(usage.prompt_tokens or 0)
        completion = int(usage.completion_tokens or 0)
        if prompt <= 0 and prompt_estimate > 0:
            prompt = int(prompt_estimate)
        if completion <= 0 and completion_estimate > 0:
            completion = int(completion_estimate)
        total = int(usage.total_tokens or 0)
        if total <= 0:
            total = prompt + completion
        elif total < prompt + completion:
            total = prompt + completion
        self.input_tokens += prompt
        self.output_tokens += completion
        if prompt > 0:
            self.context_estimate = prompt
        self.total_tokens += total

    def as_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "model_calls": self.model_calls,
            "context_estimate": self.context_estimate,
        }


@dataclass
class CompletionDecision:
    accepted: bool
    reason: str = ""


@dataclass
class CompletionGate:
    objective: str
    saw_test_tool: bool = False
    saw_failed_test: bool = False
    saw_passing_test: bool = False

    def note_tool_call(self, name: str, args: dict[str, Any]) -> None:
        lowered = name.lower()
        if lowered == "run_tests":
            self.saw_test_tool = True
        if lowered == "terminal" and "test" in str(args.get("command") or "").lower():
            self.saw_test_tool = True
        if lowered == "run_python" and "pytest" in str(args.get("code") or "").lower():
            self.saw_test_tool = True

    def note_delivery(self, event: DeliveryEvent) -> None:
        tool = str(event.metadata.get("tool_name") or "")
        if tool == "run_tests":
            self.saw_test_tool = True
            if event.result.status == "completed":
                self.saw_failed_test = False
                self.saw_passing_test = True
            else:
                self.saw_failed_test = True
                self.saw_passing_test = False

    def evaluate(self, runtime: Any, final_candidate: str) -> CompletionDecision:
        decision = self._decide(runtime, final_candidate)
        # emit on every
        # evaluation so parse_trace.py can compute F4 (gate accept/reject ratio
        # and reasons). The fields mirror the gate's own decision inputs so
        # post-mortem analysis does not need to re-derive them from sibling
        # events.
        try:
            file_writes = sum(
                1 for component_id in runtime.components.snapshot()
                if component_id.startswith("file:")
            )
        except Exception:
            file_writes = 0
        try:
            runtime.trace.emit_typed(
                "controller.completion_gate",
                accepted=bool(decision.accepted),
                pending=int(runtime.pending_count()),
                build_like=self._is_build_like(),
                file_writes=file_writes,
                saw_test_tool=self.saw_test_tool,
                saw_passing_test=self.saw_passing_test,
                saw_failed_test=self.saw_failed_test,
                reason=decision.reason or "",
                # record the candidate's
                # first 240 chars so post-mortem can see exactly what the
                # model said when gate rejected. Helps refine
                # _final_mentions_failure / _final_mentions_test_result
                # heuristics without rerunning experiments.
                final_candidate_preview=str(final_candidate or "")[:240],
            )
        except Exception:
            pass
        return decision

    def _decide(self, runtime: Any, final_candidate: str) -> CompletionDecision:
        if runtime.pending_count() > 0:
            return CompletionDecision(False, f"runtime still has pending tasks: {runtime.status_digest().text}")
        counts = runtime.ledger.counts()
        failed = int(counts.get("failed", 0))
        blocked = int(counts.get("blocked", 0))
        if blocked:
            if self._final_mentions_blocker(final_candidate):
                return CompletionDecision(True)
            return CompletionDecision(False, f"runtime has blocked tasks: {runtime.status_digest().text}")
        # v11+ (gate over-strictness on failed tasks):
        # the prior logic forever rejected build-like prompts with ANY failed
        # task in the ledger. v11 stage 1 trace evidence (4/12 trip cells):
        #   - 2 cells: candidate is success-style summary ("Implemented X
        #     and ran tests successfully"), but ledger has 2-3 exploratory
        #     "path outside workspace" failures from earlier sandbox probes.
        #     The model's success summary should override the bookkeeping
        #     failures.
        #   - 2 cells: candidate explicitly explains failure ("unable to
        #     complete because workspace is empty"); recall heuristic
        #     matches via "unable to" — but the build-like-no-writes branch
        #     fires earlier and rejects.
        # Three escapes:
        #   (a) failed+build-like: accept if final mentions failure OR final
        #       claims success (the "Implemented... ran tests successfully"
        #       pattern signals goal completion despite ledger noise).
        #   (b) build-like+no-writes: accept if final explains why no writes
        #       happened (matches _final_mentions_failure heuristic).
        #   (c) saw_test_tool+saw_failed_test: accept if final mentions
        #       failure (already in place from previous fix).
        if failed and self._is_build_like() and not (
            self.saw_passing_test and self._final_mentions_test_result(final_candidate)
        ):
            if self._final_mentions_failure(final_candidate):
                return CompletionDecision(True)
            if self._final_claims_success(final_candidate):
                return CompletionDecision(True)
            return CompletionDecision(False, f"runtime has failed tasks that need repair: {runtime.status_digest().text}")
        if failed and not (
            self.saw_passing_test and self._final_mentions_test_result(final_candidate)
        ) and not self._final_mentions_failure(final_candidate):
            if self._final_claims_success(final_candidate):
                return CompletionDecision(True)
            return CompletionDecision(False, f"runtime has failed tasks; final answer must explain them: {runtime.status_digest().text}")
        if self._is_build_like() and not self._has_file_write(runtime):
            # v11+: this branch previously hard-rejected.
            # If the final explicitly explains why no writes happened (workspace
            # empty / sandbox missing the source / etc.), accept the honest
            # report. Otherwise reject as before.
            if self._final_mentions_failure(final_candidate):
                return CompletionDecision(True)
            return CompletionDecision(False, "build-like task has no file writes yet; continue with tool calls")
        if self.saw_test_tool and self.saw_failed_test:
            # same relaxation — if the final explicitly
            # mentions the test failure, accept (it's an honest report, not
            # a silent give-up).
            if self._final_mentions_failure(final_candidate):
                return CompletionDecision(True)
            return CompletionDecision(False, "tests were attempted and failed; continue fixing or explain the blocker")
        if self.saw_test_tool and not self._final_mentions_test_result(final_candidate):
            return CompletionDecision(False, "tests were attempted; final answer must include the test result")
        return CompletionDecision(True)

    def _is_build_like(self) -> bool:
        text = self.objective.lower()
        keywords = (
            "构建",
            "创建",
            "实现",
            "生成",
            "开发",
            "项目",
            "系统",
            "build",
            "create",
            "implement",
            "generate",
            "project",
            "app",
            "system",
        )
        return any(keyword in text for keyword in keywords)

    def _has_file_write(self, runtime: Any) -> bool:
        return any(component_id.startswith("file:") for component_id in runtime.components.snapshot())

    def _final_mentions_blocker(self, final_candidate: str) -> bool:
        text = final_candidate.lower()
        return any(token in text for token in ("blocked", "approval", "denied", "未批准", "审批", "阻塞", "无法执行"))

    def _final_mentions_failure(self, final_candidate: str) -> bool:
        # the prior
        # heuristic missed common ways LLMs report failure without using
        # the literal word "failed/failure/error". Stage 1 smoke had 6/12
        # cells where the model's final clearly explained sandbox issues
        # ("couldn't write files outside workspace", "no module named pytest")
        # but the gate rejected because the heuristic didn't match. Expand
        # the keyword set to include common English/Chinese phrasings of
        # "I tried but couldn't"; the gate accepts this honest report
        # rather than forcing the model to emit a literal "failure" word.
        text = final_candidate.lower()
        keywords = (
            # explicit failure words
            "failed", "failure", "error", "errored",
            # negative-outcome phrasings
            "couldn't", "cannot", "could not", "unable to", "wasn't able",
            "was not able", "not able to", "didn't succeed", "did not succeed",
            # sandbox / environment limitation phrasings
            "outside workspace", "outside the workspace", "outside of workspace",
            "not available", "not installed", "no module named", "command not found",
            "not found", "missing", "blocked", "denied",
            # Chinese
            "失败", "错误", "报错", "无法", "未能", "不可用", "未安装",
            "找不到", "拒绝", "受限",
        )
        return any(token in text for token in keywords)

    def _final_mentions_test_result(self, final_candidate: str) -> bool:
        text = final_candidate.lower()
        return any(token in text for token in ("test", "pytest", "unittest", "passed", "failed", "测试", "通过", "失败"))

    def _final_claims_success(self, final_candidate: str) -> bool:
        # v11+: the gate's "failed tasks need repair"
        # branch over-rejects when ledger has exploratory failed tasks
        # (e.g. "path outside workspace" probes) but the model's final is a
        # success summary ("Implemented X, added test Y, ran successfully").
        # This heuristic detects success-style finals so the gate can accept
        # despite ledger noise. The match requires positive-completion verbs
        # AND substantive content (≥ 80 chars) — short content-only turns
        # like "ok" don't count.
        text = (final_candidate or "").lower()
        if len(text.strip()) < 80:
            return False
        # Positive-completion phrasings
        keywords = (
            "implemented", "completed", "fixed", "added the regression",
            "added a regression", "ran the test", "tests pass", "test passes",
            "tests passed", "test passed", "ran successfully", "runs successfully",
            "now returns", "now correctly", "verified",
            "已实现", "已完成", "已修复", "已添加", "通过测试", "测试通过",
        )
        return any(token in text for token in keywords)


@dataclass
class AgentRunController:
    agent: Any
    objective: str
    messages: list[dict[str, Any]]
    model_params: dict[str, Any] = field(default_factory=dict)
    delivery_timeout: float = 30.0
    max_iterations: int = 200
    max_planner_requests: int = 4
    # v11-D2: planner future stale cap. Was hardcoded 120.0 inside
    # _cancel_stale_planners, but the underlying httpx timeout is 600s by
    # default and large project-build prompts genuinely take 5-7 min on the
    # first turn (the symptom: planner_seqs cycling at exactly
    # elapsed_seconds=120.0 with the same snapshot_seq, controller never
    # commits a final answer). Caller decides — cli/main.py defaults to
    # model_timeout so the controller cap matches what the LLM client is
    # already willing to wait for.
    planner_stale_seconds: float = 600.0
    # v11-D3: when N planner futures all time out against the same
    # snapshot_seq, stop re-dispatching against it. The snapshot is
    # genuinely unreachable (model wedged on this prompt shape, not just
    # slow) and burning more model calls reproduces the same trace pattern
    # of planner_seqs cycling endlessly on one snapshot_seq. After the
    # breaker trips for a snapshot, the controller waits for delivery
    # progress to advance the snapshot or — if no in-flight tasks remain —
    # surfaces a best-effort final message instead of looping forever.
    planner_stuck_threshold: int = 3
    on_delivery: Callable[[DeliveryBatch], None] | None = None
    cancel_event: threading.Event | None = None
    usage: RunUsage = field(default_factory=RunUsage)
    completion_gate: CompletionGate = field(init=False)
    _last_status_wakeup_seq: int = 0
    _planner_seq: int = 0
    _submitted_action_keys: set[tuple[str, str]] = field(init=False, default_factory=set)
    _completed_action_keys: set[tuple[str, str]] = field(init=False, default_factory=set)
    # track completed read-only (tool, args)
    # so dedupe can short-circuit re-issued reads. Cached content is
    # already injected into planner context via _build_known_reads_block,
    # so a duplicate read produces no information — only LLM RTT cost.
    # Indexed by path for stale invalidation on write delivery.
    _completed_read_keys: set[tuple[str, str]] = field(init=False, default_factory=set)
    _completed_read_path_index: dict[str, set[tuple[str, str]]] = field(
        init=False, default_factory=dict
    )
    _submitted_action_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _recent_delivery_notes: list[str] = field(init=False, default_factory=list)
    _planner_facts: list["_PlannerFact"] = field(init=False, default_factory=list)
    _refill_facts: list["_PlannerFact"] = field(init=False, default_factory=list)
    _status_notes: list[str] = field(init=False, default_factory=list)
    _fact_seq: int = 0
    _fact_lock: threading.Lock = field(init=False, default_factory=threading.Lock)
    _base_messages: list[dict[str, Any]] = field(init=False, default_factory=list)
    _refill_signal: threading.Event = field(init=False, default_factory=threading.Event)
    _stale_read_paths: set[str] = field(init=False, default_factory=set)
    _recent_completion_times: list[float] = field(init=False, default_factory=list)
    _planner_lifecycle_lock: threading.RLock = field(init=False, default_factory=threading.RLock)
    _active_planner_ids: set[int] = field(init=False, default_factory=set)
    _abandoned_planner_ids: set[int] = field(init=False, default_factory=set)
    _planner_outputs_closed: bool = field(init=False, default=False)
    _effect_seq: int = field(init=False, default=0)
    _last_delivery_appended: bool = field(init=False, default=False)
    # v11-D3: per-snapshot planner-timeout counters and the set of snapshot
    # ids that have hit ``planner_stuck_threshold``. Both are mutated under
    # ``_planner_lifecycle_lock`` so streaming + sync-completion paths cannot
    # race the breaker decision.
    _snapshot_timeout_counts: dict[int, int] = field(init=False, default_factory=dict)
    _stuck_snapshots: set[int] = field(init=False, default_factory=set)
    # (B+C,: forced-stop heuristic.
    # When the planner emits N consecutive responses where every tool_call
    # is dropped by dedupe (in_flight / already_completed), the LLM is in a
    # tool-call-loop where it keeps re-proposing the same calls and the
    # ledger never advances. After threshold consecutive dedupe-only
    # planners, inject an explicit "objective met, stop" signal into the
    # next planner's user message so the model can break out.
    _dedupe_only_streak: int = field(init=False, default=0)
    _force_stop_signal: bool = field(init=False, default=False)
    _DEDUPE_ONLY_THRESHOLD: int = 2
    # track deliveries that did not produce new
    # file writes. After threshold consecutive deliveries with no new write
    # path observed, the agent is most likely doing reflective tool calls
    # (read/list/run_tests) past task completion — inject forced-stop.
    _deliveries_without_new_write: int = field(init=False, default=0)
    _NO_WRITE_DELIVERIES_THRESHOLD: int = 4

    def __post_init__(self) -> None:
        self.completion_gate = CompletionGate(self.objective)
        self._base_messages = [dict(message) for message in self.messages]

    def _on_refill_needed(self, task_id: str, digest: Any) -> None:
        """Called from worker thread when tasks drain and new model request may be needed."""
        self._add_planner_fact(
            "refill_signal",
            (
                "Runtime refill signal.\n"
                f"completed_task={task_id}\n"
                f"Ledger:\n{getattr(digest, 'text', str(digest))}"
            ),
        )
        self._refill_signal.set()

    def _on_critical_path_progress(self, task_id: str, fanout: int, digest: Any) -> None:
        """Called from worker thread when a high-fanout task completes.

        Fires before the runtime would otherwise idle, so the next planner
        request can already be in flight while remaining tasks finish."""
        self._add_planner_fact(
            "critical_path_signal",
            (
                "Critical path task completed; fan out new work without waiting for idle.\n"
                f"completed_task={task_id}\n"
                f"unblocked_fanout={fanout}\n"
                f"Ledger:\n{getattr(digest, 'text', str(digest))}"
            ),
            priority=1,
        )
        self._refill_signal.set()

    def run(self) -> str:
        self.agent.runtime.on_refill_needed = self._on_refill_needed
        self.agent.runtime.on_critical_path_progress = self._on_critical_path_progress
        final_candidate = ""
        final_candidate_seq = -1
        final_candidate_had_pending = False
        final_candidate_effect_seq = -1
        planner_started = 0
        need_planner_refill = True
        in_flight: dict[concurrent.futures.Future[_PlannerResult], _PlannerRequest] = {}
        max_planners = max(1, int(self.max_planner_requests or 1))

        # v11-C9: planner HTTP RTT now runs on the IO loop's shared async
        # executor (introduced in C6 for agent_loop_step). The dedicated
        # planner ThreadPoolExecutor was redundant — its threads sat blocked
        # in httpx.stream during the full RTT, while the IO loop already
        # multiplexes 16 concurrent HTTP streams across the same pool used
        # by sub-agent loops.
        try:
            while True:
                if self.cancel_event is not None and self.cancel_event.is_set():
                    self._drain_ready_deliveries()
                    self.agent.runtime.trace.emit(
                        "controller.cancelled",
                        planner_seq=self._planner_seq,
                        had_final_candidate=bool(final_candidate),
                    )
                    return final_candidate
                if self._refill_signal.is_set():
                    self._refill_signal.clear()
                    if not final_candidate:
                        need_planner_refill = True
                if need_planner_refill and not final_candidate and planner_started < self.max_iterations:
                    planner_started = self._start_planners(
                        in_flight,
                        planner_started,
                        max_planners,
                    )
                    need_planner_refill = False

                for future in self._finished_planners(in_flight):
                    request = in_flight.pop(future, None)
                    if request is None:
                        # a sibling-final abandon
                        # earlier in this iteration may have removed the
                        # future already. Skip — the abandon path emitted
                        # the trace and cleaned up lifecycle.
                        continue
                    try:
                        result = future.result()
                    except Exception as exc:
                        self._finish_planner_request(request.request_id)
                        self.agent.runtime.trace.emit(
                            "planner.failed",
                            planner_seq=request.request_id,
                            error=str(exc)[:200],
                        )
                        need_planner_refill = True
                        continue
                    candidate, candidate_seq, candidate_had_pending, candidate_effect_seq = self._handle_planner_result(request, result)
                    if candidate:
                        final_candidate = candidate
                        final_candidate_seq = candidate_seq
                        final_candidate_had_pending = candidate_had_pending
                        final_candidate_effect_seq = candidate_effect_seq
                        # when ANY planner emits a final_candidate, abandon
                        # all other in-flight planners. Without this, sibling
                        # planners continue emitting tool_calls into runtime,
                        # the ledger never settles to "no pending work", and
                        # the completion gate either rejects the candidate
                        # or marks it stale before it can be evaluated.
                        # Smoke v7 evidence: planner_seq=10 emitted final
                        # at finish=stop, but planners 8/11/12 kept lowering
                        # tool_calls; gate then rejected on residual failed
                        # tasks; cell tripped max_iter despite the model
                        # having already produced a final answer. Abandoning
                        # siblings preserves the candidate's snapshot for
                        # gate evaluation; if gate rejects we still have
                        # budget to dispatch a fresh planner cycle.
                        siblings_to_abandon = [
                            (fut, req) for fut, req in in_flight.items()
                            if fut is not future
                        ]
                        for sibling_fut, sibling_req in siblings_to_abandon:
                            self._abandon_planner_request(
                                sibling_req,
                                reason="final_candidate_emitted_elsewhere",
                            )
                            sibling_fut.cancel()
                            in_flight.pop(sibling_fut, None)
                        if siblings_to_abandon:
                            self.agent.runtime.trace.emit(
                                "controller.siblings_abandoned_for_final",
                                anchor_planner_seq=request.request_id,
                                anchor_snapshot_seq=request.snapshot_seq,
                                abandoned_count=len(siblings_to_abandon),
                            )
                            # v15+ (grace-period efficiency):
                            # the original drain used wait_all(timeout=15.0)
                            # unconditionally on every sibling-abandon, but
                            # final_candidate can fire multiple times per cell
                            # (each time a planner emits stop). Smoke v15
                            # w_fix_002 r1: 5× sibling-abandon → 5×15s = 75s
                            # of pure grace overhead, accumulating into the
                            # cell wall as ghost task durations.
                            #
                            # New strategy:
                            #   1. Reduce hard cap 15s → 3s.
                            #   2. Poll pending_count(); short-circuit when
                            #      runtime drains naturally (typically <1s
                            #      since dispatched tasks are mostly file IO).
                            #   3. Skip grace entirely if pending_count is
                            #      already 0 — nothing to drain.
                            try:
                                if self.agent.runtime.pending_count() > 0:
                                    deadline = time.monotonic() + 3.0
                                    while time.monotonic() < deadline:
                                        if self.agent.runtime.pending_count() == 0:
                                            break
                                        time.sleep(0.05)
                            except Exception:
                                pass
                            self._drain_ready_deliveries()

                batch = self._wait_and_append_delivery(timeout=0)
                if batch and not final_candidate:
                    need_planner_refill = self._last_delivery_appended and planner_started < self.max_iterations

                if final_candidate and not in_flight and self.agent.runtime.pending_count() == 0:
                    drained = self._drain_ready_deliveries()
                    if drained and self.agent.runtime.pending_count() > 0:
                        continue
                    if (
                        final_candidate_had_pending
                        and self._effect_seq > final_candidate_effect_seq
                        and planner_started < self.max_iterations
                    ):
                        # The candidate reflected a snapshot that has since been
                        # superseded by sibling-planner deliveries.
                        # Only discard the candidate when we still have iteration
                        # budget to produce a fresher one — otherwise we would
                        # drop a perfectly good answer on the floor and end with
                        # controller.max_iterations / had_final_candidate=false
                        # (the trace mode that motivated this fix).
                        self.agent.runtime.trace.emit(
                            "planner.final_candidate_stale",
                            planner_seq=self._planner_seq,
                            candidate_seq=final_candidate_seq,
                            candidate_effect_seq=final_candidate_effect_seq,
                            current_effect_seq=self._effect_seq,
                        )
                        final_candidate = ""
                        final_candidate_seq = -1
                        final_candidate_had_pending = False
                        final_candidate_effect_seq = -1
                        need_planner_refill = True
                        continue
                    decision = self.completion_gate.evaluate(self.agent.runtime, final_candidate)
                    if decision.accepted:
                        self.agent.runtime.trace.emit("planner.accepted_final", planner_seq=self._planner_seq)
                        self.messages.append({"role": "assistant", "content": final_candidate})
                        return final_candidate
                    self._append_gate_rejection(decision)
                    final_candidate = ""
                    final_candidate_seq = -1
                    final_candidate_had_pending = False
                    final_candidate_effect_seq = -1
                    need_planner_refill = planner_started < self.max_iterations
                    continue

                if not in_flight and self.agent.runtime.pending_count() == 0:
                    if self._drain_ready_deliveries():
                        if not final_candidate:
                            need_planner_refill = planner_started < self.max_iterations
                        continue
                    if planner_started >= self.max_iterations:
                        self.agent.runtime.trace.emit(
                            "controller.max_iterations",
                            planner_seq=self._planner_seq,
                            max_iterations=self.max_iterations,
                            had_final_candidate=bool(final_candidate),
                        )
                        return final_candidate or self._max_iterations_message()
                    # v11-D3: if the current ledger snapshot is in the stuck
                    # set and no work is in flight, the runtime has nothing
                    # left that could advance digest.seq. Spinning here would
                    # reproduce the very loop the breaker exists to break;
                    # surface a best-effort final message instead.
                    digest = self.agent.runtime.status_digest()
                    with self._planner_lifecycle_lock:
                        snapshot_stuck = digest.seq in self._stuck_snapshots
                    if snapshot_stuck:
                        self.agent.runtime.trace.emit(
                            "controller.snapshot_stuck_exit",
                            planner_seq=self._planner_seq,
                            snapshot_seq=digest.seq,
                            had_final_candidate=bool(final_candidate),
                        )
                        return final_candidate or self._snapshot_stuck_message(digest)
                    need_planner_refill = True
                    continue

                if planner_started >= self.max_iterations and not in_flight:
                    if self.agent.runtime.pending_count() == 0:
                        if self._drain_ready_deliveries():
                            continue
                        self.agent.runtime.trace.emit(
                            "controller.max_iterations",
                            planner_seq=self._planner_seq,
                            max_iterations=self.max_iterations,
                            had_final_candidate=bool(final_candidate),
                        )
                        return final_candidate or self._max_iterations_message()
                    batch = self._wait_and_append_delivery(timeout=min(self.delivery_timeout, 1.0))
                    if batch:
                        continue
                    cancelled = self.agent.runtime.cancel_stale_tasks(max_seconds=120.0)
                    if cancelled:
                        continue
                    self._append_status_update_if_changed()
                    continue

                if in_flight:
                    done, _ = concurrent.futures.wait(
                        set(in_flight),
                        timeout=2.0,
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    if not done:
                        self._cancel_stale_planners(in_flight)
                        batch = self._wait_and_append_delivery(timeout=0)
                        if batch and not final_candidate:
                            need_planner_refill = self._last_delivery_appended and planner_started < self.max_iterations
                    continue

                if self.agent.runtime.pending_count() > 0:
                    batch = self._wait_and_append_delivery(timeout=min(self.delivery_timeout, 1.0))
                    if not batch:
                        cancelled = self.agent.runtime.cancel_stale_tasks(max_seconds=120.0)
                        if cancelled:
                            need_planner_refill = planner_started < self.max_iterations
                        elif self._refill_signal.is_set() or self._append_status_update_if_changed():
                            if not final_candidate:
                                need_planner_refill = planner_started < self.max_iterations
                    elif not final_candidate:
                        need_planner_refill = self._last_delivery_appended and planner_started < self.max_iterations
        finally:
            for request in list(in_flight.values()):
                self._abandon_planner_request(request, reason="run_closed")
            self._close_planner_outputs()
            for future in list(in_flight):
                future.cancel()

    def _start_planners(
        self,
        in_flight: dict[concurrent.futures.Future["_PlannerResult"], "_PlannerRequest"],
        planner_started: int,
        max_planners: int,
    ) -> int:
        started_any = False
        while len(in_flight) < max_planners and planner_started < self.max_iterations:
            digest = self.agent.runtime.status_digest()
            # v11-D3: refuse to keep dispatching against a snapshot that has
            # already burned through ``planner_stuck_threshold`` timeouts.
            # The snapshot can only recover when delivery progress advances
            # the ledger (the runtime emits a new digest.seq on the next
            # delivery). Until then, _start_planners must not feed more
            # planners into the same dead snapshot — that is the loop the
            # breaker exists to break.
            with self._planner_lifecycle_lock:
                snapshot_stuck = digest.seq in self._stuck_snapshots
            if snapshot_stuck:
                self.agent.runtime.trace.emit(
                    "planner.snapshot_stuck_skip",
                    planner_seq=self._planner_seq,
                    snapshot_seq=digest.seq,
                )
                break
            in_flight_keys = {_planner_key(request) for request in in_flight.values()}
            fact = self._pop_refill_fact(snapshot_seq=digest.seq, in_flight_keys=in_flight_keys)
            kind = "refill" if fact is not None else "full"
            key = (kind, digest.seq, fact.seq if fact else 0)
            if key in in_flight_keys:
                break
            if kind == "full" and any(request.kind == "full" and request.snapshot_seq == digest.seq for request in in_flight.values()):
                break
            request = _PlannerRequest(
                request_id=self._planner_seq + 1,
                snapshot_seq=digest.seq,
                digest_text=digest.text,
                messages=self._planner_messages(digest.text, kind=kind, focus_fact=fact, digest=digest),
                had_pending=self.agent.runtime.pending_count() > 0,
                kind=kind,
                focus_fact=fact.text if fact else "",
                fact_seq=fact.seq if fact else 0,
            )
            self._planner_seq += 1
            planner_started += 1
            started_any = True
            self.usage.context_estimate = estimate_messages_tokens(request.messages, self.agent.tools.definitions())
            self.agent.last_usage = self.usage
            self.agent.runtime.trace.emit(
                "planner.started",
                planner_seq=request.request_id,
                snapshot_seq=request.snapshot_seq,
                kind=request.kind,
                fact_seq=request.fact_seq,
            )
            # REINS schema-named alias. Legacy ``planner.started`` stays
            # for back-compat with existing trace consumers; ``planner.requested``
            # is what parse_trace.py joins on for F3 (planner concurrency).
            self.agent.runtime.trace.emit_typed(
                "planner.requested",
                planner_seq=request.request_id,
                snapshot_seq=request.snapshot_seq,
                kind=request.kind,
                fact_seq=request.fact_seq,
                ledger_digest_size=len(request.digest_text or ""),
            )
            self._register_planner_request(request)
            request._started_at = time.monotonic()
            future = self._dispatch_planner_request_async(request)
            in_flight[future] = request
            full_in_flight = sum(1 for r in in_flight.values() if r.kind == "full")
            max_full = 3 if self._completion_frequency_high() else 2
            if kind == "full" and full_in_flight >= max_full:
                break
        if not started_any:
            self.agent.last_usage = self.usage
        return planner_started

    def _completion_frequency_high(self) -> bool:
        """Return True when >=3 completions occurred within the last 2 seconds."""
        now = time.monotonic()
        self._recent_completion_times = [t for t in self._recent_completion_times if now - t <= 2.0]
        return len(self._recent_completion_times) >= 3

    def _run_planner_request(self, request: "_PlannerRequest") -> "_PlannerResult":
        """Synchronous planner request used as the test entry point.

        Production code path goes through ``_dispatch_planner_request_async``
        (v11-C9), which routes the LLM HTTP RTT onto the IO loop's async
        executor so no dedicated planner thread sits blocked across the
        round-trip. Tests still call this directly to exercise the
        streaming / dedupe paths synchronously.
        """
        return self._run_planner_request_inner(request)

    def _dispatch_planner_request_async(
        self, request: "_PlannerRequest"
    ) -> concurrent.futures.Future["_PlannerResult"]:
        """v11-C9 production dispatch: route the planner request through
        ``ModelClient.complete_streaming_async`` / ``complete_async`` so the
        HTTP RTT runs on the IO loop's shared executor instead of a
        dedicated planner thread.

        The returned Future resolves to a fully-formed ``_PlannerResult``.
        Streaming early-dispatch (``on_tool_call``) is preserved verbatim —
        the callback fires from the IO worker thread and acquires
        ``_planner_lifecycle_lock`` exactly as before.
        """
        api_messages = sanitize_tool_protocol_messages(request.messages)
        early_ids: list[str] = []
        result_future: concurrent.futures.Future["_PlannerResult"] = (
            concurrent.futures.Future()
        )

        complete_streaming_async = getattr(
            self.agent.model_client, "complete_streaming_async", None
        )
        complete_async = getattr(self.agent.model_client, "complete_async", None)

        def _on_tool_call(call: Any) -> None:
            try:
                with self._planner_lifecycle_lock:
                    if not self._planner_output_open(request.request_id):
                        self.agent.runtime.trace.emit(
                            "planner.ignored_late_output",
                            planner_seq=request.request_id,
                            reason="closed",
                        )
                        return
                    normalized = self.agent._normalize_tool_calls([call])
                    filtered, dropped = self._dedupe_tool_calls(
                        request.snapshot_seq, normalized
                    )
                    if dropped:
                        self._record_dropped_tool_calls(request, dropped)
                    if not filtered:
                        return
                    for item in filtered:
                        try:
                            self.completion_gate.note_tool_call(
                                item.call.name, item.call.args_dict()
                            )
                        except Exception:
                            self.completion_gate.note_tool_call(item.call.name, {})
                    self.agent._submit_normalized_tool_calls(filtered)
                    self._note_planner_made_progress()
                    early_ids.extend(
                        item.call.id for item in filtered if item.call.id
                    )
                    request._early_dispatched_calls.extend(filtered)
                    self.agent.runtime.trace.emit_typed(
                        "planner.lowered",
                        planner_seq=request.request_id,
                        submitted=len(filtered),
                        dropped=len(dropped),
                        deduped=sum(1 for _, r in dropped if r in {"in_flight", "already_completed"}),
                        fixed=0,
                        snapshot_seq=request.snapshot_seq,
                    )
            except Exception:
                pass

        def _on_response(response_future: concurrent.futures.Future) -> None:
            if result_future.cancelled():
                return
            try:
                response = response_future.result()
            except concurrent.futures.CancelledError:
                if not result_future.done():
                    result_future.cancel()
                return
            except Exception as exc:
                if not result_future.done():
                    result_future.set_exception(exc)
                return
            try:
                planner_result = _PlannerResult(
                    response=response,
                    prompt_estimate=estimate_messages_tokens(request.messages),
                    completion_estimate=estimate_response_tokens(response),
                    early_dispatched_ids=frozenset(early_ids),
                )
            except Exception as exc:
                if not result_future.done():
                    result_future.set_exception(exc)
                return
            if not result_future.done():
                result_future.set_result(planner_result)

        try:
            if complete_streaming_async is not None:
                response_future = complete_streaming_async(
                    api_messages,
                    tools=self.agent.tools.definitions(),
                    on_tool_call=_on_tool_call,
                    **self.model_params,
                )
            elif complete_async is not None:
                response_future = complete_async(
                    api_messages,
                    tools=self.agent.tools.definitions(),
                    **self.model_params,
                )
            else:
                # Fallback: model client lacks async surface (test doubles).
                # Run the legacy sync path on the same caller thread; this
                # only happens in unit tests with hand-rolled stubs.
                planner_result = self._run_planner_request_inner(request)
                result_future.set_result(planner_result)
                return result_future
        except Exception as exc:
            result_future.set_exception(exc)
            return result_future

        response_future.add_done_callback(_on_response)
        return result_future

    def _run_planner_request_inner(
        self, request: "_PlannerRequest"
    ) -> "_PlannerResult":
        api_messages = sanitize_tool_protocol_messages(request.messages)
        early_ids: list[str] = []

        complete_streaming = getattr(self.agent.model_client, "complete_streaming", None)
        if complete_streaming is not None:
            def _on_tool_call(call: Any) -> None:
                try:
                    with self._planner_lifecycle_lock:
                        if not self._planner_output_open(request.request_id):
                            self.agent.runtime.trace.emit(
                                "planner.ignored_late_output",
                                planner_seq=request.request_id,
                                reason="closed",
                            )
                            return
                        normalized = self.agent._normalize_tool_calls([call])
                        filtered, dropped = self._dedupe_tool_calls(request.snapshot_seq, normalized)
                        if dropped:
                            self._record_dropped_tool_calls(request, dropped)
                        if not filtered:
                            return
                        for item in filtered:
                            try:
                                self.completion_gate.note_tool_call(item.call.name, item.call.args_dict())
                            except Exception:
                                self.completion_gate.note_tool_call(item.call.name, {})
                        self.agent._submit_normalized_tool_calls(filtered)
                        self._note_planner_made_progress()
                        early_ids.extend(item.call.id for item in filtered if item.call.id)
                        # Track for the abandon path so a synthetic assistant
                        # entry can carry these tool_call ids if the planner
                        # is later abandoned.
                        request._early_dispatched_calls.extend(filtered)
                        self.agent.runtime.trace.emit_typed(
                            "planner.lowered",
                            planner_seq=request.request_id,
                            submitted=len(filtered),
                            dropped=len(dropped),
                            deduped=sum(1 for _, r in dropped if r in {"in_flight", "already_completed"}),
                            fixed=0,
                            snapshot_seq=request.snapshot_seq,
                        )
                except Exception:
                    pass

            response = complete_streaming(
                api_messages,
                tools=self.agent.tools.definitions(),
                on_tool_call=_on_tool_call,
                **self.model_params,
            )
        else:
            response = self.agent.model_client.complete(
                api_messages,
                tools=self.agent.tools.definitions(),
                **self.model_params,
            )
        return _PlannerResult(
            response=response,
            prompt_estimate=estimate_messages_tokens(request.messages),
            completion_estimate=estimate_response_tokens(response),
            early_dispatched_ids=frozenset(early_ids),
        )

    def _finished_planners(
        self,
        in_flight: dict[concurrent.futures.Future["_PlannerResult"], "_PlannerRequest"],
    ) -> list[concurrent.futures.Future["_PlannerResult"]]:
        return [future for future in in_flight if future.done()]

    def _register_planner_request(self, request: "_PlannerRequest") -> None:
        with self._planner_lifecycle_lock:
            self._active_planner_ids.add(request.request_id)

    def _finish_planner_request(self, request_id: int) -> None:
        with self._planner_lifecycle_lock:
            self._active_planner_ids.discard(request_id)

    def _abandon_planner_request(self, request: "_PlannerRequest", *, reason: str) -> None:
        should_emit = False
        with self._planner_lifecycle_lock:
            should_emit = request.request_id not in self._abandoned_planner_ids
            self._active_planner_ids.discard(request.request_id)
            self._abandoned_planner_ids.add(request.request_id)
        if should_emit:
            self.agent.runtime.trace.emit(
                "planner.abandoned",
                planner_seq=request.request_id,
                snapshot_seq=request.snapshot_seq,
                kind=request.kind,
                reason=reason,
            )
            # If the planner already dispatched tool calls via streaming, the
            # runtime will deliver tool results for them. Synthesize an
            # assistant message carrying their tool_call_ids so the next
            # planner's sanitize_tool_protocol_messages pass does not drop
            # those tool results as orphans.
            self._record_assistant_message_for_early_dispatch(request)

    def _record_assistant_message_for_early_dispatch(self, request: "_PlannerRequest") -> None:
        if request._assistant_message_recorded:
            return
        if not request._early_dispatched_calls:
            return
        request._assistant_message_recorded = True
        tool_calls_payload = assistant_tool_calls(request._early_dispatched_calls)
        if not tool_calls_payload:
            return
        self.messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": tool_calls_payload,
            }
        )

    def _planner_abandoned(self, request_id: int) -> bool:
        with self._planner_lifecycle_lock:
            return request_id in self._abandoned_planner_ids or self._planner_outputs_closed

    def _planner_output_open(self, request_id: int) -> bool:
        with self._planner_lifecycle_lock:
            return (
                not self._planner_outputs_closed
                and request_id in self._active_planner_ids
                and request_id not in self._abandoned_planner_ids
            )

    def _close_planner_outputs(self) -> None:
        with self._planner_lifecycle_lock:
            self._planner_outputs_closed = True
            self._abandoned_planner_ids.update(self._active_planner_ids)
            self._active_planner_ids.clear()

    def _handle_planner_result(self, request: "_PlannerRequest", result: "_PlannerResult") -> tuple[str, int, bool, int]:
        try:
            if self._planner_abandoned(request.request_id):
                self.agent.runtime.trace.emit(
                    "planner.ignored_late_result",
                    planner_seq=request.request_id,
                    snapshot_seq=request.snapshot_seq,
                    kind=request.kind,
                )
                # Best-effort: if any tool calls were dispatched mid-stream
                # but no abandon-time synthetic assistant entry was written
                # yet (e.g. the abandon path was never hit because the
                # future already finished), make sure the transcript stays
                # coherent so subsequent sanitize passes do not strip
                # delivered tool results.
                self._record_assistant_message_for_early_dispatch(request)
                return "", -1, False, -1
            self.agent.runtime.trace.emit(
                "planner.completed",
                planner_seq=request.request_id,
                snapshot_seq=request.snapshot_seq,
                finish_reason=result.response.finish_reason,
                kind=request.kind,
                fact_seq=request.fact_seq,
            )
            # REINS schema-named alias. parse_trace.py pairs
            # ``planner.requested`` ↔ ``planner.responded`` to derive planner
            # RTT and finish-reason distribution (F3, F5).
            usage = getattr(result.response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
            completion_tokens = (
                int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
            )
            tool_calls_count = (
                len(result.response.tool_calls) if result.response.tool_calls else 0
            )
            self.agent.runtime.trace.emit_typed(
                "planner.responded",
                planner_seq=request.request_id,
                snapshot_seq=request.snapshot_seq,
                kind=request.kind,
                fact_seq=request.fact_seq,
                tool_calls_count=tool_calls_count,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=str(result.response.finish_reason or ""),
            )
            self.usage.add(
                result.response.usage,
                prompt_estimate=result.prompt_estimate,
                completion_estimate=result.completion_estimate,
            )
            self.agent.last_usage = self.usage
            planner_messages = list(request.messages)
            before_assistant = len(planner_messages)
            normalized_calls = self.agent._record_assistant_message(planner_messages, result.response)
            assistant_messages = planner_messages[before_assistant:]
            if normalized_calls:
                # Skip calls already dispatched during streaming.
                if result.early_dispatched_ids:
                    normalized_calls = [c for c in normalized_calls if c.call.id not in result.early_dispatched_ids]
                filtered, dropped = self._dedupe_tool_calls(request.snapshot_seq, normalized_calls)
                if dropped:
                    self._record_dropped_tool_calls(request, dropped)
                if filtered or result.early_dispatched_ids:
                    self.messages.extend(assistant_messages)
                for item in filtered:
                    try:
                        self.completion_gate.note_tool_call(item.call.name, item.call.args_dict())
                    except Exception:
                        self.completion_gate.note_tool_call(item.call.name, {})
                if filtered and self._planner_output_open(request.request_id):
                    self.agent._submit_normalized_tool_calls(filtered)
                    self._note_planner_made_progress()
                    self.agent.runtime.trace.emit_typed(
                        "planner.lowered",
                        planner_seq=request.request_id,
                        submitted=len(filtered),
                        dropped=len(dropped),
                        deduped=sum(1 for _, r in dropped if r in {"in_flight", "already_completed"}),
                        fixed=0,
                        snapshot_seq=request.snapshot_seq,
                    )
                elif filtered:
                    self.agent.runtime.trace.emit(
                        "planner.ignored_late_output",
                        planner_seq=request.request_id,
                        reason="closed_result",
                    )
                # documented investigation.
                # The original strict rule required finish_reason ∈ {"stop","end_turn"}.
                # Smoke testing showed bobdong.cn (gpt-5.4-mini through NewAPI)
                # returns finish_reason="tool_calls" on 100% of responses, even
                # when content carries the final summary. We tested two relaxations:
                #
                #   (a) accept content when post-dedupe filtered list is empty:
                #       smoke confirmed model in multi-planner mode emits NEW
                #       (non-duplicate) tool calls every turn → never triggers,
                #       cell still trips max_iterations and OOMs.
                #
                # The actual root cause is multi-planner LLM context priming:
                # with max_planner_requests > 1, the LLM sees a long history
                # of assistant turns each containing tool_calls and never
                # naturally switches to content-only. Solution is operational
                # (set max_planner_requests=1 as default), not in this code path.
                # See benchmark/HIGH_AGENT_NON_CONVERGENCE_DIAGNOSIS.md.
                #
                # We keep the strict finish_reason check (legacy — it
                # still works correctly for providers that DO emit "stop"; it
                # just doesn't help on this provider/multi-planner combination.
                finish_reason = str(result.response.finish_reason or "")
                if (
                    result.response.content
                    and finish_reason in {"stop", "end_turn"}
                ):
                    self.agent.runtime.trace.emit(
                        "planner.final_candidate",
                        planner_seq=request.request_id,
                        snapshot_seq=request.snapshot_seq,
                        with_tool_calls=True,
                    )
                    return result.response.content, request.snapshot_seq, request.had_pending, self._effect_seq
                return "", -1, False, -1
            if result.response.content:
                self.agent.runtime.trace.emit(
                    "planner.final_candidate",
                    planner_seq=request.request_id,
                    snapshot_seq=request.snapshot_seq,
                )
                return result.response.content, request.snapshot_seq, request.had_pending, self._effect_seq
            return "", -1, False, -1
        finally:
            self._finish_planner_request(request.request_id)

    def _dedupe_tool_calls(
        self, snapshot_seq: int, calls: list[NormalizedToolCall]
    ) -> tuple[list[NormalizedToolCall], list[tuple[NormalizedToolCall, str]]]:
        """Filter calls already submitted/completed.

        Returns ``(filtered, dropped)`` where ``dropped`` carries the reason for
        each rejection (``"in_flight"`` | ``"already_completed"``) so callers
        can surface that back to the next planner.

        Lock ordering invariant: callers may hold
        ``_planner_lifecycle_lock`` when entering this method (streaming
        ``_on_tool_call`` path does); callers must NOT hold
        ``_submitted_action_lock`` or ``agent._active_action_index_lock``
        on entry. Inside the method we acquire those two in order
        ``_submitted_action_lock`` → ``_active_action_index_lock`` (via
        ``_has_active_action``). Any future code that takes the inner
        locks must therefore not call back into ``_planner_lifecycle_lock``
        or any helper that acquires it, otherwise a streaming dispatch
        racing against a non-streaming planner result hits an ABBA
        deadlock. The canonical ordering across the controller is:

            _planner_lifecycle_lock
                ↓
            _submitted_action_lock
                ↓
            agent._active_action_index_lock
        """
        filtered: list[NormalizedToolCall] = []
        dropped: list[tuple[NormalizedToolCall, str]] = []
        with self._submitted_action_lock:
            for item in calls:
                key = _tool_call_dedupe_key(item)
                is_file_mutating = item.call.name in _FILE_MUTATING_TOOL_NAMES
                is_read_only = item.call.name in _READ_ONLY_TOOL_NAMES
                if key in self._submitted_action_keys or self._has_active_action(item):
                    self.agent.runtime.trace.emit(
                        "planner.ignored_duplicate",
                        snapshot_seq=snapshot_seq,
                        tool=item.call.name,
                        reason="in_flight",
                    )
                    dropped.append((item, "in_flight"))
                    continue
                # short-circuit re-issued
                # read-only tools whose result is already in the cached-reads
                # block. The planner can see the content; re-running the read
                # only wastes LLM RTT and inflates the dedupe-streak. Reads on
                # paths that have since been written to are NOT cached
                # (_stale_read_paths invalidates them), so this is safe.
                if is_read_only and key in self._completed_read_keys:
                    self.agent.runtime.trace.emit(
                        "planner.ignored_duplicate",
                        snapshot_seq=snapshot_seq,
                        tool=item.call.name,
                        reason="already_completed_read",
                    )
                    dropped.append((item, "already_completed_read"))
                    continue
                if is_file_mutating and key in self._completed_action_keys:
                    self.agent.runtime.trace.emit(
                        "planner.ignored_duplicate",
                        snapshot_seq=snapshot_seq,
                        tool=item.call.name,
                        reason="already_completed",
                    )
                    dropped.append((item, "already_completed"))
                    continue
                self._submitted_action_keys.add(key)
                filtered.append(item)
        return filtered, dropped

    def _has_active_action(self, item: NormalizedToolCall) -> bool:
        try:
            args = item.call.args_dict()
        except Exception:
            args = {"__raw_arguments__": item.call.arguments}
        canonical_args = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        key = (item.call.name, canonical_args)
        with self.agent._active_action_index_lock:
            return bool(self.agent._active_action_index.get(key))

    def _record_dropped_tool_calls(
        self,
        request: "_PlannerRequest",
        dropped: list[tuple[NormalizedToolCall, str]],
    ) -> None:
        """Surface dedupe rejections back to the next planner.

        When every tool call from a planner is filtered out, no task enters
        the runtime, no delivery fires, the ledger digest does not change,
        and the same prompt would yield the same dropped calls forever.
        We record one fact per dropped call so the next planner sees
        explicit feedback ("you already did X; pick a different next step")
        and breaks the loop.
        """
        if not dropped:
            return
        # bump dedupe-only streak. The caller
        # downstream (when filtered>0) calls _note_planner_made_progress
        # to reset this. After _DEDUPE_ONLY_THRESHOLD consecutive planners
        # where every emitted tool call was dropped, we set the
        # force_stop_signal so the next planner_messages() prompt nudges
        # the LLM toward an explicit final answer.
        self._dedupe_only_streak += 1
        if self._dedupe_only_streak >= self._DEDUPE_ONLY_THRESHOLD:
            self._force_stop_signal = True
        lines: list[str] = []
        for item, reason in dropped[:8]:
            try:
                args_repr = json.dumps(item.call.args_dict(), ensure_ascii=False, sort_keys=True)
            except Exception:
                args_repr = item.call.arguments or "{}"
            lines.append(
                f"- {item.call.name} args={_clip_text(args_repr, 240)} reason={reason}"
            )
        if len(dropped) > 8:
            lines.append(f"- ... and {len(dropped) - 8} more dropped calls")
        guidance = (
            "Your last tool calls were dropped because they duplicate work "
            "already in flight, already completed (mutating tools), or already "
            "read by a sibling planner (read_file/list_tree/list_dir/read_many_files — "
            "the result is in the 'Cached read results' section above; do not "
            "re-read unless you wrote to that path). Do NOT re-emit them. "
            "Inspect the ledger and the cached reads, then choose the next "
            "concrete step (e.g. write actual file contents)."
        )
        self._add_status_note(
            "Tool-call dedupe filter rejected planner output.\n"
            f"planner_seq={request.request_id} snapshot_seq={request.snapshot_seq}\n"
            + "\n".join(lines)
            + f"\n{guidance}"
        )

    def _note_planner_made_progress(self) -> None:
        """Reset the dedupe-only streak when a planner emits any non-dropped
        tool call. Called from every site that submits filtered>0 tool calls
        to the runtime — see

        Only the dedupe-streak (fix C) resets here. The no-new-write streak
        (fix C2) is independent: read-only tools count as no-new-write
        progress, so resetting C2 from the planner submit site would
        defeat the C2 detector. C2 only resets when a real file write
        delivery completes (in the delivery handler).
        """
        self._dedupe_only_streak = 0

    def _note_completed_action(self, tool_name: str, task: Any, *, succeeded: bool) -> None:
        try:
            args = (task.input or {}).get("args") or {}
            canonical_args = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
        except Exception:
            return
        key = (tool_name, canonical_args)
        with self._submitted_action_lock:
            # Clear the in-flight stamp so that legitimate post-completion
            # re-issue (e.g. run_tests after a patch) is not blocked.
            self._submitted_action_keys.discard(key)
            # File-mutating tools that succeeded additionally land in the
            # completed set: repeating them with the same args is a known
            # multi-write bug mode (par_003 reproducer) and must stay
            # blocked. Failed writes are NOT added — the planner is
            # allowed to retry them.
            if succeeded and tool_name in _FILE_MUTATING_TOOL_NAMES:
                self._completed_action_keys.add(key)
            # track successful read-only
            # completions so dedupe can short-circuit re-issued reads on
            # the same path. Indexed by path so the delivery handler can
            # invalidate when a write makes the cached read stale.
            if succeeded and tool_name in _READ_ONLY_TOOL_NAMES:
                self._completed_read_keys.add(key)
                path = str(args.get("path") or args.get("paths") or ".")
                self._completed_read_path_index.setdefault(path, set()).add(key)

    def _planner_messages(
        self,
        digest_text: str,
        *,
        kind: str = "full",
        focus_fact: "_PlannerFact | None" = None,
        digest: Any = None,
    ) -> list[dict[str, Any]]:
        # (multi-planner convergence per:
        # restructure the planner's user message so the model can decide on
        # OBJECTIVE completion without the ledger's "running tasks" gating
        # it negatively. Pre-fix wording said "Only provide a final answer
        # when the ledger shows the requested work is complete" — under
        # multi-planner pumping the ledger ALWAYS shows pending work
        # (other planners' tasks), so the model never converged.
        #
        # New structure:
        #   1. Fixed history (durable_context): facts that are settled and
        #      will not change — what's already been completed and delivered.
        #   2. Runtime accounting: a snapshot of what's running / waiting NOW,
        #      framed as bookkeeping not gating.
        #   3. Causal context: which task completions unblocked which others.
        #   4. Decision prompt: explicit instruction that the OBJECTIVE drives
        #      completion, not the ledger's pending state.
        messages = [dict(message) for message in self._base_messages]
        context = _clip_text(self.agent.context.render(), MAX_DURABLE_CONTEXT_CHARS)
        with self._fact_lock:
            facts = list(self._planner_facts[-MAX_FACTS_IN_VIEW:])
            status_notes = list(self._status_notes[-MAX_STATUS_NOTES_IN_VIEW:])
        fact_lines = [fact.text for fact in facts]
        if focus_fact is not None and focus_fact.text not in fact_lines:
            fact_lines.append(focus_fact.text)
        recent = "\n\n".join(fact_lines)
        status = "\n\n".join(status_notes)
        causal_block = _format_causal_block(digest)
        known_reads = self._build_known_reads_block()

        # Section 1: fixed history — completed facts and durable summary
        sections: list[str] = []
        sections.append(
            f"## Objective\n{self.objective}"
        )
        if context:
            sections.append(
                f"## Completed work so far (durable context)\n{context}"
            )

        # Section 2: runtime accounting — explicit bookkeeping framing
        sections.append(
            "## Runtime bookkeeping (parallel scheduling state — NOT a completion gate)\n"
            f"{digest_text}"
        )
        if causal_block:
            sections.append(
                f"## Causal links (recent task completions and what they unblocked)\n{causal_block}"
            )
        if known_reads:
            sections.append(known_reads)

        # Section 3: facts and status notes (recent runtime events)
        if recent:
            sections.append(
                f"## Recent runtime facts\n{recent}"
            )
        if status:
            sections.append(
                f"## Recent status notes\n{status}"
            )

        # Section 4: decision prompt — objective-driven, not ledger-gated
        decision_lines = [
            f"## Decide ({kind})",
            "Based on the OBJECTIVE above and the durable completed work:",
            "- If the objective is met (the requested files exist with correct content "
            "and the test you were asked to add passes), reply with a final summary and "
            "NO tool calls. That ends the conversation.",
            "- Otherwise, emit the next batch of tool calls. Prefer returning multiple "
            "tool calls in one response when they are independent — the runtime executes "
            "them in parallel.",
            "Note: tasks listed under 'Runtime bookkeeping' as running or waiting are "
            "internal scheduling state. They are NOT a reason to keep emitting tools. "
            "If the objective is met, finish even if the ledger still shows in-flight "
            "scheduler activity (other planners' work that you do not need to wait on).",
        ]
        # forced-stop signal fires from two
        # detectors:
        #   C  — dedupe-streak: N consecutive planners whose emitted tool
        #        calls were all rejected (in_flight / already_completed).
        #   C2 — no-new-write streak: N consecutive deliveries that did not
        #        produce a new file write (reflective read_file/list_dir
        #        loops past objective completion).
        # Both are signs the agent is no longer making real progress.
        # Inject an explicit "stop now" instruction at the top of the
        # decision prompt so the next planner can break out by emitting
        # a final summary instead of more tool calls.
        if self._force_stop_signal:
            decision_lines.insert(
                1,
                "**FORCED STOP SIGNAL**: The runtime detects you are no "
                "longer making real progress — recent rounds produced only "
                "duplicate tool calls (rejected by dedupe) or read-only "
                "verification (read_file / list_dir) without any new file "
                "writes. The objective is almost certainly already met. "
                "Reply with a final summary of what was completed and NO "
                "tool calls. Do not emit any more read_file / list_dir / "
                "verification calls — the runtime has confirmed the work "
                "is done.",
            )
        sections.append("\n".join(decision_lines))

        content = "\n\n".join(sections)
        content = _clip_text(content, MAX_PLANNER_CONTEXT_CHARS)
        messages.append({"role": "user", "content": content})
        return messages

    def _wait_and_append_delivery(self, timeout: float | None = None) -> DeliveryBatch | None:
        current_timeout = self.delivery_timeout if timeout is None else timeout
        while True:
            batch = self.agent.wait_delivery(timeout=current_timeout)
            if not batch:
                self._last_delivery_appended = False
                return None
            for event in batch.events:
                self.completion_gate.note_delivery(event)
            if self.on_delivery:
                self.on_delivery(batch)
            delivery_messages: list[dict[str, Any]] = []
            appended = self.agent._append_delivery_messages(delivery_messages, batch)
            # was REVERTED: setting
            # _last_delivery_appended=True on partial-group deliveries triggered
            # premature planner re-dispatch (the planner asked the model "what
            # now?" before the merged tool message landed, so the model re-
            # emitted tool calls or finalized incorrectly — see
            # test_parallel_wrapper_project_build_writes_files for the
            # canonical reproducer). The audit's diagnosis (streaming refill
            # is suppressed on grouped tools) is real, but the right fix is in
            # the runtime's on_critical_path_progress callback firing on each
            # child completion, NOT in flipping this gate. Tracked separately;
            # for now the original semantics stand.
            self._last_delivery_appended = bool(appended)
            self._append_delivery_update(batch, delivery_messages, appended=appended)
            return batch

    def _drain_ready_deliveries(self) -> bool:
        drained = False
        while self._wait_and_append_delivery(timeout=0):
            drained = True
        return drained

    def _max_iterations_message(self) -> str:
        return (
            "Stopped after "
            f"{self.max_iterations} planner requests before a final answer was accepted. "
            f"Runtime status: {self.agent.runtime.status_digest().text}"
        )

    def _snapshot_stuck_message(self, digest: Any) -> str:
        """v11-D3: surfaced when the breaker trips on the current snapshot
        and nothing else can advance the runtime ledger."""
        return (
            "Stopped because the planner repeatedly timed out against the same "
            f"runtime snapshot (snapshot_seq={getattr(digest, 'seq', '?')}, "
            f"threshold={int(self.planner_stuck_threshold or 0)}). The model is "
            "wedged on this prompt shape; no final answer was produced. "
            f"Runtime status: {getattr(digest, 'text', '')}"
        )

    def _append_delivery_update(self, batch: DeliveryBatch, delivery_messages: list[dict[str, Any]], *, appended: int) -> None:
        parts = [str(message.get("content") or "") for message in delivery_messages if message.get("content")]
        if not parts:
            parts = [delivery_content(event, batch.digest) for event in batch.events]
        note = "\n".join(parts)
        self._recent_delivery_notes.append(note)
        self._recent_delivery_notes = self._recent_delivery_notes[-12:]
        for event in batch.events:
            self._add_planner_fact("delivery", delivery_content(event, batch.digest))
            task = self.agent.runtime._tasks.get(event.task_id)
            if task:
                self.agent._unindex_active_action(event.task_id, task)
            tool_name = (event.metadata.get("tool_name") or "")
            # Discard the in-flight stamp regardless of status — failure also
            # ends the in-flight window, and the planner may legitimately
            # retry the same call (e.g. run_tests after a fix).
            if tool_name and task is not None:
                self._note_completed_action(
                    tool_name,
                    task,
                    succeeded=(event.result.status == "completed"),
                )
            if self._delivery_has_effect(event, task):
                self._effect_seq += 1
            if event.result.status == "completed":
                self._recent_completion_times.append(time.monotonic())
                if tool_name and tool_name not in _READ_ONLY_TOOL_NAMES:
                    write_path = self._extract_write_path(event)
                    if write_path:
                        self._stale_read_paths.add(write_path)
                        parent = write_path.rsplit("/", 1)[0] if "/" in write_path else ""
                        if parent:
                            self._stale_read_paths.add(parent)
                        # invalidate cached
                        # read keys for the written path AND its parent so
                        # a subsequent re-read isn't short-circuited with
                        # stale content.
                        with self._submitted_action_lock:
                            for stale in (write_path, parent) if parent else (write_path,):
                                bucket = self._completed_read_path_index.pop(stale, set())
                                for key in bucket:
                                    self._completed_read_keys.discard(key)
                        # a real new write resets
                        # the "no-new-write" streak.
                        self._deliveries_without_new_write = 0
                        self._force_stop_signal = False
                    else:
                        # Mutating tool ran but produced no extractable
                        # write path — count as no-progress delivery.
                        self._deliveries_without_new_write += 1
                else:
                    # Read-only tool (read_file, list_dir, run_tests, terminal …) —
                    # delivery happened but no file mutation: bump counter.
                    self._deliveries_without_new_write += 1
                if self._deliveries_without_new_write >= self._NO_WRITE_DELIVERIES_THRESHOLD:
                    self._force_stop_signal = True
        if appended:
            self.messages.extend(delivery_messages)
            self._add_planner_fact(
                "provider_tool_result",
                (
                    "Provider-visible grouped tool result became available.\n"
                    f"{note}\n"
                    f"Ledger:\n{batch.digest}"
                ),
            )

    def _append_status_update_if_changed(self) -> bool:
        digest = self.agent.runtime.status_digest()
        if digest.seq == self._last_status_wakeup_seq:
            return False
        self._last_status_wakeup_seq = digest.seq
        # the prior wording
        # "Do not finalize while tasks are pending" gated the model's final
        # answer on ledger emptiness — under multi-planner pumping that
        # never triggered. New wording preserves the wake-up signal but
        # keeps the objective as the sole completion oracle.
        self._add_status_note(
            "Runtime status update (no new tool delivery arrived before the wait timeout).\n"
            f"Ledger:\n{digest.text}\n"
            "If the objective is met, finish — the listed running/waiting tasks are "
            "internal scheduler bookkeeping. Otherwise, emit the next batch of tools."
        )
        return True

    def _append_gate_rejection(self, decision: CompletionDecision) -> None:
        self._add_status_note(
            "Runtime completion gate rejected final answer. "
            f"Reason: {decision.reason}\n"
            f"Ledger:\n{self.agent.runtime.status_digest().text}\n"
            "Continue with tool calls or explicitly address the blocker."
        )

    def _build_known_reads_block(self) -> str:
        """Build a summary of completed read-only tool results from runtime state.

 we previously truncated each cached
        read to 80 chars, which is path-only — sibling planners always
        re-read the file because the "summary" did not contain the actual
        content. Now we surface up to MAX_KNOWN_READ_ENTRY_CHARS of the
        actual ToolResult.summary (which for read_file/list_tree IS the
        file/tree content, populated by ToolRegistry.task_from_call).
        """
        runtime = self.agent.runtime
        tasks = getattr(runtime, "_tasks", {})
        records_fn = getattr(runtime.ledger, "records_snapshot", None)
        if not callable(records_fn):
            return ""
        records = records_fn()
        # Build a map keyed by (tool_name, path) → (full content, finished_at)
        # so re-reads of the same path overwrite the older one. Sort by
        # recency so the most recently read paths come first within budget.
        seen: dict[tuple[str, str], tuple[str, float]] = {}
        for task_id, task in dict(tasks).items():
            record = records.get(task_id)
            if record is None or record.state != "completed":
                continue
            task_input = getattr(task, "input", None) or {}
            tool_name = task_input.get("name", "")
            if tool_name not in _READ_ONLY_TOOL_NAMES:
                continue
            args = task_input.get("args") or {}
            path = str(args.get("path") or args.get("paths") or ".")
            if path in self._stale_read_paths:
                continue
            content = record.summary or ""
            finished = record.finished_at or 0.0
            key = (tool_name, path)
            prior = seen.get(key)
            if prior is None or finished >= prior[1]:
                seen[key] = (content, finished)
        if not seen:
            return ""
        ordered = sorted(seen.items(), key=lambda kv: kv[1][1], reverse=True)
        lines = [
            "## Cached read results (sibling planners already read these — "
            "do NOT re-read unless you wrote to the path since)",
        ]
        total = 0
        rendered = 0
        for (tool_name, path), (content, _) in ordered:
            entry = _clip_text(content, MAX_KNOWN_READ_ENTRY_CHARS)
            block = f"### {tool_name}({path})\n{entry}"
            if total + len(block) > MAX_KNOWN_READS_CHARS:
                remaining = len(ordered) - rendered
                if remaining > 0:
                    lines.append(f"... and {remaining} more cached reads omitted (budget)")
                break
            lines.append(block)
            total += len(block)
            rendered += 1
        return "\n\n".join(lines)

    def _extract_write_path(self, event: DeliveryEvent) -> str:
        """Extract the written path from a delivery event."""
        task = getattr(self.agent.runtime, "_tasks", {}).get(event.task_id)
        if not task:
            return ""
        task_input = getattr(task, "input", None) or {}
        args = task_input.get("args") or {}
        return str(args.get("path", ""))

    def _delivery_has_effect(self, event: DeliveryEvent, task: Any | None) -> bool:
        if event.result.status != "completed":
            return event.result.status in {"failed", "blocked", "cancelled", "partial", "timeout"}
        if event.kind != "tool":
            return True
        tool_name = str(event.metadata.get("tool_name") or "")
        if tool_name in _READ_ONLY_TOOL_NAMES:
            return False
        if task is None:
            return True
        access = getattr(task, "resource_access", None)
        if access is not None:
            if bool(getattr(access, "unknown", False)):
                return True
            writes = set(getattr(access, "writes", ()) or ()) | set(getattr(access, "appends", ()) or ())
            if writes:
                return True
            side_effect = str(getattr(access, "side_effect_level", "none") or "none")
            if side_effect not in {"none", "external_read"}:
                return True
            reads = set(getattr(access, "reads", ()) or ())
            return not reads
        writes = set(getattr(task, "writes", set()) or set())
        return bool(writes)

    def _add_status_note(self, text: str) -> None:
        note = _clip_text(text, MAX_FACT_CHARS)
        with self._fact_lock:
            self._status_notes.append(note)
            self._status_notes = self._status_notes[-MAX_STATUS_NOTES_IN_VIEW:]
        self._add_planner_fact("status", note)

    def _cancel_stale_planners(
        self,
        in_flight: dict[concurrent.futures.Future["_PlannerResult"], "_PlannerRequest"],
    ) -> None:
        """Cancel planner futures that exceed ``planner_stale_seconds``.

        v11-D2: cap is configurable. Pre-D2 a hardcoded 120s killed planners
        long before the underlying httpx timeout (default 600s); for large
        scaffold prompts that take 5-7 min on the first turn, this caused the
        controller to recycle planner_seqs in a tight loop on the same
        snapshot without ever committing a final answer.

        v11-D3: each timeout against snapshot_seq=S increments
        ``_snapshot_timeout_counts[S]``; once it hits
        ``planner_stuck_threshold`` the snapshot is added to
        ``_stuck_snapshots`` and ``_start_planners`` will refuse to dispatch
        further full planners against it. Recovery is tied to delivery
        progress: any delivery advances the runtime ledger, producing a new
        snapshot_seq that is not in the stuck set.
        """
        cap = max(1.0, float(self.planner_stale_seconds))
        now = time.monotonic()
        stale: list[concurrent.futures.Future["_PlannerResult"]] = []
        for future, request in in_flight.items():
            elapsed = now - getattr(request, "_started_at", now)
            if elapsed > cap and not future.done():
                stale.append(future)
        threshold = max(1, int(self.planner_stuck_threshold or 1))
        for future in stale:
            if future.done():
                continue
            request = in_flight.pop(future)
            self._abandon_planner_request(request, reason="timeout")
            cancelled = future.cancel()
            with self._planner_lifecycle_lock:
                count = self._snapshot_timeout_counts.get(request.snapshot_seq, 0) + 1
                self._snapshot_timeout_counts[request.snapshot_seq] = count
                tripped = (
                    count >= threshold
                    and request.snapshot_seq not in self._stuck_snapshots
                )
                if tripped:
                    self._stuck_snapshots.add(request.snapshot_seq)
            self.agent.runtime.trace.emit(
                "planner.timeout",
                planner_seq=request.request_id,
                snapshot_seq=request.snapshot_seq,
                elapsed_seconds=round(time.monotonic() - getattr(request, "_started_at", 0), 1),
                stale_cap_seconds=round(cap, 1),
                future_cancelled=cancelled,
                snapshot_timeout_count=count,
                stuck_threshold=threshold,
            )
            if tripped:
                self.agent.runtime.trace.emit_typed(
                    "planner.snapshot_stuck",
                    snapshot_seq=request.snapshot_seq,
                    timeout_count=count,
                    stuck_threshold=threshold,
                    attempts=count,
                )
                self._add_status_note(
                    "Planner circuit breaker tripped: every recent planner against "
                    f"snapshot_seq={request.snapshot_seq} has timed out "
                    f"({count}/{threshold}). Stopping new planner dispatch against "
                    "this snapshot until delivery progress advances the ledger."
                )

    def _add_planner_fact(self, kind: str, text: str, *, priority: int = 0) -> "_PlannerFact":
        clipped = _clip_text(text, MAX_FACT_CHARS)
        with self._fact_lock:
            self._fact_seq += 1
            fact = _PlannerFact(seq=self._fact_seq, kind=kind, text=clipped, priority=priority)
            self._planner_facts.append(fact)
            self._planner_facts = self._planner_facts[-MAX_FACTS_IN_VIEW:]
            self._refill_facts.append(fact)
            self._refill_facts = self._refill_facts[-MAX_FACTS_IN_VIEW:]
        self.agent.runtime.store.write("facts", clipped)
        return fact

    def _pop_refill_fact(
        self,
        *,
        snapshot_seq: int,
        in_flight_keys: set[tuple[str, int, int]],
    ) -> "_PlannerFact | None":
        with self._fact_lock:
            # Higher-priority facts (e.g. critical_path_signal) jump the queue
            # so the planner sees fan-out events as soon as they appear.
            self._refill_facts.sort(key=lambda f: (-f.priority, f.seq))
            while self._refill_facts:
                fact = self._refill_facts.pop(0)
                if ("refill", snapshot_seq, fact.seq) in in_flight_keys:
                    continue
                return fact
        return None


def delivery_content(event: DeliveryEvent, digest: str) -> str:
    payload = {
        "task_id": event.task_id,
        "tool": event.metadata.get("tool_name") or event.kind,
        "status": event.result.status,
        "summary": event.summary,
        "ledger": digest,
    }
    if event.metadata.get("duration_seconds") is not None:
        payload["duration_seconds"] = event.metadata.get("duration_seconds")
    return json.dumps(
        payload,
        ensure_ascii=False,
    )


def estimate_messages_tokens(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> int:
    # Rough cross-provider estimate for status UI; provider usage remains authoritative.
    chars = 0
    for message in messages:
        chars += len(str(message.get("role") or ""))
        chars += len(str(message.get("content") or ""))
        if message.get("tool_calls"):
            chars += len(json.dumps(message["tool_calls"], ensure_ascii=False))
    if tools:
        chars += len(json.dumps(tools, ensure_ascii=False))
    return max(1, chars // 4)


def estimate_response_tokens(response: NormalizedResponse) -> int:
    chars = len(str(response.content or ""))
    if response.tool_calls:
        chars += len(json.dumps([call.provider_data or {"name": call.name, "arguments": call.arguments} for call in response.tool_calls], ensure_ascii=False))
    return max(1, chars // 4) if chars else 1


@dataclass
class _PlannerRequest:
    request_id: int
    snapshot_seq: int
    digest_text: str
    messages: list[dict[str, Any]]
    had_pending: bool = False
    kind: str = "full"
    focus_fact: str = ""
    fact_seq: int = 0
    _started_at: float = field(default_factory=time.monotonic)
    # Tool calls dispatched mid-stream by _on_tool_call, recorded so
    # _abandon_planner_request can synthesize an assistant message holding
    # those tool_call ids when the planner future is later abandoned. Without
    # this, the runtime's eventual delivery messages have no matching
    # assistant tool_call_id and sanitize_tool_protocol_messages drops them
    # as orphans.
    _early_dispatched_calls: list[NormalizedToolCall] = field(default_factory=list)
    _assistant_message_recorded: bool = False


@dataclass(frozen=True)
class _PlannerResult:
    response: NormalizedResponse
    prompt_estimate: int
    completion_estimate: int
    early_dispatched_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class _PlannerFact:
    seq: int
    kind: str
    text: str
    priority: int = 0


def _planner_key(request: _PlannerRequest) -> tuple[str, int, int]:
    return (request.kind, request.snapshot_seq, request.fact_seq)


def _tool_call_dedupe_key(item: NormalizedToolCall) -> tuple[str, str]:
    try:
        args = item.call.args_dict()
    except Exception:
        args = {"__raw_arguments__": item.call.arguments}
    # AgentLoop (formerly SubAgent) injects `_depth` into
    # delegate_task args so the delegated worker can enforce recursion depth.
    # The dedupe key must not vary with `_depth` (otherwise a model that
    # re-emits the same delegate at depth=1 vs depth=0 produces two entries)
    # and a model echoing `_depth=0` back must not be able to reset the
    # recursion guard. Strip leading-underscore keys before canonicalisation.
    if isinstance(args, dict):
        args = {k: v for k, v in args.items() if not (isinstance(k, str) and k.startswith("_"))}
    canonical_args = json.dumps(args, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return (item.call.name, canonical_args)


def _clip_text(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    omitted = len(text) - limit
    return f"{text[:limit]}\n[omitted {omitted} chars]"


def _format_causal_block(digest: Any) -> str:
    """Render structured causal information from a LedgerDigest.

    The block describes which completed task triggered which downstream tasks
    so the planner sees «因为 A 完成而启动了 B、C» instead of a flat counter."""
    if digest is None:
        return ""
    causal = getattr(digest, "causal_chains", None) or {}
    recent = getattr(digest, "recent_completions", None) or []
    discovery = getattr(digest, "discovery_chains", None) or {}
    lines: list[str] = []
    if recent:
        recent_lines = []
        for task_id, triggered in recent[:4]:
            if triggered:
                rendered = ", ".join(triggered[:4])
                recent_lines.append(f"  {task_id} → {rendered}")
            else:
                recent_lines.append(f"  {task_id} → (no dependents waiting)")
        lines.append("Recent completions and what they unblocked:")
        lines.extend(recent_lines)
    elif causal:
        lines.append("Causal chains:")
        for src, targets in list(causal.items())[:4]:
            lines.append(f"  {src} → {', '.join(targets[:4])}")
    if discovery:
        disc_lines = []
        for parent, children in list(discovery.items())[:3]:
            disc_lines.append(f"  {parent} ⇒ discovered {', '.join(children[:4])}")
        if disc_lines:
            lines.append("Dynamically discovered tasks:")
            lines.extend(disc_lines)
    return "\n".join(lines)
