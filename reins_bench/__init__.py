"""REINS-Bench — public evaluation set for agent schedulers.

A reusable artifact that lets reviewers compare schedulers on the same
prompts under the same ground-truth resource-access labels, instead of
every project running its own private workload.

Layout::

    benchmark/reins_bench/                ← Python package (PEP-8 friendly)
      __init__.py
      schema.py          ← TaskSpec dataclass + YAML loader
      scoring.py         ← pass_rate + scheduler_score
      runner.py          ← CLI dispatcher
      prompts/<group>/<id>.yaml
      scripts/score.py   ← CLI wrapper around scoring.py
      schema/task_spec.json   (json-schema, hand-authored)

The on-disk package is named ``reins_bench`` (snake_case so Python can
import it); the public handle stays **REINS-Bench**.

The package is **read-only** for production Reins code — the runtime
itself does not import from here. Only benchmark drivers and the
scoring CLI touch this tree.
"""

from __future__ import annotations

from .schema import (
    Budget,
    Expected,
    GroundTruthAccess,
    Notes,
    TaskSpec,
    discover_task_specs,
    filter_corpus,
    load_corpus,
    load_task_spec,
)
from .scoring import (
    CellRecord,
    ConflictAudit,
    GateResult,
    PassResult,
    audit_conflicts,
    build_report,
    cell_record_from_artifacts,
    discover_cells,
    evaluate_pass,
    load_records_from_root,
)

__all__ = [
    "Budget",
    "CellRecord",
    "ConflictAudit",
    "Expected",
    "GateResult",
    "GroundTruthAccess",
    "Notes",
    "PassResult",
    "TaskSpec",
    "audit_conflicts",
    "build_report",
    "cell_record_from_artifacts",
    "discover_cells",
    "discover_task_specs",
    "evaluate_pass",
    "filter_corpus",
    "load_corpus",
    "load_records_from_root",
    "load_task_spec",
]
