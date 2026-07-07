"""schema — TaskSpec dataclass + YAML loader for REINS-Bench prompts.

 Each prompt lives at
``benchmark/REINS-Bench/prompts/<group>/<id>.yaml`` and follows the schema
documented in ``schema/task_spec.json``. This module is the in-process
mirror: it parses one YAML doc into a typed ``TaskSpec`` plus its
``GroundTruthAccess`` rows, normalising defaults and raising on missing
required fields.

Loading is intentionally tolerant of PyYAML being absent: when the
import fails we fall through to ``json.loads`` so unit tests can ship
JSON-shaped fixtures without dragging the dependency in.

Validation policy:

- ``id``, ``group``, ``prompt`` are required; everything else has a
  reasonable default. Forbidden patterns / min_files default to empty
  lists so a freshly-authored prompt can be added without ground truth
  before the corpus author fills it in.
- ``ground_truth_resource_access`` is **advisory**, not strict — a
  prompt can ship with an empty list and still score (the runner will
  simply skip the false-positive / false-negative conflict accounting).
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


@dataclass(frozen=True)
class GroundTruthAccess:
    """One row of expected ResourceAccess for a single tool call.

    Mirrors the runtime's ``ResourceAccess`` in shape but not in type
    so the bench package never has to import runtime code (REINS-Bench
    is intentionally read-only for the runtime — see package docstring).
    """

    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    appends: tuple[str, ...] = ()
    side_effect_level: str = "local"

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "GroundTruthAccess":
        return GroundTruthAccess(
            tool=str(d.get("tool") or ""),
            args=dict(d.get("args") or {}),
            reads=tuple(str(x) for x in (d.get("reads") or [])),
            writes=tuple(str(x) for x in (d.get("writes") or [])),
            appends=tuple(str(x) for x in (d.get("appends") or [])),
            side_effect_level=str(d.get("side_effect_level") or "local"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "args": dict(self.args),
            "reads": list(self.reads),
            "writes": list(self.writes),
            "appends": list(self.appends),
            "side_effect_level": self.side_effect_level,
        }


@dataclass(frozen=True)
class Expected:
    """Pass-rate gates: file existence, test commands, forbidden patterns."""

    min_files: tuple[str, ...] = ()
    must_pass_tests: tuple[str, ...] = ()
    forbidden_patterns: tuple[str, ...] = ()

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Expected":
        return Expected(
            min_files=tuple(str(x) for x in (d.get("min_files") or [])),
            must_pass_tests=tuple(str(x) for x in (d.get("must_pass_tests") or [])),
            forbidden_patterns=tuple(
                str(x) for x in (d.get("forbidden_patterns") or [])
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "min_files": list(self.min_files),
            "must_pass_tests": list(self.must_pass_tests),
            "forbidden_patterns": list(self.forbidden_patterns),
        }


@dataclass(frozen=True)
class Budget:
    """Hard ceilings for one run; runs that exceed any of these fail
    pass_rate even if min_files are present."""

    max_wall_seconds: float = 900.0
    max_input_tokens: int = 200_000
    max_output_tokens: int = 60_000

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "Budget":
        return Budget(
            max_wall_seconds=float(d.get("max_wall_seconds", 900.0)),
            max_input_tokens=int(d.get("max_input_tokens", 200_000)),
            max_output_tokens=int(d.get("max_output_tokens", 60_000)),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_wall_seconds": self.max_wall_seconds,
            "max_input_tokens": self.max_input_tokens,
            "max_output_tokens": self.max_output_tokens,
        }


@dataclass(frozen=True)
class Notes:
    source: str = "hand-authored"
    source_ref: str = ""
    license: str = "CC-BY-4.0"

    @staticmethod
    def from_dict(d: dict[str, Any] | None) -> "Notes":
        d = d or {}
        return Notes(
            source=str(d.get("source") or "hand-authored"),
            source_ref=str(d.get("source_ref") or ""),
            license=str(d.get("license") or "CC-BY-4.0"),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_ref": self.source_ref,
            "license": self.license,
        }


@dataclass(frozen=True)
class TaskSpec:
    """One REINS-Bench prompt, fully parsed."""

    id: str
    group: str
    prompt: str
    title: str = ""
    version: int = 1
    languages: tuple[str, ...] = ()
    expected: Expected = field(default_factory=Expected)
    budget: Budget = field(default_factory=Budget)
    ground_truth_resource_access: tuple[GroundTruthAccess, ...] = ()
    notes: Notes = field(default_factory=Notes)
    source_path: Path | None = None

    @staticmethod
    def from_mapping(doc: dict[str, Any], *, source: Path | None = None) -> "TaskSpec":
        if not isinstance(doc, dict):
            raise ValueError(f"task spec must be a mapping, got {type(doc).__name__}")
        for required in ("id", "group", "prompt"):
            if not doc.get(required):
                raise ValueError(
                    f"task spec missing required field {required!r}"
                    f" (source={source})"
                )
        return TaskSpec(
            id=str(doc["id"]),
            group=str(doc["group"]),
            prompt=str(doc["prompt"]),
            title=str(doc.get("title") or ""),
            version=int(doc.get("version") or 1),
            languages=tuple(str(x) for x in (doc.get("languages") or [])),
            expected=Expected.from_dict(doc.get("expected") or {}),
            budget=Budget.from_dict(doc.get("budget") or {}),
            ground_truth_resource_access=tuple(
                GroundTruthAccess.from_dict(x)
                for x in (doc.get("ground_truth_resource_access") or [])
            ),
            notes=Notes.from_dict(doc.get("notes")),
            source_path=source,
        )

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id,
            "group": self.group,
            "version": self.version,
            "title": self.title,
            "languages": list(self.languages),
            "prompt": self.prompt,
            "expected": self.expected.as_dict(),
            "budget": self.budget.as_dict(),
            "ground_truth_resource_access": [
                row.as_dict() for row in self.ground_truth_resource_access
            ],
            "notes": self.notes.as_dict(),
        }
        return out

    @property
    def declared_writes(self) -> set[str]:
        """Union of every write in ``ground_truth_resource_access``.

        This is the «authorised write set»: any conflict the runtime
        reports on a path outside this set is a candidate false positive,
        and any path in this set that the runtime never flagged is a
        candidate false negative (subject to the prompt actually writing
        to it during the run — the scoring layer joins on observed events).
        """
        out: set[str] = set()
        for row in self.ground_truth_resource_access:
            out.update(row.writes)
            out.update(row.appends)
        return out


def _load_doc(path: Path) -> Any:
    """Parse one task spec file into a Python mapping.

    Tolerates PyYAML being absent: falls through to ``json.loads`` so
    JSON-shaped fixtures still parse in environments without yaml.
    """
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        return json.loads(text or "{}")
    try:
        import yaml  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return json.loads(text or "{}")
    return yaml.safe_load(text) or {}


def load_task_spec(path: Path) -> TaskSpec:
    """Parse one ``<id>.yaml`` (or ``.json``) into a TaskSpec."""
    doc = _load_doc(path)
    return TaskSpec.from_mapping(doc, source=path)


def discover_task_specs(root: Path) -> list[Path]:
    """Walk ``prompts/<group>/<id>.yaml`` and return their paths sorted.

    The root may be either ``benchmark/REINS-Bench/prompts/`` (groups
    sub-divided) or a single group directory; both layouts compose.
    Files whose basename starts with ``_`` are skipped (reserved for
    docstrings / templates).
    """
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".yaml", ".yml", ".json"}:
            continue
        if p.name.startswith("_"):
            continue
        out.append(p)
    return out


def load_corpus(root: Path) -> list[TaskSpec]:
    """Load every prompt under ``root`` into TaskSpec list."""
    return [load_task_spec(p) for p in discover_task_specs(root)]


def filter_corpus(
    specs: Iterable[TaskSpec],
    *,
    group: str | None = None,
    ids: Iterable[str] | None = None,
    languages: Iterable[str] | None = None,
) -> list[TaskSpec]:
    """Apply a uniform filter to a list of task specs."""
    out = list(specs)
    if group:
        out = [s for s in out if s.group == group]
    if ids:
        wanted = set(ids)
        out = [s for s in out if s.id in wanted]
    if languages:
        wlangs = set(languages)
        out = [s for s in out if any(lang in wlangs for lang in s.languages)]
    return out
