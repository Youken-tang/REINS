"""Resource access normalization and conflict rules."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

SideEffectLevel = Literal[
    "none",
    "local",
    "external",
    "external_read",
    "external_write",
    "interactive",
    "unknown",
]


@dataclass(frozen=True)
class ResourceAccess:
    reads: frozenset[str] = field(default_factory=frozenset)
    writes: frozenset[str] = field(default_factory=frozenset)
    appends: frozenset[str] = field(default_factory=frozenset)
    unknown: bool = False
    side_effect_level: SideEffectLevel = "none"

    @staticmethod
    def empty() -> "ResourceAccess":
        return ResourceAccess()

    @staticmethod
    def read(*resources: str) -> "ResourceAccess":
        return ResourceAccess(reads=frozenset(resources))

    @staticmethod
    def write(*resources: str) -> "ResourceAccess":
        return ResourceAccess(writes=frozenset(resources), side_effect_level="local")

    @staticmethod
    def append(*resources: str) -> "ResourceAccess":
        return ResourceAccess(appends=frozenset(resources), side_effect_level="local")

    @staticmethod
    def unknown_workspace() -> "ResourceAccess":
        return ResourceAccess(unknown=True, side_effect_level="unknown")

    @staticmethod
    def external_read(*resources: str) -> "ResourceAccess":
        return ResourceAccess(reads=frozenset(resources), side_effect_level="external_read")

    @staticmethod
    def external_write(*resources: str) -> "ResourceAccess":
        return ResourceAccess(writes=frozenset(resources), side_effect_level="external_write")

    def normalized(self, workspace_root: str | None = None) -> "ResourceAccess":
        return ResourceAccess(
            reads=frozenset(normalize_component(r, workspace_root) for r in self.reads),
            writes=frozenset(normalize_component(r, workspace_root) for r in self.writes),
            appends=frozenset(normalize_component(r, workspace_root) for r in self.appends),
            unknown=self.unknown,
            side_effect_level=self.side_effect_level,
        )

    def all_resources(self) -> set[str]:
        return set(self.reads) | set(self.writes) | set(self.appends)


def normalize_component(component: str, workspace_root: str | None = None) -> str:
    """Normalize file-like component keys into stable file:/ or dir:/ keys."""

    raw = str(component or "").strip()
    if not raw:
        return raw
    if raw.startswith(("file:", "dir:")):
        prefix, path = raw.split(":", 1)
        return f"{prefix}:{_abs_path(path, workspace_root)}"
    if raw.startswith(("artifact:", "memory:", "test:", "failure:", "tool:", "subagent:", "external:", "terminal:")):
        return raw
    if raw == "workspace:*":
        return raw
    if _looks_like_path(raw):
        path = _abs_path(raw, workspace_root)
        if raw.endswith(os.sep) or Path(path).is_dir():
            return f"dir:{path}"
        return f"file:{path}"
    return raw


def access_conflicts(left: ResourceAccess, right: ResourceAccess) -> bool:
    """Return True when two resource declarations cannot run concurrently.

 the previous ``unknown`` short-circuit returned True
    whenever either side had ``unknown=True``, regardless of the other side's
    declared resources. That made ``ResourceAccess.empty()`` (declared by 7+
    hermes read-only tools and the entire ``_AGENT_INTERNAL_HERMES_TOOLS``
    bucket) falsely conflict with ``unknown_workspace()``. Those pairs SHOULD
    be safely concurrent — the empty side touches nothing.

    The new rule: when one side is unknown but the other is empty AND has no
    side-effect, return False. Anything else with unknown still serialises.
    """

    if left.unknown or right.unknown:
        unknown_side = left if left.unknown else right
        other = right if left.unknown else left
        # Interactive tools (clarify) MUST serialise against everything,
        # even ResourceAccess.empty(). Hermes' own _NEVER_PARALLEL_TOOLS
        # gate enforces this; we replicate it.
        if unknown_side.side_effect_level == "interactive":
            return True
        # Empty + no side-effect + not-unknown other side coexists with
        # a plain unknown — read-only in-memory tools, agent-internal trivia.
        if (
            not other.all_resources()
            and other.side_effect_level == "none"
            and not other.unknown
        ):
            return False
        # if the other side is purely
        # read-only-by-declaration (only 'reads', no writes/appends, side
        # effect 'none' or 'external_read'), let it run concurrently with
        # an unknown task whose own side_effect is not 'external_write'.
        # Worst case: the unknown writes something the read later observes —
        # but that is ordering noise, not corruption (no two writers race).
        # Without this exception, every batch with one ill-declared
        # write_file (e.g. write_file(path="")) serialised the whole batch
        # behind the unknown. interactive / external_write unknowns still
        # serialise against everything per the rules above.
        if (
            not other.writes
            and not other.appends
            and other.side_effect_level in ("none", "external_read")
            and not other.unknown
            and unknown_side.side_effect_level != "external_write"
        ):
            return False
        return True
    if left.side_effect_level in {"interactive", "unknown"}:
        return True
    if right.side_effect_level in {"interactive", "unknown"}:
        return True
    if left.side_effect_level == "external" or right.side_effect_level == "external":
        return True
    if left.side_effect_level == "external_write" and right.side_effect_level in {"external_read", "external_write"}:
        return True
    if right.side_effect_level == "external_write" and left.side_effect_level in {"external_read", "external_write"}:
        return True

    left_write = set(left.writes)
    right_write = set(right.writes)
    left_append = set(left.appends)
    right_append = set(right.appends)
    left_read = set(left.reads)
    right_read = set(right.reads)

    for a in left_write:
        for b in right_write | right_read | right_append:
            if resources_overlap(a, b):
                return True
    for a in right_write:
        for b in left_read | left_append:
            if resources_overlap(a, b):
                return True
    for a in left_append:
        for b in right_read:
            if resources_overlap(a, b):
                return True
    for a in right_append:
        for b in left_read:
            if resources_overlap(a, b):
                return True
    return False


def resources_overlap(left: str, right: str) -> bool:
    if left == right:
        return True
    if left == "workspace:*" or right == "workspace:*":
        return True
    l_kind, l_path = _split_path_component(left)
    r_kind, r_path = _split_path_component(right)
    if l_kind and r_kind:
        return _paths_overlap(l_kind, Path(l_path), r_kind, Path(r_path))
    return False


def _split_path_component(component: str) -> tuple[str | None, str]:
    if component.startswith("file:"):
        return "file", component[5:]
    if component.startswith("dir:"):
        return "dir", component[4:]
    return None, component


def _paths_overlap(left_kind: str, left: Path, right_kind: str, right: Path) -> bool:
    """Check whether two file/dir components touch the same workspace region.

    Rules:
    - file vs file: overlap only when the paths are identical. Two distinct
      files cannot mask each other even if one prefix is a subpath of the
      other (e.g. ``file:/a/b`` and ``file:/a/b/c`` are *different* files).
    - dir vs dir: a directory write covers all of its descendants, so two
      directories overlap if either is a prefix of the other.
    - dir vs file (or file vs dir): the file is overlapped by a directory
      whenever the file lives at or below the directory.
    """
    if left_kind == "file" and right_kind == "file":
        return left == right
    if left_kind == "dir" and right_kind == "dir":
        return _is_path_prefix(left, right) or _is_path_prefix(right, left)
    dir_path = left if left_kind == "dir" else right
    file_path = right if left_kind == "dir" else left
    return _is_path_prefix(dir_path, file_path)


def _is_path_prefix(prefix: Path, candidate: Path) -> bool:
    prefix_parts = prefix.parts
    candidate_parts = candidate.parts
    if len(candidate_parts) < len(prefix_parts):
        return False
    return candidate_parts[: len(prefix_parts)] == prefix_parts


def _looks_like_path(raw: str) -> bool:
    return raw.startswith(("/", "./", "../", "~")) or os.sep in raw


def _abs_path(path: str, workspace_root: str | None) -> str:
    expanded = Path(path).expanduser()
    if not expanded.is_absolute():
        base = Path(workspace_root or os.getcwd())
        expanded = base / expanded
    return os.path.abspath(str(expanded))
