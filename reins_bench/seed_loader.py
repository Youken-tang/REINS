"""seed_loader — copy a W_fix prompt's buggy seed into the agent's workdir.

W_fix prompts assume an existing buggy codebase the agent debugs and
patches. The runners (high_agent_runner / live_runner / opencode_runner)
historically created an empty workdir and let the agent scaffold from
scratch; that mismatched the "find the bug, fix it" wording in every
prompt body and is the reason v1's W_fix pass_rate ranged 0–17%
across all systems and models — the agent was guessing the project
layout the oracle expected.

Usage from a runner:

    from benchmark.reins_bench.seed_loader import seed_workdir
    seed_workdir(prompt_id="w_fix_001", workdir=Path(".../workdir"))

For W_scaffold prompts (no seed), this is a no-op.
"""

from __future__ import annotations

import shutil
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SEEDS_ROOT = _REPO / "benchmark" / "reins_bench" / "seeds"


def has_seed(prompt_id: str) -> bool:
    seed_dir = _SEEDS_ROOT / prompt_id
    if not seed_dir.is_dir():
        return False
    for child in seed_dir.iterdir():
        if child.name == "_reference":
            continue
        return True
    return False


def seed_workdir(prompt_id: str, workdir: Path) -> bool:
    """Copy the seed for `prompt_id` into `workdir` if a seed exists.

    Returns True if a seed was copied, False otherwise (no seed dir →
    workdir stays untouched). The `_reference/` subdirectory inside the
    seed is **excluded** — that holds the reference solution used by
    `selftest_seeds.py`, not part of the buggy starting state.
    """
    seed_dir = _SEEDS_ROOT / prompt_id
    if not seed_dir.is_dir():
        return False
    workdir.mkdir(parents=True, exist_ok=True)
    copied = False
    for child in seed_dir.iterdir():
        if child.name == "_reference":
            continue
        target = workdir / child.name
        if child.is_dir():
            shutil.copytree(child, target, dirs_exist_ok=True)
        else:
            shutil.copy2(child, target)
        copied = True
    return copied
