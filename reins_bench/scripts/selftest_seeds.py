"""selftest_seeds — verify each W_fix seed is solvable by its reference fix.

Pipeline per prompt:

    1. Make a tempdir as cell workdir.
    2. Copy seed (excluding `_reference/`) into workdir.
    3. Apply reference fix: copy `_reference/<file>.py` → matching `app/<subpkg>/<file>.py`.
    4. Drop the reference test into the prompt's `min_files` location:
       e.g. `_reference/test_users_pagination.py` → `tests/test_users_pagination.py`.
    5. Run the prompt's `must_pass_tests` command. PASS = seed solvable.

A prompt that fails self-test is flagged unsolvable; the operator
either fixes the seed or drops the prompt from the sweep corpus.

Usage:

    PYTHONPATH=src .venv/bin/python -m benchmark.reins_bench.scripts.selftest_seeds
    PYTHONPATH=src .venv/bin/python -m benchmark.reins_bench.scripts.selftest_seeds w_fix_001 w_fix_002
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from benchmark.reins_bench.schema import load_task_spec  # noqa: E402
from benchmark.reins_bench.seed_loader import seed_workdir  # noqa: E402

_SEEDS_ROOT = _REPO / "benchmark" / "reins_bench" / "seeds"
_PROMPTS_ROOT = _REPO / "benchmark" / "reins_bench" / "prompts"


def _apply_reference(seed_dir: Path, workdir: Path, min_files: list[str]) -> None:
    """Apply the seed's `_reference/` patch onto a workdir.

    `_reference/<name>.py` corresponds to a buggy file `app/<subpkg>/<name>.py`
    (we recover the destination by searching `app/**/<name>.py` in the seed).
    `_reference/test_*.py` → goes to `min_files[0]` (or the matching tests/ path).
    """
    ref = seed_dir / "_reference"
    if not ref.is_dir():
        return
    # First: locate the buggy app/* path for each non-test ref file.
    seed_app_files = {p.name: p for p in (seed_dir / "app").rglob("*.py") if p.name != "__init__.py"}
    test_target = None
    if min_files:
        # min_files[0] is the test path the prompt expects.
        test_target = workdir / min_files[0]
    for ref_file in ref.iterdir():
        if not ref_file.is_file() or ref_file.suffix != ".py":
            continue
        if ref_file.name.startswith("test_"):
            if test_target is None:
                continue
            test_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ref_file, test_target)
            # tests/__init__.py for package-mode discovery
            init = test_target.parent / "__init__.py"
            if not init.exists():
                init.write_text("", encoding="utf-8")
        else:
            # Map ref/<name>.py → app/<subpkg>/<name>.py
            target_in_seed = seed_app_files.get(ref_file.name)
            if target_in_seed is None:
                continue
            rel = target_in_seed.relative_to(seed_dir)
            (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ref_file, workdir / rel)


def _run_pytest(workdir: Path, command: str, timeout: float = 60.0) -> tuple[bool, str]:
    """Run `command` in `workdir` with PYTHONPATH=workdir."""
    import shlex
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{workdir.resolve()}{os.pathsep}{env.get('PYTHONPATH','')}"
    venv_bin = _REPO / ".venv" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}:{env.get('PATH','')}"
    try:
        proc = subprocess.run(
            shlex.split(command),
            cwd=str(workdir),
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError as e:
        return False, f"missing binary: {e}"
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s"
    if proc.returncode == 0:
        return True, ""
    tail = (proc.stderr or proc.stdout or "")[-600:]
    return False, f"exit {proc.returncode}: {tail.strip()}"


def selftest_one(prompt_id: str) -> tuple[bool, str]:
    """Self-test one prompt. Returns (passed, detail)."""
    seed_dir = _SEEDS_ROOT / prompt_id
    if not seed_dir.is_dir():
        return False, "no seed directory"
    yaml_path = _PROMPTS_ROOT / "w_fix" / f"{prompt_id}.yaml"
    if not yaml_path.is_file():
        yaml_path = _PROMPTS_ROOT / "w_scaffold" / f"{prompt_id}.yaml"
    if not yaml_path.is_file():
        return False, f"no prompt yaml at {yaml_path}"
    spec = load_task_spec(yaml_path)
    if not spec.expected.must_pass_tests:
        return False, "prompt has no must_pass_tests"

    with tempfile.TemporaryDirectory(prefix="seedtest_") as td:
        workdir = Path(td) / "workdir"
        workdir.mkdir()
        # Step 1+2: copy seed minus _reference
        seed_workdir(prompt_id, workdir)
        # Step 3+4: apply reference patch
        _apply_reference(seed_dir, workdir, list(spec.expected.min_files))
        # Step 5: run must_pass_tests
        for cmd in spec.expected.must_pass_tests:
            ok, detail = _run_pytest(workdir, cmd, timeout=60.0)
            if not ok:
                return False, f"{cmd!r} failed: {detail}"
    return True, ""


def main(argv: list[str]) -> int:
    targets = argv[1:] if len(argv) > 1 else sorted(
        p.name for p in _SEEDS_ROOT.iterdir()
        if p.is_dir() and p.name.startswith("w_fix_")
    )
    failures: list[tuple[str, str]] = []
    print(f"selftest {len(targets)} prompts")
    for pid in targets:
        ok, detail = selftest_one(pid)
        marker = "ok" if ok else "FAIL"
        print(f"  [{marker}] {pid}{'  ' + detail[:200] if detail else ''}")
        if not ok:
            failures.append((pid, detail))
    print()
    if failures:
        print(f"{len(failures)} / {len(targets)} prompts FAILED self-test:")
        for pid, detail in failures:
            print(f"  - {pid}: {detail[:200]}")
        return 1
    print(f"all {len(targets)} prompts pass self-test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
