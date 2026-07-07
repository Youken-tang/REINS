"""score — thin CLI wrapper around ``reins_bench.scoring.build_report``.

The unified runner exposes ``runner.py score`` already; this script
preserves an alternate CLI shape so external tooling can call it
without remembering the runner sub-command shape.

Both forms speak the same args; this script is just::

    python -m benchmark.reins_bench.scripts.score --root <results> --system reins
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def main(argv: list[str] | None = None) -> int:
    from benchmark.reins_bench.runner import main as runner_main

    args = list(sys.argv[1:] if argv is None else argv)
    # Inject the `score` sub-command — the wrapper exists so callers can
    # type ``score --root ... --system ...`` without remembering it.
    return runner_main(["score", *args])


if __name__ == "__main__":
    raise SystemExit(main())
