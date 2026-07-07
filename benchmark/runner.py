"""Top-level benchmark dispatcher.

Per the entry point is:

    benchmark/runner.py microbench --shape chain --n 8 --workers 8 --runs 5

This module just routes the first positional argument to a sub-module
``main()``. New experiments …) can register here as
they land.
"""

from __future__ import annotations

import sys
from typing import Callable


def _microbench_main(argv: list[str]) -> int:
    from benchmark.microbench.runner import main

    return main(argv)


def _reins_bench_main(argv: list[str]) -> int:
    from benchmark.reins_bench.runner import main

    return main(argv)


_DISPATCH: dict[str, Callable[[list[str]], int]] = {
    "microbench": _microbench_main,
    "reins-bench": _reins_bench_main,
}


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        print("usage: benchmark/runner.py <experiment> [args...]")
        print(f"experiments: {', '.join(sorted(_DISPATCH))}")
        return 0
    name, *rest = args
    handler = _DISPATCH.get(name)
    if handler is None:
        print(f"unknown experiment: {name!r}", file=sys.stderr)
        return 1
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
