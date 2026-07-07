"""Benchmark CLI entry point.

Dispatches to a sub-experiment's own ``main()``:

* ``python -m benchmark microbench --help``   → :mod:`benchmark.microbench.runner`
* ``python -m benchmark reins-bench --help``  → :mod:`benchmark.reins_bench.runner`
"""

from __future__ import annotations

from benchmark.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
