"""Compile REINS-Bench yaml prompts into the corpus.json schema.

Reads every prompt under benchmark/reins_bench/prompts/{w_fix,w_scaffold}/
and emits a single corpus.json compatible with the///
loader (CorpusEntry: id / category / language / title / prompt /
expected_min_tools / notes).

The yaml schema (per uses ``group`` and ``languages`` (list);
corpus.json's CorpusEntry uses ``category`` and ``language`` (single).
We map by:

  category = group  (w_fix -> W_fix; w_scaffold -> W_scaffold)
  language = languages[0]  (corpus only carries one)
  expected_min_tools = max(5, len(ground_truth_resource_access))
  notes = empty string (yaml's notes is repo-level metadata, not commentary)

Usage:
    PYTHONPATH=src .venv/bin/python -m benchmark.reins_bench.compile_corpus \\
        --out benchmark/reins_bench/prompts/corpus_50.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))


def _load_yaml_minimal(path: Path) -> dict[str, Any]:
    """Reuse the sweep_runner's hand-rolled YAML parser.

    The / prompt files are flat dicts with a small set of
    scalar / multiline-block / nested-list keys. We don't need pyyaml.
    """
    text = path.read_text(encoding="utf-8")
    fields: dict[str, Any] = {}
    current_key: str | None = None
    multiline_buf: list[str] = []
    multiline_indent: int | None = None
    list_key: str | None = None
    list_buf: list[Any] = []
    nested_block: list[str] | None = None

    lines = text.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        stripped = raw.strip()
        if current_key is not None:
            if not raw.strip():
                multiline_buf.append("")
                i += 1
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            if multiline_indent is None:
                if indent == 0:
                    fields[current_key] = "\n".join(multiline_buf).rstrip()
                    current_key = None
                    multiline_buf = []
                    multiline_indent = None
                    continue
                multiline_indent = indent
                multiline_buf.append(raw[multiline_indent:])
                i += 1
                continue
            if indent < multiline_indent:
                fields[current_key] = "\n".join(multiline_buf).rstrip()
                current_key = None
                multiline_buf = []
                multiline_indent = None
                continue
            multiline_buf.append(raw[multiline_indent:])
            i += 1
            continue
        if list_key is not None:
            if raw.startswith("  - ") or raw.startswith("    - "):
                list_buf.append(stripped[2:].strip())
                i += 1
                continue
            if not raw.strip():
                i += 1
                continue
            indent = len(raw) - len(raw.lstrip(" "))
            if indent == 0:
                fields[list_key] = list_buf
                list_key = None
                list_buf = []
                continue
            i += 1
            continue
        if not raw or raw.startswith("#"):
            i += 1
            continue
        if raw.startswith(" "):
            i += 1
            continue
        if ":" not in raw:
            i += 1
            continue
        key, _, value = raw.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "|":
            current_key = key
            multiline_buf = []
            multiline_indent = None
        elif value == "":
            list_key = key
            list_buf = []
        elif value:
            fields[key] = value
        i += 1

    if current_key is not None:
        fields[current_key] = "\n".join(multiline_buf).rstrip()
    if list_key is not None:
        fields[list_key] = list_buf
    return fields


def yaml_to_corpus_entry(path: Path) -> dict[str, Any]:
    f = _load_yaml_minimal(path)
    group = (f.get("group") or path.parent.name).strip()
    if group == "w_fix":
        category = "W_fix"
    elif group == "w_scaffold":
        category = "W_scaffold"
    else:
        category = group
    langs = f.get("languages") or []
    if isinstance(langs, list) and langs:
        language = langs[0]
    else:
        language = "python"
    return {
        "id": f.get("id", path.stem),
        "category": category,
        "language": language,
        "title": f.get("title", path.stem),
        "prompt": (f.get("prompt") or "").strip(),
        "expected_min_tools": 5,
        "notes": "",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark.reins_bench.compile_corpus")
    ap.add_argument("--prompts-dir", type=Path,
                    default=_REPO / "benchmark/reins_bench/prompts")
    ap.add_argument("--out", type=Path,
                    default=_REPO / "benchmark/reins_bench/prompts/corpus_50.json")
    ns = ap.parse_args(argv)

    items: list[dict[str, Any]] = []
    for group in ("w_fix", "w_scaffold"):
        gdir = ns.prompts_dir / group
        if not gdir.is_dir():
            continue
        for yaml_path in sorted(gdir.glob("*.yaml")):
            entry = yaml_to_corpus_entry(yaml_path)
            items.append(entry)
            print(f"  loaded {entry['id']:<14} ({entry['category']:<10} "
                  f"{entry['language']:<10}) {entry['title'][:50]}")

    corpus = {
        "_doc": ("REINS-Bench corpus compiled from yaml prompts. "
                 "W_fix 30 + W_scaffold 20 = 50 entries. Compatible with "
                 "/// CorpusEntry loader."),
        "_schema": {
            "id": "stable id (e.g. w_fix_001, w_scaffold_007)",
            "category": "W_fix | W_scaffold",
            "language": "python | typescript | go | rust | java",
            "title": "human-readable label",
            "prompt": "objective sent to the planner",
            "expected_min_tools": "lower bound on tool_calls (advisory)",
            "notes": "optional commentary",
        },
        "items": items,
    }
    ns.out.write_text(json.dumps(corpus, indent=2, ensure_ascii=False),
                      encoding="utf-8")
    print(f"\nwrote {len(items)} entries to {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
