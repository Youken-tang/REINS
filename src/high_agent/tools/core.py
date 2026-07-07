"""Core v1 tools."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any

from high_agent.approval import ApprovalManager, ApprovalRequest
from high_agent.runtime.resource_access import ResourceAccess, normalize_component
from high_agent.runtime.types import TaskResult
from high_agent.tools.registry import ToolRegistry
from high_agent.tools.result_store import ToolResultStore


def create_core_registry(
    *,
    allow_terminal: bool = False,
    allow_outside_workspace: bool = False,
    approval_manager: ApprovalManager | None = None,
    result_store: ToolResultStore | None = None,
) -> ToolRegistry:
    registry = ToolRegistry(result_store=result_store)

    registry.register(
        name="noop",
        schema={"description": "Return a simple acknowledgement.", "parameters": {"type": "object", "properties": {}}},
        handler=lambda args: {"ok": True},
        resource_access=lambda args, root: ResourceAccess.empty(),
    )

    registry.register(
        name="sleep",
        schema={
            "description": "Sleep for a number of seconds.",
            "parameters": {"type": "object", "properties": {"seconds": {"type": "number"}}},
        },
        handler=lambda args: _sleep(float(args.get("seconds", 0.01))),
        resource_access=lambda args, root: ResourceAccess.empty(),
    )

    registry.register(
        name="read_file",
        schema={
            "description": "Read a UTF-8 text file.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
        handler=lambda args: _read_file(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess.read(_file_component(args["path"], root)),
    )

    registry.register(
        name="list_dir",
        schema={
            "description": "List a directory.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
        handler=lambda args: _list_dir(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess.read(normalize_component(f"dir:{args['path']}", root)),
    )

    registry.register(
        name="list_tree",
        schema={
            "description": "Return a compact tree of files and directories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_depth": {"type": "integer"},
                    "max_entries": {"type": "integer"},
                },
            },
        },
        handler=lambda args: _list_tree(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess.read(normalize_component(f"dir:{args.get('path') or '.'}", root)),
        max_result_size_chars=50_000,
    )

    registry.register(
        name="mkdir",
        schema={
            "description": "Create a directory, including parents.",
            "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
        },
        handler=lambda args: _mkdir(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess.write(normalize_component(f"dir:{args['path']}", root)),
    )

    registry.register(
        name="write_file",
        schema={
            "description": "Write a UTF-8 text file, creating parents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
        handler=lambda args: _write_file(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess.write(_file_component(args["path"], root)),
    )

    registry.register(
        name="read_many_files",
        schema={
            "description": "Read multiple UTF-8 text files.",
            "parameters": {
                "type": "object",
                "properties": {"paths": {"type": "array", "items": {"type": "string"}}},
                "required": ["paths"],
            },
        },
        handler=lambda args: _read_many_files(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess(
            reads=frozenset(_file_component(str(path), root) for path in args.get("paths", []))
        ),
        max_result_size_chars=100_000,
    )

    registry.register(
        name="write_many_files",
        schema={
            "description": "Write multiple UTF-8 text files, creating parents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                            "required": ["path", "content"],
                        },
                    }
                },
                "required": ["files"],
            },
        },
        handler=lambda args: _write_many_files(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess(
            writes=frozenset(_file_component(str(item.get("path")), root) for item in args.get("files", []) if isinstance(item, dict)),
            side_effect_level="local",
        ),
        max_result_size_chars=80_000,
    )

    registry.register(
        name="append_file",
        schema={
            "description": "Append UTF-8 text to a file, creating parents.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
        handler=lambda args: _append_file(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess.append(_file_component(args["path"], root)),
    )

    registry.register(
        name="replace_in_file",
        schema={
            "description": "Replace text in a UTF-8 file. Fails unless old text occurs exactly once by default.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["path", "old", "new"],
            },
        },
        handler=lambda args: _replace_in_file(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess.write(_file_component(args["path"], root)),
    )

    registry.register(
        name="patch_file",
        schema={
            "description": "Apply multi-replacement edits or a simple unified diff to a UTF-8 file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "diff": {"type": "string"},
                    "replacements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"old": {"type": "string"}, "new": {"type": "string"}},
                            "required": ["old", "new"],
                        },
                    },
                },
            },
        },
        handler=lambda args: _patch_file(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess.write(_file_component(args.get("path") or _diff_target_path(str(args.get("diff") or "")) or ".", root)),
    )

    registry.register(
        name="delete_path",
        schema={
            "description": "Delete a file or directory inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "recursive": {"type": "boolean"}},
                "required": ["path"],
            },
        },
        handler=lambda args: _delete_path(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess.write(_file_component(args["path"], root)),
    )

    registry.register(
        name="move_path",
        schema={
            "description": "Move or rename a file/directory inside the workspace.",
            "parameters": {
                "type": "object",
                "properties": {"src": {"type": "string"}, "dst": {"type": "string"}},
                "required": ["src", "dst"],
            },
        },
        handler=lambda args: _move_path(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess(
            reads=frozenset({_file_component(args["src"], root)}),
            writes=frozenset({_file_component(args["src"], root), _file_component(args["dst"], root)}),
            side_effect_level="local",
        ),
    )

    registry.register(
        name="search_files",
        schema={
            "description": "Search workspace files for a literal or regex pattern. Uses rg when available.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "regex": {"type": "boolean"},
                    "max_results": {"type": "integer"},
                },
                "required": ["pattern"],
            },
        },
        handler=lambda args: _search_files(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess.read(normalize_component(f"dir:{args.get('path') or '.'}", root)),
        max_result_size_chars=80_000,
    )

    registry.register(
        name="terminal",
        schema={
            "description": (
                "Run a shell command. Unknown workspace mutation is serialized. "
                "timeout is in SECONDS (default 60s, hard cap 600s); do not pass "
                "milliseconds — values above 600 are clamped to 600."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "workdir": {"type": "string"},
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in SECONDS (default 60, max 600). NOT milliseconds.",
                    },
                    "mutates_workspace": {"type": "boolean"},
                    "reads": {"type": "array", "items": {"type": "string"}},
                    "writes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["command"],
            },
        },
        handler=lambda args: _terminal(args, allow_terminal=allow_terminal, approval_manager=approval_manager),
        resource_access=lambda args, root: _process_resource_access(args, root, kind="terminal"),
        barrier="none",
        max_result_size_chars=50_000,
    )

    registry.register(
        name="todo_write",
        schema={
            "description": "Replace the session todo list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {"task": {"type": "string"}, "status": {"type": "string"}},
                        },
                    }
                },
                "required": ["items"],
            },
        },
        handler=_todo_write,
        resource_access=lambda args, root: ResourceAccess.write("memory:todo"),
    )

    registry.register(
        name="todo_read",
        schema={"description": "Read the session todo list.", "parameters": {"type": "object", "properties": {}}},
        handler=_todo_read,
        resource_access=lambda args, root: ResourceAccess.read("memory:todo"),
    )

    registry.register(
        name="run_python",
        schema={
            "description": (
                "Run a short Python script in the workspace. "
                "timeout is in SECONDS (default 30s, hard cap 600s); do not pass "
                "milliseconds — values above 600 are clamped to 600."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in SECONDS (default 30, max 600). NOT milliseconds.",
                    },
                    "mutates_workspace": {"type": "boolean"},
                    "reads": {"type": "array", "items": {"type": "string"}},
                    "writes": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["code"],
            },
        },
        handler=lambda args: _run_python(args, allow_terminal=allow_terminal, approval_manager=approval_manager),
        resource_access=lambda args, root: _process_resource_access(args, root, kind="run_python"),
        max_result_size_chars=50_000,
    )

    registry.register(
        name="run_tests",
        schema={
            "description": (
                "Run a project test command such as pytest. Requires approval unless --yes is active. "
                "timeout is in SECONDS (default 120s, hard cap 600s); do not pass "
                "milliseconds — values above 600 are clamped to 600."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "workdir": {"type": "string"},
                    "timeout": {
                        "type": "number",
                        "description": "Timeout in SECONDS (default 120, max 600). NOT milliseconds.",
                    },
                    "mutates_workspace": {"type": "boolean"},
                    "reads": {"type": "array", "items": {"type": "string"}},
                    "writes": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        handler=lambda args: _run_tests(args, allow_terminal=allow_terminal, approval_manager=approval_manager),
        resource_access=lambda args, root: _process_resource_access(args, root, kind="run_tests"),
        max_result_size_chars=60_000,
    )

    registry.register(
        name="http_fetch",
        schema={
            "description": "Fetch a URL as text for browser-lite/http inspection.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}, "timeout": {"type": "number"}},
                "required": ["url"],
            },
        },
        handler=_http_fetch,
        resource_access=lambda args, root: ResourceAccess.external_read(f"external:{args['url']}"),
        max_result_size_chars=50_000,
    )

    registry.register(
        name="mcp_call",
        schema={
            "description": "Call a configured MCP-like command bridge. v0.4 supports local echo/fake bridges for tests.",
            "parameters": {
                "type": "object",
                "properties": {"server": {"type": "string"}, "tool": {"type": "string"}, "arguments": {"type": "object"}},
                "required": ["server", "tool"],
            },
        },
        handler=lambda args: _mcp_call(args, approval_manager=approval_manager),
        resource_access=lambda args, root: ResourceAccess(unknown=True, side_effect_level="external_write"),
    )

    from high_agent.tools.delegate import register_delegate_task
    register_delegate_task(registry)

    return registry


def _sleep(seconds: float) -> dict[str, Any]:
    time.sleep(max(0.0, seconds))
    return {"slept": seconds}


def _process_resource_access(args: dict[str, Any], root: str | None, *, kind: str = "terminal") -> ResourceAccess:
    explicit_reads = args.get("reads")
    explicit_writes = args.get("writes")
    if explicit_reads is not None or explicit_writes is not None or args.get("mutates_workspace") is False:
        return ResourceAccess(
            reads=frozenset(_file_component(str(path), root) for path in args.get("reads") or []),
            writes=frozenset(_file_component(str(path), root) for path in args.get("writes") or []),
            side_effect_level="local" if args.get("writes") else "none",
        )
    if kind == "terminal":
        composite = _infer_composite_terminal_access(args, root)
        if composite is not None:
            return composite
        inferred = _infer_readonly_terminal_access(args, root)
        if inferred is not None:
            return inferred
        scoped = _infer_scoped_mutation_access(args, root)
        if scoped is not None:
            return scoped
    if kind == "run_tests":
        return _default_test_resource_access(args, root)
    return ResourceAccess.unknown_workspace()


def _default_test_resource_access(args: dict[str, Any], root: str | None) -> ResourceAccess:
    workdir = str(args.get("workdir") or ".")
    reads = frozenset({normalize_component(f"dir:{workdir}", root)})
    writes = frozenset(
        normalize_component(path, root)
        for path in (
            "dir:.pytest_cache",
            "file:.coverage",
            "dir:htmlcov",
            "dir:build",
            "dir:dist",
            "dir:.tox",
            "dir:.nox",
        )
    )
    return ResourceAccess(reads=reads, writes=writes, side_effect_level="local")


def _infer_readonly_terminal_access(args: dict[str, Any], root: str | None) -> ResourceAccess | None:
    command = str(args.get("command") or "").strip()
    if not command or _has_unsafe_shell_syntax(command):
        return None
    if "|" in command:
        return _infer_pipe_safe_terminal_access(command, args, root)
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens:
        return None
    tool = Path(tokens[0]).name
    workdir = str(args.get("workdir") or ".")
    if tool == "pwd":
        return ResourceAccess.empty()
    if tool == "git":
        # `git -C /path status` lays out tokens as
        # ['git', '-C', '/path', 'status']; _first_non_option(tokens[1:])
        # returns '/path' (since '-C' takes a value), so the read-only
        # subcommand check missed and the call fell through to
        # unknown_workspace. _git_subcommand strips the global options
        # (`-C <dir>`, `-c k=v`, `--git-dir=<dir>`, `--work-tree=<dir>`,
        # `--namespace=<name>`) before extracting the subcommand.
        subcommand, sub_tokens = _git_subcommand(tokens[1:])
        if subcommand in {"status", "diff", "show", "log", "grep", "ls-files", "blame", "rev-parse", "branch", "tag"}:
            # 把命令行里出现的具体 path 也提进 reads；缺省回退到 workdir
            extra_reads = _git_path_reads(subcommand, sub_tokens, root)
            base = normalize_component(f"dir:{workdir}", root)
            reads = {base, *extra_reads} if extra_reads else {base}
            return ResourceAccess(reads=frozenset(reads))
        return None
    if tool == "sed":
        if "-i" in tokens or not any(token == "-n" or token.startswith("-n") for token in tokens[1:]):
            return None
        reads = _terminal_path_reads(tokens[1:], root, default_dir=workdir)
        return ResourceAccess(reads=frozenset(reads))
    if tool in {
        "ls", "find", "rg", "grep", "cat", "head", "tail", "wc",
        "awk", "cut", "sort", "uniq", "xxd", "file", "stat", "du",
        "realpath", "basename", "dirname", "column", "nl", "tr",
        "diff", "comm", "tac", "fold", "expand", "od", "hexdump",
        "jq", "yq", "tree", "which", "type", "env", "printenv",
        "date", "uname", "hostname", "id", "whoami",
    }:
        reads = _terminal_path_reads(tokens[1:], root, default_dir=workdir)
        return ResourceAccess(reads=frozenset(reads))
    if tool == "python" or tool == "python3" or tool == "python3.13" or tool == "python3.13t":
        # 仅识别 `python -m json.tool [file]` / `python -m tokenize file` 等
        # 已知无副作用形式；其他 python 调用走 unknown。
        if len(tokens) >= 3 and tokens[1] == "-m" and tokens[2] in {"json.tool", "tokenize", "py_compile", "ast"}:
            reads = _terminal_path_reads(tokens[3:], root, default_dir=workdir)
            return ResourceAccess(reads=frozenset(reads))
        return None
    if tool in _NETWORK_TOOLS:
        return _classify_network_command(tool, tokens, command)
    return None


_NETWORK_TOOLS = {"curl", "wget", "ping", "dig", "host", "nslookup", "http", "https"}
_CURL_WRITE_FLAGS = {
    "--upload-file", "-T",
    "--data", "-d", "--data-binary", "--data-raw", "--data-urlencode",
    "--form", "-F",
}
_WGET_WRITE_HINTS = ("--post-data", "--post-file", "--method=POST", "--method=PUT", "--method=DELETE", "--method=PATCH")


def _classify_network_command(tool: str, tokens: list[str], command: str) -> ResourceAccess:
    """Classify network commands as external_read or external_write.

    Many network probes (curl GET, ping, dig, nslookup) are safe to run in
    parallel — they have no workspace side effects. Marking them as
    external_read instead of unknown_workspace lets multiple probes execute
    concurrently while still serialising against external_write.
    """
    resource = f"external:{tool}"
    if tool == "curl":
        # Honour both `-X METHOD` and the long form `--request METHOD` /
        # `--request=METHOD` previously only `-X` was checked, so
        # `curl --request DELETE ...` was misclassified as external_read).
        method = _curl_method(tokens)
        if method is not None:
            if method in {"GET", "HEAD", "OPTIONS"}:
                return ResourceAccess.external_read(resource)
            return ResourceAccess.external_write(resource)
        if any(flag in tokens for flag in _CURL_WRITE_FLAGS):
            return ResourceAccess.external_write(resource)
        return ResourceAccess.external_read(resource)
    if tool == "wget":
        if any(hint in command for hint in _WGET_WRITE_HINTS):
            return ResourceAccess.external_write(resource)
        return ResourceAccess.external_read(resource)
    # ping / dig / host / nslookup / http / https：纯探测，无写入。
    return ResourceAccess.external_read(resource)


def _curl_method(tokens: list[str]) -> str | None:
    """Extract the HTTP method from curl tokens, honouring -X and --request forms."""
    if "-X" in tokens:
        i = tokens.index("-X")
        if i + 1 < len(tokens):
            return tokens[i + 1].upper()
    if "--request" in tokens:
        i = tokens.index("--request")
        if i + 1 < len(tokens):
            return tokens[i + 1].upper()
    for token in tokens:
        if token.startswith("--request="):
            return token.split("=", 1)[1].upper()
    return None


def _infer_scoped_mutation_access(args: dict[str, Any], root: str | None) -> ResourceAccess | None:
    """Infer fine-grained write access for common mutation commands with predictable scope."""
    command = str(args.get("command") or "").strip()
    if not command or _has_unsafe_shell_syntax(command):
        return None
    if "|" in command:
        return None
    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    if not tokens:
        return None
    tool = Path(tokens[0]).name
    workdir = str(args.get("workdir") or ".")

    if tool in {"mkdir"}:
        # when -p is present mkdir creates every missing ancestor,
        # so the write set must include them. `_paths_overlap`'s prefix
        # check otherwise treats `dir:a/b/c` as overlapping only entries
        # *under* `a/b/c`; a concurrent `write_file file:a/foo` would not
        # conflict even though mkdir is materialising `a` and `a/b` at the
        # same time.
        has_p = any(t == "-p" or (t.startswith("-") and "p" in t[1:] and not t.startswith("--")) for t in tokens[1:])
        paths = [t for t in tokens[1:] if not t.startswith("-")]
        if not paths:
            return None
        write_components: set[str] = set()
        for path in paths:
            write_components.add(normalize_component(f"dir:{path}", root))
            if has_p:
                for ancestor in _path_ancestors(path):
                    write_components.add(normalize_component(f"dir:{ancestor}", root))
        return ResourceAccess(writes=frozenset(write_components), side_effect_level="local")

    if tool == "touch":
        paths = [t for t in tokens[1:] if not t.startswith("-")]
        if not paths:
            return None
        writes = frozenset(normalize_component(f"file:{p}", root) for p in paths)
        return ResourceAccess(writes=writes, side_effect_level="local")

    if tool in {"chmod", "chown"}:
        non_opts = [t for t in tokens[1:] if not t.startswith("-")]
        if len(non_opts) < 2:
            return None
        paths = non_opts[1:]
        writes = frozenset(normalize_component(f"file:{p}", root) for p in paths)
        return ResourceAccess(writes=writes, side_effect_level="local")

    if tool == "git":
        # same global-option handling as the read-only branch.
        subcommand, sub_tokens = _git_subcommand(tokens[1:])
        if subcommand == "init":
            target_args = [t for t in sub_tokens[1:] if not t.startswith("-")]
            target = target_args[0] if target_args else workdir
            writes = frozenset({normalize_component(f"dir:{target}/.git", root)})
            reads = frozenset({normalize_component(f"dir:{target}", root)})
            return ResourceAccess(reads=reads, writes=writes, side_effect_level="local")
        return None

    if tool in {"npm", "yarn", "pnpm", "bun"}:
        # feat-S2: extend the previous install-only branch to recognise the
        # common dev-workflow subcommands. Anything not classified here still
        # falls through to unknown_workspace — wrong parallelisation is more
        # dangerous than over-serialising.
        subcommand = _first_non_option(tokens[1:])
        if subcommand in {"install", "i", "add"}:
            writes = frozenset(
                normalize_component(p, root) for p in (
                    f"dir:{workdir}/node_modules",
                    f"file:{workdir}/package-lock.json",
                    f"file:{workdir}/yarn.lock",
                    f"file:{workdir}/pnpm-lock.yaml",
                )
            )
            reads = frozenset({normalize_component(f"file:{workdir}/package.json", root)})
            return ResourceAccess(reads=reads, writes=writes, side_effect_level="local")
        if subcommand in _NPM_READONLY_SUBCOMMANDS:
            reads = frozenset({normalize_component(f"dir:{workdir}", root)})
            return ResourceAccess(reads=reads, side_effect_level="local")
        run_target = _npm_run_target(tokens, tool)
        if run_target is not None:
            classified = _classify_npm_run_target(run_target, workdir, root)
            if classified is not None:
                return classified
        return None

    if tool == "make":
        targets = [t for t in tokens[1:] if not t.startswith("-")]
        return _classify_make_targets(targets, workdir, root)

    if tool == "cp":
        non_opts = [t for t in tokens[1:] if not t.startswith("-")]
        if len(non_opts) < 2:
            return None
        dest = non_opts[-1]
        sources = non_opts[:-1]
        reads = frozenset(normalize_component(f"file:{s}", root) for s in sources)
        prefix = "dir:" if dest.endswith("/") else "file:"
        writes = frozenset({normalize_component(f"{prefix}{dest}", root)})
        return ResourceAccess(reads=reads, writes=writes, side_effect_level="local")

    if tool == "mv":
        non_opts = [t for t in tokens[1:] if not t.startswith("-")]
        if len(non_opts) < 2:
            return None
        dest = non_opts[-1]
        sources = non_opts[:-1]
        writes = frozenset(
            normalize_component(f"file:{s}", root) for s in sources
        ) | frozenset({normalize_component(f"file:{dest}", root)})
        return ResourceAccess(writes=writes, side_effect_level="local")

    if tool == "rm":
        paths = [t for t in tokens[1:] if not t.startswith("-")]
        if not paths:
            return None
        writes = frozenset(normalize_component(f"file:{p}", root) for p in paths)
        return ResourceAccess(writes=writes, side_effect_level="local")

    if tool == "docker":
        return _classify_docker_command(tokens)

    if tool == "kubectl":
        return _classify_kubectl_command(tokens)

    # Local script invocations: `./build.sh`, `bash deploy.sh`,
    # `python script.py` (no `-m`). The script body is opaque to the
    # heuristic, so we conservatively model it as reading the script file
    # and writing the workdir — same shape as a "limited-scope mutation".
    # This is strictly an upgrade over the previous fall-through to
    # unknown_workspace (global serialization): two scripts in different
    # workdirs can now run in parallel, and a script alongside an
    # unrelated `git status` no longer blocks each other on the
    # unknown-workspace edge. The unknown_workspace fallback still applies
    # when the command contains shell metacharacters (handled upstream by
    # `_has_unsafe_shell_syntax`).
    scoped = _classify_local_script_command(tokens, tool, command, workdir, root)
    if scoped is not None:
        return scoped

    return None


_DOCKER_READONLY_SUBCOMMANDS = frozenset({
    "ps", "inspect", "logs", "images", "image", "version", "info",
    "events", "history", "port", "top", "stats", "diff", "search",
})

_DOCKER_BUILD_SUBCOMMANDS = frozenset({"build", "buildx"})
_DOCKER_REGISTRY_SUBCOMMANDS = frozenset({"push", "pull", "login", "logout"})
_DOCKER_RUN_SUBCOMMANDS = frozenset({"run", "exec", "compose", "start", "restart", "stop", "rm", "kill", "create"})


def _docker_subcommand(tokens: list[str]) -> str:
    """Extract the docker subcommand, skipping leading global flags."""
    for token in tokens[1:]:
        if not token.startswith("-"):
            return token
    return ""


def _classify_docker_command(tokens: list[str]) -> ResourceAccess | None:
    sub = _docker_subcommand(tokens)
    if not sub:
        return None
    if sub in _DOCKER_READONLY_SUBCOMMANDS:
        return ResourceAccess.external_read("external:docker")
    if sub in _DOCKER_BUILD_SUBCOMMANDS:
        return ResourceAccess.external_write("external:docker:image")
    if sub in _DOCKER_REGISTRY_SUBCOMMANDS:
        return ResourceAccess.external_write("external:docker:registry")
    if sub in _DOCKER_RUN_SUBCOMMANDS:
        return ResourceAccess.external_write("external:docker:runtime")
    return None


_KUBECTL_READONLY_VERBS = frozenset({
    "get", "describe", "logs", "top", "explain",
    "api-resources", "api-versions", "cluster-info", "version",
    "config", "auth", "diff",
})

_KUBECTL_WRITE_VERBS = frozenset({
    "apply", "create", "delete", "patch", "replace", "scale",
    "edit", "rollout", "expose", "label", "annotate", "set",
    "drain", "cordon", "uncordon", "taint",
})


def _kubectl_verb(tokens: list[str]) -> str:
    """Extract the kubectl verb, skipping leading global flags."""
    for token in tokens[1:]:
        if not token.startswith("-"):
            return token
    return ""


def _classify_kubectl_command(tokens: list[str]) -> ResourceAccess | None:
    verb = _kubectl_verb(tokens)
    if not verb:
        return None
    if verb in _KUBECTL_READONLY_VERBS:
        return ResourceAccess.external_read("external:kubectl")
    if verb in _KUBECTL_WRITE_VERBS:
        return ResourceAccess.external_write("external:kubectl")
    return None


_SHELL_INTERPRETERS = frozenset({"bash", "sh", "zsh", "ksh", "dash", "fish"})
_PYTHON_INTERPRETERS = frozenset({"python", "python3", "python3.13", "python3.13t"})
_PYTHON_READONLY_M_TARGETS = frozenset({"json.tool", "tokenize", "py_compile", "ast"})


def _classify_local_script_command(
    tokens: list[str],
    tool: str,
    command: str,
    workdir: str,
    root: str | None,
) -> ResourceAccess | None:
    """Classify local script invocations as scoped (file:script + dir:workdir).

    Covers three shapes:
    - ``./path/to/script.sh`` and absolute-path scripts (tool starts with
      ``./``, ``../``, or ``/``).
    - ``bash script.sh`` / ``sh script.sh`` / other POSIX shells with a
      single positional script argument.
    - ``python script.py`` / ``python3 deploy.py`` (no ``-m``; the
      readonly ``-m`` targets are already classified upstream and reach
      this point only on fall-through, which we let happen by returning
      ``None`` here).

    Conservative model: reads={file:script}, writes={dir:workdir}. The
    script body is opaque so we cannot prove a tighter scope, but the
    workdir-level write is still strictly better than unknown_workspace
    (which serialises against every workspace mutation globally).
    """
    if not tokens:
        return None
    first = tokens[0]
    # 1) Direct script path: `./build.sh`, `../tools/run.py`, `/usr/local/bin/script.sh`.
    if first.startswith("./") or first.startswith("../") or first.startswith("/"):
        # Reject if extra positional args contain shell-looking redirects;
        # `_has_unsafe_shell_syntax` already filtered most, but be defensive.
        reads = frozenset({normalize_component(f"file:{first}", root)})
        writes = frozenset({normalize_component(f"dir:{workdir}", root)})
        return ResourceAccess(reads=reads, writes=writes, side_effect_level="local")
    # 2) Shell interpreter: `bash deploy.sh`.
    if tool in _SHELL_INTERPRETERS:
        positional = [t for t in tokens[1:] if not t.startswith("-")]
        if not positional:
            return None
        # Common flag form `bash -c "...";` — `-c` consumes the next token as
        # an inline script body, not a path. Defer to unknown.
        if any(t == "-c" for t in tokens[1:]):
            return None
        script = positional[0]
        reads = frozenset({normalize_component(f"file:{script}", root)})
        writes = frozenset({normalize_component(f"dir:{workdir}", root)})
        return ResourceAccess(reads=reads, writes=writes, side_effect_level="local")
    # 3) Bare python script: `python deploy.py`. The readonly `-m` targets are
    #    handled upstream in `_infer_readonly_terminal_access` and never reach
    #    this branch. `python -c "..."` defers to unknown.
    if tool in _PYTHON_INTERPRETERS:
        if any(t == "-c" for t in tokens[1:]):
            return None
        if any(t == "-m" for t in tokens[1:]):
            # `-m` targets that aren't in the readonly allowlist still
            # might mutate; defer to unknown rather than guess.
            return None
        positional = [t for t in tokens[1:] if not t.startswith("-")]
        if not positional:
            return None
        script = positional[0]
        # Require a `.py`-ish suffix to avoid misclassifying things like
        # `python module_name` (less common but defensive).
        if not script.endswith((".py", ".pyi")):
            return None
        reads = frozenset({normalize_component(f"file:{script}", root)})
        writes = frozenset({normalize_component(f"dir:{workdir}", root)})
        return ResourceAccess(reads=reads, writes=writes, side_effect_level="local")
    return None


_PATH_EXTENSIONS = (
    ".py", ".pyi", ".txt", ".md", ".rst",
    ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".go", ".rs", ".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs",
    ".html", ".htm", ".css", ".scss", ".sass", ".less",
    ".lock", ".sh", ".bash", ".zsh", ".fish",
    ".in", ".out", ".log", ".csv", ".tsv",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".java", ".kt",
    ".xml", ".svg", ".env",
)


def _has_unsafe_shell_syntax(command: str) -> bool:
    """Return True for shell features the per-tool single-segment classifiers cannot handle.

    Notes:
    - Composite separators ``&&`` / ``||`` / ``;`` are unsafe at this layer
      because per-tool classifiers tokenize naively and would misclassify
      `npm test && npm run build` as just `npm test`. Composite commands are
      decomposed by ``_infer_composite_terminal_access`` BEFORE this check
      runs ().
    - ``&`` (background), ``>`` ``<`` (redirect), `` ` `` ``$(`` (subshell)
      remain unsafe because the per-segment classifier cannot reason about
      the indirected target.
    """
    return any(token in command for token in (";", "&", ">", "<", "`", "$("))


_COMPOSITE_SEPARATORS = ("&&", "||", ";")


def _split_composite_command(command: str) -> list[str] | None:
    """Split a top-level composite shell command on ``&&`` / ``||`` / ``;``.

    Returns the list of trimmed segments when ``command`` actually contains a
    composite separator at top level, else ``None``. Quote-aware: separators
    inside single or double quotes are NOT split. Returns ``None`` when any
    segment after splitting is empty (malformed).
    """
    segments: list[str] = []
    buf: list[str] = []
    quote: str | None = None
    i = 0
    n = len(command)
    found_separator = False
    while i < n:
        ch = command[i]
        if quote is not None:
            buf.append(ch)
            if ch == quote and (i == 0 or command[i - 1] != "\\"):
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        # Match longest separator first.
        matched = False
        for sep in _COMPOSITE_SEPARATORS:
            if command.startswith(sep, i):
                segments.append("".join(buf).strip())
                buf = []
                i += len(sep)
                matched = True
                found_separator = True
                break
        if matched:
            continue
        buf.append(ch)
        i += 1
    if quote is not None:
        return None  # unbalanced quote
    if not found_separator:
        return None
    segments.append("".join(buf).strip())
    if any(not s for s in segments):
        return None
    return segments


def _infer_composite_terminal_access(
    args: dict[str, Any], root: str | None
) -> ResourceAccess | None:
    """Decompose top-level && / || / ; into per-segment access and union them.

    Returns ``None`` when the command is not composite, when any segment
    contains a redirect / subshell / background marker, or when any segment
    cannot be classified (so the caller falls back to unknown_workspace).
.
    """
    command = str(args.get("command") or "").strip()
    if not command:
        return None
    segments = _split_composite_command(command)
    if segments is None:
        return None
    workdir = str(args.get("workdir") or ".")
    all_reads: set[str] = set()
    all_writes: set[str] = set()
    elevated_level: str = "none"
    level_priority = {
        "none": 0, "local": 1, "external_read": 2, "external_write": 3,
        "external": 4, "interactive": 5, "unknown": 6,
    }
    for segment in segments:
        seg_args = {"command": segment, "workdir": workdir}
        # Each segment must itself be safe in isolation. _has_unsafe_shell_syntax
        # rejects nested composite (already split here) plus redirects /
        # subshells / background which we cannot decompose.
        if _has_unsafe_shell_syntax(segment):
            return None
        ro = _infer_readonly_terminal_access(seg_args, root)
        if ro is not None:
            access = ro
        else:
            scoped = _infer_scoped_mutation_access(seg_args, root)
            if scoped is None:
                return None
            access = scoped
        all_reads.update(access.reads)
        all_writes.update(access.writes)
        if level_priority.get(access.side_effect_level, 0) > level_priority.get(elevated_level, 0):
            elevated_level = access.side_effect_level
    return ResourceAccess(
        reads=frozenset(all_reads),
        writes=frozenset(all_writes),
        side_effect_level=elevated_level,  # type: ignore[arg-type]
    )


_PIPE_SAFE_TOOLS = frozenset({
    "ls", "find", "rg", "grep", "cat", "head", "tail", "wc",
    "awk", "cut", "sort", "uniq", "xxd", "file", "stat", "du",
    "realpath", "basename", "dirname", "column", "nl", "tr",
    "diff", "comm", "tac", "fold", "expand", "od", "hexdump",
    "jq", "yq", "rev", "paste", "join", "fmt", "pr",
    "strings", "base64", "md5sum", "sha256sum", "sha1sum",
    "sed", "git", "tree", "which", "type", "env", "printenv",
    "date", "uname", "hostname", "id", "whoami",
})


def _infer_pipe_safe_terminal_access(command: str, args: dict[str, Any], root: str | None) -> ResourceAccess | None:
    segments = command.split("|")
    all_reads: set[str] = set()
    workdir = str(args.get("workdir") or ".")
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        if _has_unsafe_shell_syntax(segment):
            return None
        try:
            tokens = shlex.split(segment)
        except ValueError:
            return None
        if not tokens:
            continue
        tool = Path(tokens[0]).name
        if tool == "sed" and "-i" in tokens:
            return None
        if tool not in _PIPE_SAFE_TOOLS:
            return None
        reads = _terminal_path_reads(tokens[1:], root, default_dir=workdir)
        all_reads.update(reads)
    if not all_reads:
        all_reads.add(normalize_component(f"dir:{workdir}", root))
    return ResourceAccess(reads=frozenset(all_reads))


def _first_non_option(tokens: list[str]) -> str:
    for token in tokens:
        if not token.startswith("-"):
            return token
    return ""


_NPM_READONLY_SUBCOMMANDS = frozenset({
    "test", "lint", "typecheck", "check", "audit",
    "outdated", "list", "ls", "view", "info",
})

_NPM_RUN_READONLY_TARGETS = frozenset({
    "test", "test:unit", "test:integration", "test:e2e", "test:watch",
    "lint", "lint:check", "lint:fix",
    "typecheck", "type-check", "tsc",
    "check", "format:check", "fmt:check", "prettier:check",
})

_NPM_RUN_BUILD_TARGETS = frozenset({
    "build", "build:prod", "build:dev", "build:staging",
    "compile", "bundle", "dist", "package",
    "rollup", "webpack",
})

_NPM_RUN_INTERACTIVE_TARGETS = frozenset({
    "start", "dev", "serve", "watch", "preview",
})

_NPM_BUILD_OUTPUT_DIRS = ("dist", "build", ".next", ".nuxt", "out", "lib")


def _npm_run_target(tokens: list[str], tool: str) -> str | None:
    """Extract the script name from `npm run X` / `yarn X` / `pnpm run X` / `bun run X`.

    Returns None when the form is not recognised; caller falls through to
    unknown_workspace so unfamiliar shapes serialise globally.
    """
    if len(tokens) < 2:
        return None
    rest = tokens[1:]
    # Skip leading global options like `--prefix dir`, `-w workspace`.
    i = 0
    while i < len(rest) and rest[i].startswith("-"):
        # Best-effort skip: if the option carries a value (no `=`), skip it too.
        if "=" not in rest[i] and i + 1 < len(rest) and not rest[i + 1].startswith("-"):
            i += 2
        else:
            i += 1
    head = rest[i:]
    if not head:
        return None
    first = head[0]
    if tool in {"npm", "pnpm", "bun", "yarn"} and first in {"run", "run-script"}:
        return head[1] if len(head) >= 2 else None
    # `yarn <script>` and `pnpm <script>` shorthand: only treat as a run target
    # when the first token is *not* a known yarn/pnpm subcommand. We can't
    # enumerate every subcommand, so be conservative: treat as run target only
    # via the explicit `run` form. Returning None lets the caller fall through.
    return None


def _classify_npm_run_target(target: str, workdir: str, root: str | None) -> ResourceAccess | None:
    if not target:
        return None
    if target in _NPM_RUN_READONLY_TARGETS:
        reads = frozenset({normalize_component(f"dir:{workdir}", root)})
        return ResourceAccess(reads=reads, side_effect_level="local")
    if target in _NPM_RUN_BUILD_TARGETS:
        reads = frozenset({normalize_component(f"dir:{workdir}", root)})
        writes = frozenset(
            normalize_component(f"dir:{workdir}/{out}", root)
            for out in _NPM_BUILD_OUTPUT_DIRS
        )
        return ResourceAccess(reads=reads, writes=writes, side_effect_level="local")
    if target in _NPM_RUN_INTERACTIVE_TARGETS:
        return ResourceAccess(side_effect_level="interactive")
    return None


def _classify_make_targets(targets: list[str], workdir: str, root: str | None) -> ResourceAccess | None:
    if not targets:
        return None
    target_set = set(targets)
    clean_targets = {"clean", "distclean", "mostlyclean", "maintainer-clean", "realclean"}
    test_targets = {"test", "tests", "check", "lint", "verify"}
    if target_set <= clean_targets:
        writes = frozenset(
            normalize_component(f"dir:{workdir}/{out}", root)
            for out in ("build", "dist", "target", "out", "obj")
        )
        return ResourceAccess(writes=writes, side_effect_level="local")
    if target_set <= test_targets:
        reads = frozenset({normalize_component(f"dir:{workdir}", root)})
        return ResourceAccess(reads=reads, side_effect_level="local")
    return None


def _path_ancestors(path: str) -> list[str]:
    """Return the proper ancestor directories of `path`, root-side first.

    Used by `mkdir -p` to declare writes on all created intermediate
    directories. Empty / "." / "/" return empty list.
    """
    if not path or path in {".", "/"}:
        return []
    p = Path(path)
    parents = list(p.parents)
    out: list[str] = []
    for parent in parents:
        s = str(parent)
        if s in {".", "/"} or not s:
            continue
        out.append(s)
    return out


# Git global options that consume a following argument.
_GIT_GLOBAL_OPTIONS_WITH_VALUE = frozenset({"-C", "-c"})
# Git global options that take their value attached via `=` (or no value).
_GIT_GLOBAL_OPTIONS_PREFIX = ("--git-dir=", "--work-tree=", "--namespace=", "--exec-path=", "--config-env=", "--super-prefix=")


def _git_subcommand(tokens: list[str]) -> tuple[str, list[str]]:
    """Return (subcommand, tokens-from-subcommand-onward) skipping git globals.

    Handles `-C <dir>`, `-c key=value`, `--git-dir=<dir>`, `--work-tree=<dir>`,
    `--namespace=<name>`, `--exec-path[=<dir>]`, `--super-prefix=<prefix>`, and
    bare flags such as `--no-pager`, `--paginate`, `--bare`. Returns
    (``""``, tokens) when no subcommand is found.
    """
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in _GIT_GLOBAL_OPTIONS_WITH_VALUE:
            i += 2
            continue
        if token.startswith(_GIT_GLOBAL_OPTIONS_PREFIX):
            i += 1
            continue
        if token.startswith("-"):
            # Bare flag (--no-pager / --paginate / --bare / -P / -p / etc.)
            i += 1
            continue
        return token, tokens[i:]
    return "", tokens


def _terminal_path_reads(tokens: list[str], root: str | None, *, default_dir: str) -> set[str]:
    reads: set[str] = set()
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token in {"-e", "-f", "-m", "-C", "--context", "--max-count", "-o", "--output", "--include", "--exclude"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        if token.startswith(('"', "'")):
            continue
        if _looks_like_path(token):
            prefix = "dir:" if token.endswith("/") or token in {".", "./"} else "file:"
            reads.add(normalize_component(f"{prefix}{token}", root))
    if not reads:
        reads.add(normalize_component(f"dir:{default_dir}", root))
    return reads


def _looks_like_path(token: str) -> bool:
    if token in {".", "./"}:
        return True
    if "/" in token:
        return True
    if token.endswith(_PATH_EXTENSIONS):
        return True
    return False


def _git_path_reads(subcommand: str, tokens: list[str], root: str | None) -> set[str]:
    """Best-effort path extraction for `git <sub> [opts] [paths...]`.

    Skips option flags and `--` separator handling is loose. Returns an empty
    set when no obvious path tokens are present so the caller can fall back to
    `dir:{workdir}`."""
    reads: set[str] = set()
    seen_subcommand = False
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if not seen_subcommand:
            if not token.startswith("-"):
                seen_subcommand = True
            continue
        if token in {"-C", "-c", "--git-dir", "--work-tree"}:
            skip_next = True
            continue
        if token.startswith("-"):
            continue
        if token == "--":
            continue
        # `git show HEAD:src/main.py` 这类 ref:path 也提取出来
        if ":" in token and not token.startswith(":"):
            _, _, path_part = token.partition(":")
            if path_part and _looks_like_path(path_part):
                reads.add(normalize_component(f"file:{path_part}", root))
                continue
        if _looks_like_path(token):
            prefix = "dir:" if token.endswith("/") else "file:"
            reads.add(normalize_component(f"{prefix}{token}", root))
    return reads


def _read_file(args: dict[str, Any], *, allow_outside_workspace: bool,
               approval_manager: ApprovalManager | None = None) -> str:
    path = _workspace_path(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    return path.read_text(encoding="utf-8")


def _list_dir(args: dict[str, Any], *, allow_outside_workspace: bool,
              approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    path = _workspace_path(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    entries = []
    for item in sorted(path.iterdir(), key=lambda value: value.name):
        entries.append({"name": item.name, "type": "dir" if item.is_dir() else "file"})
    return {"path": str(path), "entries": entries}


def _list_tree(args: dict[str, Any], *, allow_outside_workspace: bool,
               approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    root = _workspace_path(
        {"path": args.get("path") or ".", "_workspace_root": args.get("_workspace_root")},
        allow_outside_workspace=allow_outside_workspace,
        approval_manager=approval_manager,
    )
    max_depth = max(1, min(int(args.get("max_depth") or 4), 12))
    max_entries = max(1, min(int(args.get("max_entries") or 300), 2000))
    lines: list[str] = [root.name or str(root)]
    count = 0

    def walk(path: Path, prefix: str, depth: int) -> None:
        nonlocal count
        if depth >= max_depth or count >= max_entries or not path.is_dir():
            return
        try:
            children = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name))
        except OSError:
            return
        for index, child in enumerate(children):
            if count >= max_entries:
                break
            count += 1
            branch = "`-- " if index == len(children) - 1 else "|-- "
            lines.append(f"{prefix}{branch}{child.name}")
            walk(child, prefix + ("    " if index == len(children) - 1 else "|   "), depth + 1)

    walk(root, "", 0)
    return {"path": str(root), "tree": "\n".join(lines), "entries": count, "truncated": count >= max_entries}


def _mkdir(args: dict[str, Any], *, allow_outside_workspace: bool,
           approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    path = _workspace_path(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    path.mkdir(parents=True, exist_ok=True)
    return {"path": str(path), "created": True}


def _write_file(args: dict[str, Any], *, allow_outside_workspace: bool,
                approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    path = _workspace_path(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = str(args.get("content", ""))
    path.write_text(content, encoding="utf-8")
    return {"path": str(path), "bytes_written": len(content.encode("utf-8"))}


def _read_many_files(args: dict[str, Any], *, allow_outside_workspace: bool,
                     approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    paths = args.get("paths") or []
    if not isinstance(paths, list):
        raise ValueError("paths must be a list")
    files: list[dict[str, Any]] = []
    for raw in paths:
        path = _workspace_path(
            {"path": str(raw), "_workspace_root": args.get("_workspace_root")},
            allow_outside_workspace=allow_outside_workspace,
            approval_manager=approval_manager,
        )
        files.append({"path": str(path), "content": path.read_text(encoding="utf-8")})
    return {"files": files, "count": len(files)}


def _write_many_files(args: dict[str, Any], *, allow_outside_workspace: bool,
                      approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    files = args.get("files") or []
    if not isinstance(files, list):
        raise ValueError("files must be a list")
    written: list[dict[str, Any]] = []
    for item in files:
        if not isinstance(item, dict):
            raise ValueError("each file item must be an object")
        result = _write_file(
            {"path": item["path"], "content": item.get("content") or "", "_workspace_root": args.get("_workspace_root")},
            allow_outside_workspace=allow_outside_workspace,
            approval_manager=approval_manager,
        )
        written.append(result)
    return {"files": written, "count": len(written)}


def _append_file(args: dict[str, Any], *, allow_outside_workspace: bool,
                 approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    path = _workspace_path(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = str(args.get("content", ""))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(content)
    return {"path": str(path), "bytes_appended": len(content.encode("utf-8"))}


def _replace_in_file(args: dict[str, Any], *, allow_outside_workspace: bool,
                     approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    path = _workspace_path(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    old = str(args["old"])
    new = str(args["new"])
    replace_all = bool(args.get("replace_all", False))
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count == 0:
        raise ValueError("old text was not found")
    if count > 1 and not replace_all:
        raise ValueError(f"old text occurs {count} times; set replace_all=true or make old text unique")
    updated = text.replace(old, new) if replace_all else text.replace(old, new, 1)
    path.write_text(updated, encoding="utf-8")
    return {"path": str(path), "replacements": count if replace_all else 1}


def _patch_file(args: dict[str, Any], *, allow_outside_workspace: bool,
                approval_manager: ApprovalManager | None) -> dict[str, Any]:
    if args.get("diff"):
        return _patch_file_diff(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    path = _workspace_path(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    replacements = args.get("replacements") or []
    if not isinstance(replacements, list):
        raise ValueError("replacements must be a list")
    text = path.read_text(encoding="utf-8")
    applied = 0
    for item in replacements:
        old = str(item.get("old") or "")
        new = str(item.get("new") or "")
        if not old:
            raise ValueError("replacement old text must be non-empty")
        count = text.count(old)
        if count != 1:
            raise ValueError(f"old text must occur exactly once, got {count}: {old[:80]}")
        text = text.replace(old, new, 1)
        applied += 1
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "replacements": applied}


def _patch_file_diff(args: dict[str, Any], *, allow_outside_workspace: bool,
                     approval_manager: ApprovalManager | None) -> dict[str, Any]:
    diff = str(args.get("diff") or "")
    target_path = str(args.get("path") or _diff_target_path(diff) or "")
    if not target_path:
        raise ValueError("diff patch requires path or +++ target")
    path = _workspace_path(
        {"path": target_path, "_workspace_root": args.get("_workspace_root")},
        allow_outside_workspace=allow_outside_workspace,
        approval_manager=approval_manager,
    )
    text = path.read_text(encoding="utf-8")
    replacements = _diff_replacements(diff)
    if not replacements:
        raise ValueError("no simple unified diff hunks found")
    applied = 0
    for old, new in replacements:
        if old not in text:
            raise ValueError(f"diff hunk old block not found: {old[:120]}")
        text = text.replace(old, new, 1)
        applied += 1
    path.write_text(text, encoding="utf-8")
    return {"path": str(path), "hunks": applied}


def _delete_path(args: dict[str, Any], *, allow_outside_workspace: bool,
                 approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    path = _workspace_path(args, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    if not path.exists() and not path.is_symlink():
        return {"path": str(path), "deleted": False, "reason": "not_found"}
    if path.is_dir() and not path.is_symlink():
        if not bool(args.get("recursive", False)):
            raise IsADirectoryError("delete_path requires recursive=true for directories")
        shutil.rmtree(path)
    else:
        path.unlink()
    return {"path": str(path), "deleted": True}


def _move_path(args: dict[str, Any], *, allow_outside_workspace: bool,
               approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    src = _workspace_path({"path": args["src"], "_workspace_root": args.get("_workspace_root")}, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    dst = _workspace_path({"path": args["dst"], "_workspace_root": args.get("_workspace_root")}, allow_outside_workspace=allow_outside_workspace, approval_manager=approval_manager)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {"src": str(src), "dst": str(dst), "moved": True}


def _search_files(args: dict[str, Any], *, allow_outside_workspace: bool,
                  approval_manager: ApprovalManager | None = None) -> dict[str, Any]:
    root = _workspace_path(
        {"path": args.get("path") or ".", "_workspace_root": args.get("_workspace_root")},
        allow_outside_workspace=allow_outside_workspace,
        approval_manager=approval_manager,
    )
    pattern = str(args["pattern"])
    max_results = max(1, min(int(args.get("max_results") or 100), 1000))
    regex = bool(args.get("regex", False))
    rg = shutil.which("rg")
    if rg:
        cmd = [rg, "--line-number", "--no-heading", "--color", "never"]
        if not regex:
            cmd.append("--fixed-strings")
        cmd.extend(["--", pattern, str(root)])
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=20, check=False)
        lines = proc.stdout.splitlines()[:max_results]
        return {"path": str(root), "matches": lines, "truncated": len(proc.stdout.splitlines()) > max_results}
    return _python_search(root, pattern, regex=regex, max_results=max_results)


_SHELL_EXECUTABLE = shutil.which("bash") or "/bin/bash"


def _terminal(args: dict[str, Any], *, allow_terminal: bool,
              approval_manager: ApprovalManager | None = None) -> str | TaskResult:
    if not allow_terminal:
        decision = approval_manager.request(
            ApprovalRequest("terminal", str(args.get("command") or ""), "terminal command")
        ) if approval_manager else None
        if decision is None or not decision.approved:
            return TaskResult.blocked("terminal requires approval", blocked_on=["approval:terminal"])
    # ``shell=True`` without ``executable`` defaults to /bin/sh (dash on
    # Debian/Ubuntu), which lacks brace expansion / process substitution /
    # arrays. Pin bash so commands like ``mkdir -p webtest/{app,tests}`` work
    # the way the model expects.
    # use shared
    # _clamp_timeout_seconds helper so terminal / run_python / run_tests
    # behave identically on ms vs seconds confusion. See helper for
    # full diagnosis (smoke v15 w_fix_001 r0 evidence).
    timeout_seconds = _clamp_timeout_seconds(args.get("timeout"), default=60.0)

    proc = subprocess.run(
        str(args["command"]),
        cwd=args.get("workdir") or args.get("_workspace_root") or None,
        shell=True,
        executable=_SHELL_EXECUTABLE,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    return json.dumps(
        {"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr},
        ensure_ascii=False,
    )


_TODO_ITEMS: list[dict[str, str]] = []


def _clamp_timeout_seconds(raw: Any, default: float, *, hard_cap: float = 600.0) -> float:
    """ defensive clamp shared by
    terminal / run_python / run_tests / http_fetch handlers.

    Reasoning: tool schemas advertise ``timeout: number`` without specifying
    units. gpt-5.4 was observed passing 120000 (intended as 120000 ms = 120 s,
    but subprocess.run treats as 120000 s = 33 h). Smoke v15 w_fix_001 r0 had
    47 unittest-discover calls each effectively unbounded by the tool-side
    timeout, kept alive by the cell-level 2400 s subprocess kill — accumulating
    1029 s of dead tool wall on what should have been quick failures.

    Two defenses:
      1. ``raw >= 1000``  → almost certainly milliseconds; divide by 1000.
      2. clamp to ``[0.5, hard_cap]`` so a single per-command timeout cannot
         eclipse the cell-level subprocess timeout.
    """
    try:
        seconds = float(raw)
    except (TypeError, ValueError):
        seconds = float(default)
    if seconds >= 1000.0:
        seconds = seconds / 1000.0
    return max(0.5, min(seconds, hard_cap))


def _todo_write(args: dict[str, Any]) -> dict[str, Any]:
    global _TODO_ITEMS
    items = args.get("items") or []
    if not isinstance(items, list):
        raise ValueError("items must be a list")
    _TODO_ITEMS = [{"task": str(item.get("task") or ""), "status": str(item.get("status") or "pending")} for item in items]
    return {"items": _TODO_ITEMS, "count": len(_TODO_ITEMS)}


def _todo_read(args: dict[str, Any]) -> dict[str, Any]:
    return {"items": list(_TODO_ITEMS), "count": len(_TODO_ITEMS)}


def _run_python(args: dict[str, Any], *, allow_terminal: bool,
                approval_manager: ApprovalManager | None = None) -> str | TaskResult:
    if not allow_terminal:
        decision = approval_manager.request(
            ApprovalRequest("run_python", "python", "code execution")
        ) if approval_manager else None
        if decision is None or not decision.approved:
            return TaskResult.blocked("run_python requires approval", blocked_on=["approval:run_python"])
    with tempfile.NamedTemporaryFile("w", suffix=".py", encoding="utf-8", delete=False) as handle:
        handle.write(str(args["code"]))
        script = handle.name
    try:
        proc = subprocess.run(
            [sys.executable, script],
            cwd=args.get("_workspace_root") or None,
            text=True,
            capture_output=True,
            timeout=_clamp_timeout_seconds(args.get("timeout"), default=30.0),
            check=False,
        )
        return json.dumps({"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}, ensure_ascii=False)
    finally:
        try:
            os.unlink(script)
        except OSError:
            pass


def _run_tests(args: dict[str, Any], *, allow_terminal: bool,
               approval_manager: ApprovalManager | None = None) -> str | TaskResult:
    command = str(args.get("command") or "pytest")
    if not allow_terminal:
        decision = approval_manager.request(
            ApprovalRequest("run_tests", command, "test command")
        ) if approval_manager else None
        if decision is None or not decision.approved:
            return TaskResult.blocked("run_tests requires approval", blocked_on=["approval:run_tests"])
    proc = subprocess.run(
        command,
        cwd=args.get("workdir") or args.get("_workspace_root") or None,
        shell=True,
        executable=_SHELL_EXECUTABLE,
        text=True,
        capture_output=True,
        timeout=_clamp_timeout_seconds(args.get("timeout"), default=120.0),
        check=False,
    )
    payload = json.dumps({"exit_code": proc.returncode, "stdout": proc.stdout, "stderr": proc.stderr}, ensure_ascii=False)
    if proc.returncode != 0:
        return TaskResult.failed(payload, error_type="test_failure", retryable=True)
    return payload


def _http_fetch(args: dict[str, Any]) -> str:
    request = urllib.request.Request(str(args["url"]), headers={"user-agent": "high-agent/0.4"})
    with urllib.request.urlopen(request, timeout=float(args.get("timeout", 20))) as response:
        raw = response.read(1_000_000)
    return raw.decode("utf-8", errors="replace")


def _mcp_call(args: dict[str, Any], *, approval_manager: ApprovalManager | None = None) -> dict[str, Any] | TaskResult:
    server = str(args.get("server") or "")
    tool = str(args.get("tool") or "")
    decision = approval_manager.request(
        ApprovalRequest("mcp_call", f"{server}:{tool}", "MCP bridge call")
    ) if approval_manager else None
    if decision is None or not decision.approved:
        return TaskResult.blocked("mcp_call requires approval", blocked_on=["approval:mcp"])
    return {"server": server, "tool": tool, "arguments": args.get("arguments") or {}, "ok": True}


def _workspace_path(args: dict[str, Any], *, allow_outside_workspace: bool,
                    approval_manager: ApprovalManager | None = None) -> Path:
    root_raw = args.get("_workspace_root")
    raw = Path(str(args["path"])).expanduser()
    if not raw.is_absolute():
        raw = Path(root_raw or ".") / raw
    path = raw.resolve()
    if root_raw and not allow_outside_workspace:
        root = Path(str(root_raw)).resolve()
        if path != root and root not in path.parents and not _is_trusted_project_path(path, root):
            decision = approval_manager.request(
                ApprovalRequest("outside_workspace_path", str(path), f"workspace root is {root}")
            ) if approval_manager else None
            if decision is None or not decision.approved:
                raise PermissionError(f"path is outside workspace: {path}")
    return path


def _is_trusted_project_path(path: Path, root: Path) -> bool:
    """Allow sibling/parent project files without prompting for approval."""
    parent = root.parent
    if parent == root:
        return False
    home = Path.home().resolve()
    if parent == home or parent == home.parent:
        return False
    return path == parent or parent in path.parents


def _file_component(path: str, root: str | None) -> str:
    return normalize_component(f"file:{path}", root)


def _python_search(root: Path, pattern: str, *, regex: bool, max_results: int) -> dict[str, Any]:
    import re

    compiled = re.compile(pattern) if regex else None
    matches: list[str] = []
    paths = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
    for path in paths:
        if len(matches) >= max_results:
            break
        try:
            rel = os.fspath(path)
            for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                found = bool(compiled.search(line)) if compiled else pattern in line
                if found:
                    matches.append(f"{rel}:{line_no}:{line}")
                    if len(matches) >= max_results:
                        break
        except (UnicodeDecodeError, OSError):
            continue
    return {"path": str(root), "matches": matches, "truncated": len(matches) >= max_results}


def _diff_target_path(diff: str) -> str:
    for line in diff.splitlines():
        if line.startswith("+++ "):
            value = line[4:].strip()
            if value.startswith("b/"):
                value = value[2:]
            if value != "/dev/null":
                return value
    return ""


def _diff_replacements(diff: str) -> list[tuple[str, str]]:
    replacements: list[tuple[str, str]] = []
    old_lines: list[str] = []
    new_lines: list[str] = []
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith("@@"):
            if in_hunk and (old_lines or new_lines):
                replacements.append(("\n".join(old_lines) + "\n", "\n".join(new_lines) + "\n"))
            old_lines = []
            new_lines = []
            in_hunk = True
            continue
        if not in_hunk or line.startswith(("--- ", "+++ ")):
            continue
        if line.startswith("-"):
            old_lines.append(line[1:])
        elif line.startswith("+"):
            new_lines.append(line[1:])
        elif line.startswith(" "):
            old_lines.append(line[1:])
            new_lines.append(line[1:])
    if in_hunk and (old_lines or new_lines):
        replacements.append(("\n".join(old_lines) + "\n", "\n".join(new_lines) + "\n"))
    return replacements
