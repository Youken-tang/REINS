"""scoring — REINS-Bench evaluation gates and scheduler_score aggregator.

 Two metrics, reported separately:

- ``pass_rate`` — fraction of (prompt, run) cells that pass every gate
  in the prompt's ``expected:`` block (min_files exist, must_pass_tests
  all return zero, no forbidden_pattern matches), plus the run stayed
  inside ``budget`` (wall_seconds, input/output tokens).
- ``scheduler_score`` — geometric mean ratio of wall_seconds / token
  totals against a baseline system, computed only over the **passing
  subset of the candidate** so a fast-but-wrong run can never win.

The scoring layer is intentionally trace-agnostic: it consumes
``CellRecord`` (a flat dict) so every adapter — REINS, Ray-wrapper,
sequential, third-party — can produce the same record shape from its
own artefacts. ``cell_record_from_artifacts`` builds one such record
from a ``T1d_*/T2a_*/T2b_*`` directory; the scoring CLI in
``scripts/score.py`` walks any results tree and folds it into a
``report.json``.
"""

from __future__ import annotations

import fnmatch
import json
import math
import os
import re
import shlex
import statistics
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .schema import TaskSpec


# ──────────────────────────── Fuzzy path helpers ────────────────────────────
#
# REINS-Bench v2 oracle 在 ``min_files`` / ``must_pass_tests`` 里硬编码了文件名
# (e.g. ``tests/test_users_pagination.py``)；多数 agent 实现只要写出"语义合理"
# 的回归测试 (e.g. ``tests/test_users.py``) 就足够 oracle 测出 patch 是否正确。
# 把 oracle 路径当成精确字符串匹配会把这些 cell 一律判 fail，掩盖真实算法收益。
#
# 下列 helper 把 oracle 路径当 *hint* 用——支持精确命中、glob、token-fuzzy、
# 文件名兜底四级 fallback。每次 fallback 决策记入 gate.detail 以便论文 交代。


_TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".sh", ".ts", ".tsx", ".js", ".jsx", ".html", ".css", ".rst", ".env",
    ".dockerfile", ".lock",
}
_SKIP_DIRS = {
    "__pycache__", ".git", ".hg", "node_modules", ".venv", "venv", ".tox",
    "build", "dist", ".mypy_cache", ".pytest_cache", ".ruff_cache",
    "target", "out",
}
_STOP_TOKENS = {
    "tests", "test", "src", "app", "py", "js", "ts", "the", "a", "an",
}

# 跨 cell 共享的 pip --target 缓存目录。装一次累积，不污染 grader 主 venv。
# pip 装包前用 fcntl 锁串行化，避免 grader 多 worker 并发装同一个包冲突。
_PIP_CACHE_DIR = Path(__file__).resolve().parents[1] / "T3v2" / "_pip_cache"
_PIP_LOCK_FILE = _PIP_CACHE_DIR / ".lock"
# requirements.txt 里 agent 偶尔写这些"惯用占位"——pip 装会失败但无害，过滤掉。
_REQ_BLACKLIST = re.compile(
    r"^\s*(?:#|--|-e\s+\.|-e\s+\.\.|"
    r"\.|\.\.|"
    r"your[-_]?package|placeholder|TODO|local[-_]?lib"
    r")", re.IGNORECASE)

# import name → pip 包名映射。覆盖 W_scaffold corpus 常见情况：agent 写
# ``from flask_login import LoginManager``，pip 包是 ``Flask-Login``；
# ``import jwt`` → ``PyJWT``；等等。未列出的 import 名直接当作 pip 名用。
_IMPORT_TO_PIP: dict[str, str] = {
    "jwt": "PyJWT",
    "yaml": "PyYAML",
    "cv2": "opencv-python",
    "bs4": "beautifulsoup4",
    "PIL": "Pillow",
    "sklearn": "scikit-learn",
    "google": "protobuf",
    "attr": "attrs",
    "flask_login": "Flask-Login",
    "flask_sqlalchemy": "Flask-SQLAlchemy",
    "flask_migrate": "Flask-Migrate",
    "flask_jwt_extended": "Flask-JWT-Extended",
    "flask_wtf": "Flask-WTF",
    "flask_cors": "Flask-Cors",
    "flask_caching": "Flask-Caching",
    "flask_restx": "flask-restx",
    "wtforms": "WTForms",
    "email_validator": "email-validator",
    "passlib": "passlib[bcrypt]",
    "confluent_kafka": "confluent-kafka",
    "google_auth": "google-auth",
    "googleapiclient": "google-api-python-client",
    "_pytest": "pytest",
    "pytest_asyncio": "pytest-asyncio",
    "pytest_mock": "pytest-mock",
    "pytest_xdist": "pytest-xdist",
    "dateutil": "python-dateutil",
}

# 当 cell 没声明 requirements 且 import-scan 扫不到时的 corpus-final-fallback。
# 这些包在 W_scaffold corpus 的 oracle 命令里反复出现（fastapi/flask/grpc/celery/...）。
# 装一次跨所有 cell 复用，避免"agent 写了完整代码但忘了 requirements.txt"被误判 fail。
_CORPUS_FALLBACK_DEPS: tuple[str, ...] = (
    "fastapi", "uvicorn", "starlette", "httpx", "pydantic",
    "Flask", "Flask-Login", "Flask-SQLAlchemy", "Flask-Migrate", "Flask-WTF",
    "Flask-JWT-Extended", "Flask-Cors", "Flask-Caching", "WTForms",
    "SQLAlchemy", "alembic", "asyncpg", "psycopg2-binary",
    "celery", "redis", "fakeredis",
    "PyJWT", "bcrypt", "passlib[bcrypt]", "email-validator",
    "duckdb", "tornado", "respx",
    "grpcio", "grpcio-tools", "protobuf",
    "confluent-kafka",
    "PyYAML", "python-dateutil", "requests", "aiohttp",
    "pytest-asyncio", "pytest-mock",
)


def _path_tokens(path: str) -> list[str]:
    """把路径拆成可用作 fuzzy 匹配的 token 集合。

    例：``tests/test_users_pagination.py`` → ['users', 'pagination']
    剥掉 stem、test_ 前缀、扩展名、stop tokens。
    """
    p = Path(path)
    stem = p.stem
    stem = re.sub(r"^test[_-]", "", stem)
    raw = re.split(r"[\W_]+", stem)
    return [t.lower() for t in raw if t and t.lower() not in _STOP_TOKENS]


def _is_text_file(path: Path) -> bool:
    try:
        if path.is_symlink() or not path.is_file():
            return False
    except OSError:
        return False
    if path.suffix.lower() in _TEXT_EXTS:
        return True
    if path.name in {"Dockerfile", "Makefile", "setup.cfg"}:
        return True
    return False


def _walk_workspace(workspace: Path) -> Iterable[Path]:
    """rglob 但跳过 _SKIP_DIRS。"""
    for root, dirs, files in os.walk(workspace):
        dirs[:] = [d for d in dirs if d not in _SKIP_DIRS]
        root_p = Path(root)
        for f in files:
            yield root_p / f


# ──────────────────────────── Dependency install ───────────────────────────


def _read_requirements(workspace: Path) -> list[str]:
    """从 workspace/requirements.txt 读出 pip 安装条目，跳过空行 / 注释 / 占位符。"""
    req = workspace / "requirements.txt"
    if not req.is_file():
        return []
    out: list[str] = []
    try:
        for raw in req.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            if _REQ_BLACKLIST.match(line):
                continue
            out.append(line)
    except OSError:
        return []
    return out


def _read_pyproject_deps(workspace: Path) -> list[str]:
    """从 pyproject.toml 的 [project.dependencies] 拉 deps；解析失败返回 []."""
    pp = workspace / "pyproject.toml"
    if not pp.is_file():
        return []
    try:
        # tomllib stdlib 3.11+
        import tomllib
        doc = tomllib.loads(pp.read_text(encoding="utf-8"))
    except Exception:
        return []
    deps: list[str] = []
    project = doc.get("project") or {}
    for d in project.get("dependencies") or []:
        if isinstance(d, str) and not _REQ_BLACKLIST.match(d):
            deps.append(d)
    # poetry 风格 [tool.poetry.dependencies] 也兼容
    poetry = (doc.get("tool") or {}).get("poetry") or {}
    for k, v in (poetry.get("dependencies") or {}).items():
        if k.lower() == "python":
            continue
        if isinstance(v, str):
            spec = v if v[0] in {">", "<", "=", "~", "^", "!"} else f">={v}"
            deps.append(f"{k}{spec}")
        else:
            deps.append(k)
    return deps


def _scan_workspace_imports(workspace: Path) -> list[str]:
    """扫 workdir 所有 .py 的 top-level import，把不在 stdlib + workdir 本地的当三方包。

    常用于 agent 写了完整代码但忘了 requirements.txt 的 cell。映射经过 _IMPORT_TO_PIP
    转成 pip 包名。
    """
    import ast
    import sys as _sys

    stdlib = set(_sys.stdlib_module_names)
    # workdir 内所有顶层目录 + 顶层 .py 文件名都视作"本地包"，不当三方装。
    local_pkgs: set[str] = set()
    for entry in workspace.iterdir() if workspace.is_dir() else []:
        if entry.is_dir() and not entry.name.startswith(".") and entry.name not in _SKIP_DIRS:
            local_pkgs.add(entry.name)
        elif entry.is_file() and entry.suffix == ".py":
            local_pkgs.add(entry.stem)

    seen: set[str] = set()
    out: list[str] = []
    for p in _walk_workspace(workspace):
        if p.suffix.lower() != ".py":
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8", errors="ignore"))
        except (SyntaxError, ValueError, OSError):
            continue
        for node in ast.walk(tree):
            mod_name: str | None = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    _maybe_add(top, stdlib, local_pkgs, seen, out)
            elif isinstance(node, ast.ImportFrom):
                if node.level != 0 or not node.module:
                    continue
                top = node.module.split(".")[0]
                _maybe_add(top, stdlib, local_pkgs, seen, out)
    return out


def _maybe_add(top: str, stdlib: set[str], local_pkgs: set[str],
               seen: set[str], out: list[str]) -> None:
    if not top or top in stdlib or top in local_pkgs:
        return
    if top in seen:
        return
    seen.add(top)
    pip_name = _IMPORT_TO_PIP.get(top, top)
    out.append(pip_name)


def _ensure_workspace_deps(workspace: Path, *, install_timeout: float = 30.0) -> str:
    """Best-effort: 把 workspace 声明 + import-scan 推断的依赖装到 _PIP_CACHE_DIR。

    顺序：cell 自己的 requirements.txt / pyproject.toml → 扫 .py top-level import →
    若以上都没有但 cell 看起来像 W_scaffold 项目（含 ``app/`` 或 ``tests/``）就装一次
    corpus 通用 fallback。所有装失败容错——返回 note 字符串记入 gate.detail。
    """
    declared = _read_requirements(workspace) + _read_pyproject_deps(workspace)
    scanned = _scan_workspace_imports(workspace)
    deps_set: set[str] = set(declared)
    for s in scanned:
        deps_set.add(s)
    deps = sorted(deps_set)
    fallback_used = False
    if not deps:
        # cell 没声明 + 没扫到 → 用 corpus fallback（仅当目录结构像 W_scaffold）
        if (workspace / "tests").is_dir() or (workspace / "app").is_dir():
            deps = list(_CORPUS_FALLBACK_DEPS)
            fallback_used = True
        else:
            return ""
    elif len(deps) < 3 and ((workspace / "tests").is_dir() or (workspace / "app").is_dir()):
        # cell 只声明了少量 dep（agent 写漏）但目录结构像 W_scaffold，叠加 fallback
        deps = sorted(deps_set | set(_CORPUS_FALLBACK_DEPS))
        fallback_used = True

    _PIP_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _PIP_LOCK_FILE.touch(exist_ok=True)

    # _pip_cache 是 cp310 wheel（cp313t noGIL 上很多 C-ext 包没有 wheel），
    # 跑 pytest 也走 cp310 (见 _run_test_command)。装包同样用 cp310。
    cp310 = Path("/usr/bin/python3")
    uv_bin = Path("/home/yhsim/.local/bin/uv")
    cmd: list[str]
    if uv_bin.is_file() and cp310.is_file():
        cmd = [
            str(uv_bin), "pip", "install",
            "--python", str(cp310),
            "--target", str(_PIP_CACHE_DIR),
            "--quiet", "--no-progress",
            *deps,
        ]
    elif cp310.is_file():
        cmd = [
            str(cp310), "-m", "pip", "install",
            "--target", str(_PIP_CACHE_DIR),
            "--quiet", "--disable-pip-version-check",
            "--no-input", "--upgrade-strategy", "only-if-needed",
            *deps,
        ]
    else:
        return f"[deps_skip: no cp310 / uv found]"

    try:
        import fcntl
    except ImportError:
        fcntl = None  # type: ignore[assignment]

    lock_fh = None
    if fcntl is not None:
        try:
            lock_fh = open(_PIP_LOCK_FILE, "r+")
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        except OSError:
            lock_fh = None
    try:
        proc = subprocess.run(  # noqa: S603 — pip install of agent-declared deps
            cmd, capture_output=True, text=True, timeout=install_timeout, check=False,
        )
    except subprocess.TimeoutExpired:
        return f"[deps_timeout: {install_timeout}s on {len(deps)} dep(s)]"
    except OSError as exc:
        return f"[deps_oserror: {exc}]"
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)  # type: ignore[union-attr]
                lock_fh.close()
            except OSError:
                pass

    tag = f"deps_{'fallback_ok' if fallback_used else 'ok'}"
    if proc.returncode == 0:
        return f"[{tag}: {len(deps)} dep(s)]"
    err_tail = (proc.stderr or proc.stdout or "")[-200:]
    return f"[deps_partial: rc={proc.returncode} on {len(deps)} dep(s): {err_tail.strip()}]"


# ──────────────────────────── Pass-rate gates ───────────────────────────────


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class PassResult:
    """Outcome of running every gate against one (prompt, run)."""

    passed: bool
    gates: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in self.gates
            ],
        }


def _check_min_files(
    workspace: Path, patterns: Sequence[str]
) -> tuple[GateResult, dict[str, str]]:
    """每条 oracle 路径独立判断：精确 → glob → fuzzy token → 文件名子串。

    返回：(GateResult, {oracle_path: matched_path}) ——matched_path 供
    must_pass_tests 命令 rewrite 使用，并记入 gate.detail 让论文 交代。
    """
    if not patterns:
        return GateResult(
            name="min_files", passed=True, detail="no patterns declared"), {}
    if not workspace.exists():
        return (
            GateResult(name="min_files", passed=False,
                       detail=f"workspace {workspace} missing"),
            {},
        )

    matched: dict[str, str] = {}
    missing: list[str] = []
    notes: list[str] = []
    all_files = [p for p in _walk_workspace(workspace) if p.is_file()]

    for oracle_path in patterns:
        # 1) 精确路径 / glob
        if "*" in oracle_path or "?" in oracle_path or "[" in oracle_path:
            hits = list(workspace.glob(oracle_path))
            if hits:
                matched[oracle_path] = str(hits[0].relative_to(workspace))
                continue
        else:
            exact = workspace / oracle_path
            if exact.exists():
                matched[oracle_path] = oracle_path
                continue

        oracle_p = Path(oracle_path)
        oracle_tokens = _path_tokens(oracle_path)
        oracle_dir = oracle_p.parent
        oracle_stem_simple = re.sub(r"^test[_-]", "", oracle_p.stem).lower()

        # 2) 同 dir 下找：扩展名一致 + 文件名含任一 token
        candidates: list[tuple[int, Path]] = []
        for f in all_files:
            try:
                rel = f.relative_to(workspace)
            except ValueError:
                continue
            if f.suffix.lower() != oracle_p.suffix.lower():
                continue
            stem_simple = re.sub(r"^test[_-]", "", f.stem).lower()
            score = 0
            if str(rel.parent) == str(oracle_dir):
                score += 4
            elif rel.parts and oracle_dir.parts and rel.parts[0] == oracle_dir.parts[0]:
                score += 2
            hits = sum(1 for t in oracle_tokens if t and t in stem_simple)
            score += hits * 3
            if oracle_stem_simple and (
                oracle_stem_simple in stem_simple
                or stem_simple in oracle_stem_simple
            ):
                score += 2
            if score > 0 and (
                hits > 0
                or stem_simple in oracle_stem_simple
                or oracle_stem_simple in stem_simple
            ):
                candidates.append((score, f))

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            best = candidates[0][1]
            matched[oracle_path] = str(best.relative_to(workspace))
            notes.append(f"{oracle_path}→{matched[oracle_path]}")
            continue

        # 3) 文件名兜底（"app/main.py" → "src/app/main.py"）
        same_name = [f for f in all_files if f.name == oracle_p.name]
        if same_name:
            matched[oracle_path] = str(same_name[0].relative_to(workspace))
            notes.append(f"{oracle_path}~name→{matched[oracle_path]}")
            continue

        missing.append(oracle_path)

    detail = ""
    if missing:
        detail = f"missing: {missing}"
    elif notes:
        detail = "fuzzy: " + ", ".join(notes)
    return GateResult(
        name="min_files", passed=not missing, detail=detail), matched


def _check_forbidden_patterns(
    workspace: Path, patterns: Sequence[str]
) -> GateResult:
    if not patterns:
        return GateResult(
            name="forbidden_patterns",
            passed=True,
            detail="no patterns declared",
        )
    if not workspace.exists():
        return GateResult(
            name="forbidden_patterns",
            passed=True,
            detail="workspace absent (vacuously clean)",
        )
    compiled = [re.compile(p) for p in patterns]
    hits: list[str] = []
    for path in _walk_workspace(workspace):
        if not _is_text_file(path):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for rx in compiled:
            if rx.search(text):
                try:
                    rel = path.relative_to(workspace)
                except ValueError:
                    rel = path
                hits.append(f"{rel}::{rx.pattern}")
                break
    return GateResult(
        name="forbidden_patterns",
        passed=not hits,
        detail="" if not hits else f"matches: {hits[:5]}",
    )


def _resolve_test_command(
    workspace: Path, command: str, matched_min_files: dict[str, str]
) -> str | None:
    """如果命令里指定的路径在 workspace 不存在，自动 rewrite。

    优先级：oracle path 同义替换 → 同 dir 下 fuzzy 匹配的 test_*.py →
    退化到父目录 → 退化到 ``.``（让 pytest 自 discover）。
    """
    try:
        toks = shlex.split(command)
    except ValueError:
        return None
    if not toks:
        return None

    new_toks: list[str] = []
    rewrites_made = False
    all_files: list[Path] | None = None  # 懒加载

    for tok in toks:
        if tok.startswith("-") or ("/" not in tok and "." not in tok):
            new_toks.append(tok)
            continue
        path_candidate = Path(tok)
        if (workspace / path_candidate).exists():
            new_toks.append(tok)
            continue

        # oracle path 同义替换
        replaced = False
        for oracle_path, real_path in matched_min_files.items():
            if path_candidate == Path(oracle_path) or tok == oracle_path:
                new_toks.append(real_path)
                replaced = True
                rewrites_made = True
                break
        if replaced:
            continue

        # 同 stem token 的 test 文件
        if all_files is None:
            all_files = list(_walk_workspace(workspace))
        oracle_stem = re.sub(r"^test[_-]", "", path_candidate.stem).lower()
        oracle_tokens = _path_tokens(tok)
        # 严格只接受 ``tests/`` 或 ``test/`` 目录下、文件名以 ``test_`` 开头的
        # py 文件——避免把 migrations/0001_create_blog_tables.py 这种"含 blog
        # 关键字"的非测试文件误当作测试 rewrite 出去。
        same_dir_tests = [
            f for f in all_files
            if f.suffix.lower() == ".py"
            and f.name.startswith("test_")
            and any(part in {"tests", "test"} for part in f.parts)
        ]
        scored: list[tuple[int, Path]] = []
        for f in same_dir_tests:
            stem = re.sub(r"^test[_-]", "", f.stem).lower()
            sc = sum(2 for t in oracle_tokens if t and t in stem)
            if oracle_stem and oracle_stem in stem:
                sc += 1
            if sc > 0:
                scored.append((sc, f))
        if scored:
            scored.sort(key=lambda x: -x[0])
            new_toks.append(str(scored[0][1].relative_to(workspace)))
            rewrites_made = True
            continue

        # 父目录跑全部测试
        parent = path_candidate.parent
        if parent.parts and (workspace / parent).exists():
            new_toks.append(str(parent))
            rewrites_made = True
            continue

        # tests/ 这种通用根
        if path_candidate.parts and path_candidate.parts[0] in {"tests", "test"}:
            test_root = workspace / path_candidate.parts[0]
            if test_root.exists():
                new_toks.append(path_candidate.parts[0])
                rewrites_made = True
                continue

        new_toks.append(".")
        rewrites_made = True

    if not rewrites_made:
        return command
    return " ".join(shlex.quote(t) for t in new_toks)


def _run_test_command(
    workspace: Path, command: str, timeout: float,
    matched_min_files: dict[str, str] | None = None,
    *,
    deps_note: str = "",
) -> tuple[bool, str]:
    if not workspace.exists():
        return False, f"workspace {workspace} missing"
    # Cell workdirs hand-authored by the agent rarely include __init__.py
    # files or a pyproject.toml; without PYTHONPATH=workspace the test
    # commands fail collection with ModuleNotFoundError because pytest's
    # default rootdir discovery does not put the workspace on sys.path.
    # We also prepend _PIP_CACHE_DIR so any deps installed by
    # _ensure_workspace_deps are import-resolvable from the test process.
    abs_workspace = workspace.resolve()
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [str(abs_workspace)]
    if _PIP_CACHE_DIR.is_dir():
        paths.append(str(_PIP_CACHE_DIR))
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)

    matched = matched_min_files or {}
    rewritten = _resolve_test_command(workspace, command, matched) or command
    note = ""
    if rewritten != command:
        note = f" [rewrote: {command!r}→{rewritten!r}]"
    if deps_note:
        note += f" {deps_note}"

    # 子进程用 cp310 (system Python) 跑 pytest——_pip_cache 是 cp310 wheel，
    # 而 grader 主 venv 是 cp313t (noGIL) 跑 pytest 时会因 ABI 不匹配 import
    # 失败 (e.g. "_duckdb.cpython-310-x86_64-linux-gnu.so" 不能 load 到 cp313t)。
    # 把 cmd[0] 是 pytest / python / py.test 的命令重定向到 /usr/bin/python3 -m pytest。
    try:
        toks = shlex.split(rewritten)
    except ValueError:
        toks = [rewritten]
    cp310 = Path("/usr/bin/python3")
    if toks and cp310.is_file():
        head = toks[0]
        if head in ("pytest", "py.test"):
            toks = [str(cp310), "-m", "pytest", *toks[1:]]
        elif head == "python" or head.startswith("python3"):
            toks = [str(cp310), *toks[1:]]

    try:
        proc = subprocess.run(  # noqa: S603 — workspace-scoped, prompt-authored
            toks,
            cwd=str(workspace),
            timeout=timeout,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
    except FileNotFoundError as e:
        return False, f"missing binary: {e}{note}"
    except subprocess.TimeoutExpired:
        return False, f"timeout after {timeout}s{note}"
    if proc.returncode == 0:
        return True, note.strip()
    tail = (proc.stderr or proc.stdout or "")[-400:]
    return False, f"exit {proc.returncode}: {tail.strip()}{note}"


def _check_must_pass_tests(
    workspace: Path,
    commands: Sequence[str],
    *,
    timeout_per_command: float,
    run_tests: bool,
    matched_min_files: dict[str, str] | None = None,
) -> GateResult:
    if not commands:
        return GateResult(
            name="must_pass_tests",
            passed=True,
            detail="no commands declared",
        )
    if not run_tests:
        return GateResult(
            name="must_pass_tests",
            passed=True,
            detail=f"skipped ({len(commands)} command(s) — run_tests=False)",
        )
    # 跑 pytest 之前装 workspace 声明的依赖；非阻塞失败容错。
    deps_note = _ensure_workspace_deps(workspace)
    failures: list[str] = []
    notes: list[str] = []
    for cmd in commands:
        ok, detail = _run_test_command(
            workspace, cmd, timeout_per_command,
            matched_min_files=matched_min_files,
            deps_note=deps_note,
        )
        if not ok:
            failures.append(f"{cmd!r}: {detail}")
        elif detail:
            notes.append(detail)
    return GateResult(
        name="must_pass_tests",
        passed=not failures,
        detail="; ".join(failures) if failures else "; ".join(notes),
    )


def _check_budget(
    *,
    wall_seconds: float | None,
    input_tokens: int | None,
    output_tokens: int | None,
    budget_max_wall: float,
    budget_max_input: int,
    budget_max_output: int,
) -> GateResult:
    breaches: list[str] = []
    if wall_seconds is not None and wall_seconds > budget_max_wall:
        breaches.append(f"wall {wall_seconds:.2f}>{budget_max_wall}")
    if input_tokens is not None and input_tokens > budget_max_input:
        breaches.append(f"in_tokens {input_tokens}>{budget_max_input}")
    if output_tokens is not None and output_tokens > budget_max_output:
        breaches.append(f"out_tokens {output_tokens}>{budget_max_output}")
    return GateResult(
        name="budget",
        passed=not breaches,
        detail="" if not breaches else "; ".join(breaches),
    )


def evaluate_pass(
    spec: TaskSpec,
    *,
    workspace: Path,
    wall_seconds: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    run_tests: bool = False,
    timeout_per_command: float = 60.0,
) -> PassResult:
    """Run every gate against one cell's workspace + cost numbers.

    ``run_tests`` defaults to False so unit tests / replay flows can
    score deterministically without spawning subprocesses; the runner
    sets it True for live evaluation.

    Path matching is fuzzy: ``min_files`` 命中精确 / glob / token-fuzzy /
    文件名兜底任一即过；``must_pass_tests`` 命令里指向不存在的路径会自动
    rewrite 到 workspace 内的同义文件 / 父目录。fallback 决策记入 gate.detail。
    """
    min_files_gate, matched = _check_min_files(workspace, spec.expected.min_files)
    gates = [
        min_files_gate,
        _check_forbidden_patterns(workspace, spec.expected.forbidden_patterns),
        _check_must_pass_tests(
            workspace,
            spec.expected.must_pass_tests,
            timeout_per_command=timeout_per_command,
            run_tests=run_tests,
            matched_min_files=matched,
        ),
        _check_budget(
            wall_seconds=wall_seconds,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            budget_max_wall=spec.budget.max_wall_seconds,
            budget_max_input=spec.budget.max_input_tokens,
            budget_max_output=spec.budget.max_output_tokens,
        ),
    ]
    return PassResult(passed=all(g.passed for g in gates), gates=gates)


# ──────────────────────────── Conflict accounting ───────────────────────────


@dataclass
class ConflictAudit:
    """Compare runtime-observed conflicts against the prompt's oracle."""

    resource_conflict_count: int = 0
    false_positive_conflicts: int = 0
    false_negative_conflicts: int = 0
    corruption_detected: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "resource_conflict_count": self.resource_conflict_count,
            "false_positive_conflicts": self.false_positive_conflicts,
            "false_negative_conflicts": self.false_negative_conflicts,
            "corruption_detected": self.corruption_detected,
        }


def audit_conflicts(
    spec: TaskSpec,
    *,
    observed_conflict_writes: Iterable[Iterable[str]] = (),
    observed_workspace_writes: Iterable[str] = (),
    corruption_count: int = 0,
) -> ConflictAudit:
    """Cross-check conflict events against the ground-truth write set.

    - ``observed_conflict_writes`` is one set of paths per `resource.conflict`
      event (the union of ``attempted`` ∪ ``running`` writes). A conflict
      counted as **false positive** if no path it touches falls inside the
      prompt's declared write set — i.e. the scheduler reported a conflict
      on a path the oracle says nothing actually writes.
    - ``observed_workspace_writes`` is the set of paths actually written
      during the run (read off ``task.submitted.resource_access.writes``).
      A path declared in oracle but missing from observed writes is a
      candidate **false negative** — either the prompt didn't reach that
      step or the scheduler suppressed a conflict that should have fired.
    - ``corruption_count`` flips ``corruption_detected`` if it's >0; this
      lets callers reuse the corruption detector here.

    When the prompt ships with no ground-truth, both FP and FN counts are
    zeroed (we can't audit what we don't have).
    """
    declared = spec.declared_writes
    observed_conflict_writes_list = [
        set(s) for s in observed_conflict_writes
    ]
    observed_writes = set(observed_workspace_writes)

    conflict_count = len(observed_conflict_writes_list)
    if not declared:
        return ConflictAudit(
            resource_conflict_count=conflict_count,
            false_positive_conflicts=0,
            false_negative_conflicts=0,
            corruption_detected=corruption_count > 0,
        )

    fp = 0
    for paths in observed_conflict_writes_list:
        if not (paths & declared):
            fp += 1

    # An oracle write is FN if it was actually written during the run but
    # never showed up in any conflict event (i.e. a write/write situation
    # the scheduler should have arbitrated, but didn't). Paths the run
    # didn't touch at all aren't false negatives — they're just unreached.
    flat_conflict_writes: set[str] = set()
    for paths in observed_conflict_writes_list:
        flat_conflict_writes.update(paths)
    fn = 0
    for path in declared:
        if path in observed_writes and path not in flat_conflict_writes:
            fn += 1

    return ConflictAudit(
        resource_conflict_count=conflict_count,
        false_positive_conflicts=fp,
        false_negative_conflicts=fn,
        corruption_detected=corruption_count > 0,
    )


# ──────────────────────────── Cell record + report ──────────────────────────


@dataclass
class CellRecord:
    """One (prompt, system, run) folded down to a flat record.

    Adapter-agnostic shape; ``cell_record_from_artifacts`` shows how to
    populate it from a / / run dir, but third-party
    adapters can build the same shape directly.
    """

    prompt_id: str
    system: str
    run: int
    passed: bool = False
    wall_seconds: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    resource_conflict_count: int = 0
    false_positive_conflicts: int = 0
    false_negative_conflicts: int = 0
    corruption_count: int = 0
    gates: list[GateResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "system": self.system,
            "run": self.run,
            "passed": self.passed,
            "wall_seconds": self.wall_seconds,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "resource_conflict_count": self.resource_conflict_count,
            "false_positive_conflicts": self.false_positive_conflicts,
            "false_negative_conflicts": self.false_negative_conflicts,
            "corruption_count": self.corruption_count,
            "corruption_detected": self.corruption_count > 0,
            "gates": [
                {"name": g.name, "passed": g.passed, "detail": g.detail}
                for g in self.gates
            ],
        }


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated quantile on a sorted float list."""
    if not values:
        return 0.0
    arr = sorted(values)
    if len(arr) == 1:
        return float(arr[0])
    pos = q * (len(arr) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(arr[lo])
    frac = pos - lo
    return float(arr[lo] + (arr[hi] - arr[lo]) * frac)


def _summarise_per_prompt(records: Sequence[CellRecord]) -> dict[str, Any]:
    """Fold N runs of one prompt into the report shape."""
    if not records:
        return {}
    # A prompt is counted as passing if at least one of its runs passed —
    # this matches's per-prompt boolean field. p50s are computed on
    # the passing subset only (so a single broken run can't drag medians).
    passing = [r for r in records if r.passed]
    walls = [r.wall_seconds for r in passing if r.wall_seconds is not None]
    in_tokens = [r.input_tokens for r in passing if r.input_tokens is not None]
    out_tokens = [r.output_tokens for r in passing if r.output_tokens is not None]
    return {
        "passed": bool(passing),
        "n_runs": len(records),
        "n_passed": len(passing),
        "wall_seconds_p50": _percentile([float(x) for x in walls], 0.5),
        "input_tokens_p50": _percentile([float(x) for x in in_tokens], 0.5),
        "output_tokens_p50": _percentile([float(x) for x in out_tokens], 0.5),
        "resource_conflict_count": sum(r.resource_conflict_count for r in records),
        "false_positive_conflicts": sum(r.false_positive_conflicts for r in records),
        "false_negative_conflicts": sum(r.false_negative_conflicts for r in records),
        "corruption_detected": any(r.corruption_count > 0 for r in records),
    }


def _geomean(values: Iterable[float]) -> float:
    """Geometric mean; returns 0.0 on empty input."""
    arr = [v for v in values if v is not None and v > 0]
    if not arr:
        return 0.0
    log_sum = sum(math.log(v) for v in arr)
    return math.exp(log_sum / len(arr))


def _scheduler_score(
    candidate: dict[str, dict[str, Any]],
    baseline: dict[str, dict[str, Any]] | None,
) -> dict[str, float]:
    """Geomean of candidate / baseline ratios on (wall, in_tokens, out_tokens).

    The score is **only** computed on prompts where both candidate and
    baseline pass — requires «scheduler_score and pass_rate are
    reported separately, never multiplied». Prompts the candidate fails
    don't get to push the score down through artificially large medians.
    """
    if not baseline:
        # No baseline → score is 1.0 by convention (the candidate is its
        # own baseline). Useful when scoring a single system in isolation.
        return {
            "scheduler_score": 1.0,
            "wall_seconds_ratio_geomean": 1.0,
            "input_tokens_ratio_geomean": 1.0,
            "output_tokens_ratio_geomean": 1.0,
            "n_compared_prompts": sum(
                1 for v in candidate.values() if v.get("passed")
            ),
        }
    walls: list[float] = []
    ins: list[float] = []
    outs: list[float] = []
    for pid, cand_summary in candidate.items():
        if not cand_summary.get("passed"):
            continue
        base_summary = baseline.get(pid)
        if not base_summary or not base_summary.get("passed"):
            continue
        cw = cand_summary.get("wall_seconds_p50")
        bw = base_summary.get("wall_seconds_p50")
        if cw and bw:
            walls.append(cw / bw)
        ci = cand_summary.get("input_tokens_p50")
        bi = base_summary.get("input_tokens_p50")
        if ci and bi:
            ins.append(ci / bi)
        co = cand_summary.get("output_tokens_p50")
        bo = base_summary.get("output_tokens_p50")
        if co and bo:
            outs.append(co / bo)

    wall_g = _geomean(walls)
    in_g = _geomean(ins)
    out_g = _geomean(outs)
    factors = [v for v in (wall_g, in_g, out_g) if v > 0]
    if not factors:
        score = 1.0
    else:
        score = math.exp(sum(math.log(v) for v in factors) / len(factors))
    return {
        "scheduler_score": round(score, 6),
        "wall_seconds_ratio_geomean": round(wall_g, 6) if wall_g else 0.0,
        "input_tokens_ratio_geomean": round(in_g, 6) if in_g else 0.0,
        "output_tokens_ratio_geomean": round(out_g, 6) if out_g else 0.0,
        "n_compared_prompts": len(walls),
    }


def build_report(
    *,
    system: str,
    records: Sequence[CellRecord],
    baseline_per_prompt: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fold a flat list of CellRecords into the shape."""
    by_prompt: dict[str, list[CellRecord]] = {}
    for rec in records:
        by_prompt.setdefault(rec.prompt_id, []).append(rec)

    prompts: dict[str, Any] = {}
    for pid, rs in sorted(by_prompt.items()):
        prompts[pid] = _summarise_per_prompt(rs)

    n_prompts = len(prompts)
    n_passed = sum(1 for v in prompts.values() if v.get("passed"))
    pass_rate = (n_passed / n_prompts) if n_prompts else 0.0
    walls = [
        v["wall_seconds_p50"]
        for v in prompts.values()
        if v.get("passed") and v.get("wall_seconds_p50")
    ]
    geomean_wall = _geomean(walls)
    score_block = _scheduler_score(prompts, baseline_per_prompt)
    return {
        "system": system,
        "prompts": prompts,
        "aggregate": {
            "n_prompts": n_prompts,
            "n_passed": n_passed,
            "pass_rate": round(pass_rate, 6),
            "geomean_wall_seconds": round(geomean_wall, 6),
            **score_block,
        },
    }


# ──────────────────────── Loading// cells ──────────────────────


def cell_record_from_artifacts(
    spec: TaskSpec,
    run_dir: Path,
    *,
    system: str,
    run_tests: bool = False,
) -> CellRecord:
    """Build a CellRecord from a // run directory.

    Reads ``run_meta.json`` for usage + wall_seconds, the workspace
    sub-directory for gate evaluation, and ``trace.jsonl`` for conflict
    + corruption accounting.
    """
    meta_path = run_dir / "run_meta.json"
    trace_path = run_dir / "trace.jsonl"
    workspace = run_dir / "workspace"

    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    usage = meta.get("usage") or {}
    wall = meta.get("wall_seconds")
    in_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
    out_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None

    pass_result = evaluate_pass(
        spec,
        workspace=workspace,
        wall_seconds=float(wall) if isinstance(wall, (int, float)) else None,
        input_tokens=int(in_tokens) if isinstance(in_tokens, int) else None,
        output_tokens=int(out_tokens) if isinstance(out_tokens, int) else None,
        run_tests=run_tests,
    )

    conflict_writes, observed_writes = _conflict_signals(trace_path)
    corruption_count = _corruption_count_from_dir(run_dir)
    audit = audit_conflicts(
        spec,
        observed_conflict_writes=conflict_writes,
        observed_workspace_writes=observed_writes,
        corruption_count=corruption_count,
    )

    return CellRecord(
        prompt_id=spec.id,
        system=system,
        run=int(meta.get("run") or 0),
        passed=pass_result.passed,
        wall_seconds=float(wall) if isinstance(wall, (int, float)) else None,
        input_tokens=int(in_tokens) if isinstance(in_tokens, int) else None,
        output_tokens=int(out_tokens) if isinstance(out_tokens, int) else None,
        resource_conflict_count=audit.resource_conflict_count,
        false_positive_conflicts=audit.false_positive_conflicts,
        false_negative_conflicts=audit.false_negative_conflicts,
        corruption_count=corruption_count,
        gates=pass_result.gates,
    )


def _conflict_signals(
    trace_path: Path,
) -> tuple[list[set[str]], set[str]]:
    if not trace_path.exists():
        return [], set()
    conflict_writes: list[set[str]] = []
    observed_writes: set[str] = set()
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        name = ev.get("event")
        if name == "task.submitted":
            access = ev.get("resource_access") or {}
            for w in access.get("writes") or []:
                observed_writes.add(str(w))
        elif name == "resource.conflict":
            paths: set[str] = set()
            attempted = ev.get("attempted_access") or {}
            running = ev.get("running_access") or {}
            for w in attempted.get("writes") or []:
                paths.add(str(w))
            for w in running.get("writes") or []:
                paths.add(str(w))
            conflict_writes.append(paths)
    return conflict_writes, observed_writes


def _corruption_count_from_dir(run_dir: Path) -> int:
    """Read corruption.json if present adapters drop it)."""
    p = run_dir / "corruption.json"
    if not p.exists():
        return 0
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0
    return int(doc.get("corruption_count") or 0)


def discover_cells(
    root: Path,
    *,
    prefixes: Sequence[str] = ("T1d_", "T2a_", "T2b_"),
) -> list[Path]:
    """Walk a results root and yield every per-cell run dir."""
    if not root.exists():
        return []
    out: list[Path] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if not any(p.name.startswith(prefix) for prefix in prefixes):
            continue
        if (p / "run_meta.json").exists():
            out.append(p)
    return out


def _glob_match(name: str, patterns: Sequence[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, pat) for pat in patterns)


def load_records_from_root(
    root: Path,
    specs: Sequence[TaskSpec],
    *,
    system: str,
    run_dir_glob: Sequence[str] | None = None,
    run_tests: bool = False,
) -> list[CellRecord]:
    """Load every cell under ``root`` whose prompt id matches a TaskSpec."""
    spec_by_id: dict[str, TaskSpec] = {s.id: s for s in specs}
    out: list[CellRecord] = []
    for cell_dir in discover_cells(root):
        if run_dir_glob and not _glob_match(cell_dir.name, run_dir_glob):
            continue
        meta = cell_dir / "run_meta.json"
        if not meta.exists():
            continue
        try:
            meta_doc = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        prompt_id = str(meta_doc.get("prompt_id") or "")
        spec = spec_by_id.get(prompt_id)
        if spec is None:
            continue
        out.append(
            cell_record_from_artifacts(
                spec, cell_dir, system=system, run_tests=run_tests
            )
        )
    return out


# Re-export for convenience at the package level.
__all__ = [
    "CellRecord",
    "ConflictAudit",
    "GateResult",
    "PassResult",
    "audit_conflicts",
    "build_report",
    "cell_record_from_artifacts",
    "discover_cells",
    "evaluate_pass",
    "load_records_from_root",
]
