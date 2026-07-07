from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.tools.core import _infer_readonly_terminal_access
from high_agent.runtime.resource_access import access_conflicts


class TerminalResourceAccessInferenceTests(unittest.TestCase):
    def _infer(self, command: str, *, workdir: str = "."):
        return _infer_readonly_terminal_access({"command": command, "workdir": workdir}, root=None)

    def test_awk_with_explicit_path_yields_file_read(self) -> None:
        access = self._infer("awk '{print $1}' file.txt")
        self.assertIsNotNone(access)
        self.assertTrue(any(item.startswith("file:") and item.endswith("file.txt") for item in access.reads))
        self.assertEqual(access.writes, frozenset())
        self.assertFalse(access.unknown)

    def test_unsafe_redirect_falls_back_to_unknown(self) -> None:
        access = self._infer("awk '{print}' file > out.txt")
        self.assertIsNone(access)

    def test_pipe_falls_back_to_unknown(self) -> None:
        access = self._infer("grep pattern file | tee output.txt")
        self.assertIsNone(access)

    def test_pipe_readonly_yields_read_access(self) -> None:
        access = self._infer("grep pattern file.py | wc -l")
        self.assertIsNotNone(access)
        self.assertTrue(any(item.endswith("file.py") for item in access.reads))
        self.assertEqual(access.writes, frozenset())
        self.assertFalse(access.unknown)

    def test_pipe_with_sed_i_falls_back(self) -> None:
        access = self._infer("cat f.txt | sed -i 's/a/b/' f.txt")
        self.assertIsNone(access)

    def test_pipe_with_unsafe_segment_falls_back(self) -> None:
        access = self._infer("grep x file.py | rm -rf /")
        self.assertIsNone(access)

    def test_pipe_multi_stage_readonly(self) -> None:
        access = self._infer("cat src/main.py | grep def | sort | uniq")
        self.assertIsNotNone(access)
        self.assertTrue(any(item.endswith("src/main.py") for item in access.reads))
        self.assertFalse(access.unknown)

    def test_jq_single_command_readonly(self) -> None:
        access = self._infer("jq '.key' data.json")
        self.assertIsNotNone(access)
        self.assertTrue(any(item.endswith("data.json") for item in access.reads))
        self.assertFalse(access.unknown)

    def test_two_pipe_commands_on_different_files_parallelize(self) -> None:
        access_a = self._infer("grep foo a.py | wc -l")
        access_b = self._infer("grep bar b.py | wc -l")
        self.assertIsNotNone(access_a)
        self.assertIsNotNone(access_b)
        self.assertFalse(access_conflicts(access_a, access_b))

    def test_sed_inplace_is_not_inferred(self) -> None:
        access = self._infer("sed -i 's/a/b/g' file.txt")
        self.assertIsNone(access)

    def test_git_show_path_is_extracted(self) -> None:
        access = self._infer("git show HEAD:src/main.py")
        self.assertIsNotNone(access)
        self.assertTrue(any(item.endswith("src/main.py") for item in access.reads))

    def test_git_diff_two_paths(self) -> None:
        access = self._infer("git diff src/a.py src/b.py")
        self.assertIsNotNone(access)
        path_components = [item for item in access.reads if item.startswith("file:")]
        rendered = " ".join(path_components)
        self.assertIn("src/a.py", rendered)
        self.assertIn("src/b.py", rendered)

    def test_python_module_only_for_known_safe_modules(self) -> None:
        safe = self._infer("python -m json.tool data.json")
        self.assertIsNotNone(safe)
        self.assertTrue(any(item.endswith("data.json") for item in safe.reads))

        unsafe = self._infer("python script.py")
        self.assertIsNone(unsafe)

    def test_two_independent_awk_can_run_in_parallel(self) -> None:
        access_a = self._infer("awk '{print}' a.txt")
        access_b = self._infer("awk '{print}' b.txt")
        self.assertIsNotNone(access_a)
        self.assertIsNotNone(access_b)
        self.assertFalse(access_conflicts(access_a, access_b))


class NetworkCommandClassificationTests(unittest.TestCase):
    """v8-P4: curl/wget/ping/dig/... should be external_read/write, not unknown."""

    def _infer(self, command: str):
        return _infer_readonly_terminal_access({"command": command, "workdir": "."}, root=None)

    def test_curl_default_is_external_read(self) -> None:
        access = self._infer("curl https://example.com/api")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_read")
        self.assertFalse(access.unknown)

    def test_curl_with_data_is_external_write(self) -> None:
        access = self._infer("curl -d 'hello' https://example.com/post")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")

    def test_curl_explicit_get_is_external_read(self) -> None:
        access = self._infer("curl -X GET https://example.com/r")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_read")

    def test_curl_explicit_post_is_external_write(self) -> None:
        access = self._infer("curl -X POST https://example.com/r")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")

    def test_curl_upload_is_external_write(self) -> None:
        access = self._infer("curl -T file.txt https://example.com/upload")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")

    def test_wget_default_is_external_read(self) -> None:
        access = self._infer("wget https://example.com/file.tar")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_read")

    def test_wget_post_is_external_write(self) -> None:
        access = self._infer("wget --post-data='k=v' https://example.com/api")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")

    def test_ping_dig_nslookup_are_external_read(self) -> None:
        for cmd in ("ping example.com", "dig example.com", "nslookup example.com", "host example.com"):
            access = self._infer(cmd)
            self.assertIsNotNone(access, cmd)
            self.assertEqual(access.side_effect_level, "external_read", cmd)

    def test_two_curl_get_can_parallelize(self) -> None:
        a = self._infer("curl https://a.example.com/")
        b = self._infer("curl https://b.example.com/")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertFalse(access_conflicts(a, b))

    def test_curl_post_serializes_with_curl_get(self) -> None:
        a = self._infer("curl -X POST https://a.example.com/")
        b = self._infer("curl https://b.example.com/")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertTrue(access_conflicts(a, b))

    # --request must be honoured the same as -X.
    def test_curl_long_request_post_is_external_write(self) -> None:
        access = self._infer("curl --request POST https://example.com/r")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")

    def test_curl_long_request_equals_delete_is_external_write(self) -> None:
        access = self._infer("curl --request=DELETE https://example.com/r")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")

    def test_curl_long_request_get_is_external_read(self) -> None:
        access = self._infer("curl --request GET https://example.com/r")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_read")


class GitGlobalOptionTests(unittest.TestCase):
    """ git -C / -c / --git-dir must not be mistaken for the subcommand."""

    def _infer(self, command: str):
        return _infer_readonly_terminal_access({"command": command, "workdir": "."}, root=None)

    def test_git_dash_C_status_is_recognised_as_readonly(self) -> None:
        access = self._infer("git -C /repo status")
        self.assertIsNotNone(access)
        self.assertEqual(access.writes, frozenset())

    def test_git_dash_c_config_then_log(self) -> None:
        access = self._infer("git -c core.pager=cat log -1")
        self.assertIsNotNone(access)
        self.assertEqual(access.writes, frozenset())

    def test_git_dash_C_diff_two_paths(self) -> None:
        access = self._infer("git -C /repo diff a.py b.py")
        self.assertIsNotNone(access)
        # path extraction still works after global option strip
        paths = {entry.split(":", 1)[1] for entry in access.reads if ":" in entry}
        self.assertTrue(any(p.endswith("a.py") for p in paths))


class MkdirAncestorTests(unittest.TestCase):
    """ mkdir -p declares writes for every ancestor it would create."""

    def _scoped(self, command: str):
        from high_agent.tools.core import _infer_scoped_mutation_access
        return _infer_scoped_mutation_access({"command": command, "workdir": "."}, root="/repo")

    def test_mkdir_p_includes_ancestors(self) -> None:
        access = self._scoped("mkdir -p a/b/c")
        self.assertIsNotNone(access)
        write_paths = {entry.split(":", 1)[1] for entry in access.writes}
        self.assertIn("/repo/a/b/c", write_paths)
        self.assertIn("/repo/a/b", write_paths)
        self.assertIn("/repo/a", write_paths)

    def test_mkdir_without_p_only_declares_leaf(self) -> None:
        access = self._scoped("mkdir solo")
        self.assertIsNotNone(access)
        write_paths = {entry.split(":", 1)[1] for entry in access.writes}
        self.assertEqual(write_paths, {"/repo/solo"})


class BuildCommandClassificationTests(unittest.TestCase):
    """feat-S2: npm/yarn/pnpm/make should land on scoped reads/writes/interactive
    rather than unknown_workspace, so common dev workflows can parallelise."""

    def _scoped(self, command: str, *, workdir: str = "."):
        from high_agent.tools.core import _infer_scoped_mutation_access
        return _infer_scoped_mutation_access({"command": command, "workdir": workdir}, root="/repo")

    def test_npm_test_is_readonly(self) -> None:
        for cmd in ("npm test", "npm run test", "yarn test", "pnpm test", "bun test"):
            access = self._scoped(cmd)
            self.assertIsNotNone(access, cmd)
            self.assertEqual(access.writes, frozenset(), cmd)
            self.assertFalse(access.unknown, cmd)

    def test_npm_lint_is_readonly(self) -> None:
        for cmd in ("npm run lint", "yarn run lint", "pnpm run lint"):
            access = self._scoped(cmd)
            self.assertIsNotNone(access, cmd)
            self.assertEqual(access.writes, frozenset(), cmd)

    def test_npm_run_build_writes_output_dirs(self) -> None:
        access = self._scoped("npm run build")
        self.assertIsNotNone(access)
        write_paths = {entry.split(":", 1)[1] for entry in access.writes if entry.startswith("dir:")}
        self.assertIn("/repo/dist", write_paths)
        self.assertIn("/repo/build", write_paths)
        self.assertIn("/repo/.next", write_paths)

    def test_npm_run_dev_is_interactive(self) -> None:
        for cmd in ("npm run dev", "npm run start", "yarn run watch"):
            access = self._scoped(cmd)
            self.assertIsNotNone(access, cmd)
            self.assertEqual(access.side_effect_level, "interactive", cmd)

    def test_make_clean_writes_build_artefacts(self) -> None:
        access = self._scoped("make clean")
        self.assertIsNotNone(access)
        write_paths = {entry.split(":", 1)[1] for entry in access.writes}
        self.assertIn("/repo/build", write_paths)
        self.assertIn("/repo/dist", write_paths)
        self.assertIn("/repo/target", write_paths)

    def test_make_test_is_readonly(self) -> None:
        for cmd in ("make test", "make check", "make lint"):
            access = self._scoped(cmd)
            self.assertIsNotNone(access, cmd)
            self.assertEqual(access.writes, frozenset(), cmd)

    def test_bare_make_falls_through(self) -> None:
        self.assertIsNone(self._scoped("make"))

    def test_unknown_npm_run_target_falls_through(self) -> None:
        self.assertIsNone(self._scoped("npm run my-custom-target"))
        self.assertIsNone(self._scoped("npm run frobnicate"))

    def test_npm_install_branch_unchanged(self) -> None:
        for cmd in ("npm install", "yarn install", "pnpm i", "bun add foo"):
            access = self._scoped(cmd)
            self.assertIsNotNone(access, cmd)
            self.assertNotEqual(access.writes, frozenset(), cmd)

    def test_compound_command_decomposed_via_composite(self) -> None:
        # Composite commands no longer fall through to unknown.
        # `_infer_scoped_mutation_access` (called directly here) still
        # returns None on composites; the composite path lives at the
        # `_process_resource_access` level via `_infer_composite_terminal_access`.
        self.assertIsNone(self._scoped("npm test && npm run build"))
        self.assertIsNone(self._scoped("npm test; npm run lint"))

    def test_npm_test_and_npm_lint_can_parallelize(self) -> None:
        a = self._scoped("npm test")
        b = self._scoped("npm run lint")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertFalse(access_conflicts(a, b))

    def test_npm_install_conflicts_with_node_modules_read(self) -> None:
        from high_agent.tools.core import _infer_scoped_mutation_access
        from high_agent.runtime.resource_access import ResourceAccess, normalize_component
        install = _infer_scoped_mutation_access({"command": "npm install", "workdir": "."}, root="/repo")
        reader = ResourceAccess(reads=frozenset({normalize_component("dir:./node_modules", "/repo")}))
        self.assertIsNotNone(install)
        self.assertTrue(access_conflicts(install, reader))


class DockerKubectlClassificationTests(unittest.TestCase):
    """ docker/kubectl scoped mutation classification."""

    def _infer(self, command: str):
        from high_agent.tools.core import _infer_scoped_mutation_access
        return _infer_scoped_mutation_access({"command": command, "workdir": "."}, root="/repo")

    def test_docker_ps_is_external_read(self) -> None:
        access = self._infer("docker ps")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_read")
        self.assertIn("external:docker", access.reads)

    def test_docker_logs_is_external_read(self) -> None:
        access = self._infer("docker logs my-container")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_read")

    def test_docker_build_is_external_write(self) -> None:
        access = self._infer("docker build -t myimg .")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")
        self.assertIn("external:docker:image", access.writes)

    def test_docker_push_is_external_write(self) -> None:
        access = self._infer("docker push myimg:latest")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")

    def test_docker_run_is_external_write(self) -> None:
        access = self._infer("docker run --rm myimg")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")

    def test_kubectl_get_is_external_read(self) -> None:
        access = self._infer("kubectl get pods")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_read")
        self.assertIn("external:kubectl", access.reads)

    def test_kubectl_describe_is_external_read(self) -> None:
        access = self._infer("kubectl describe ns/default")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_read")

    def test_kubectl_apply_is_external_write(self) -> None:
        access = self._infer("kubectl apply -f manifest.yaml")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")

    def test_kubectl_delete_is_external_write(self) -> None:
        access = self._infer("kubectl delete pod my-pod")
        self.assertIsNotNone(access)
        self.assertEqual(access.side_effect_level, "external_write")

    def test_two_docker_reads_can_parallelize(self) -> None:
        a = self._infer("docker ps")
        b = self._infer("docker logs c1")
        self.assertIsNotNone(a)
        self.assertIsNotNone(b)
        self.assertFalse(access_conflicts(a, b))

    def test_docker_write_serialises_with_read(self) -> None:
        write = self._infer("docker run myimg")
        read = self._infer("docker ps")
        self.assertIsNotNone(write)
        self.assertIsNotNone(read)
        self.assertTrue(access_conflicts(write, read))

    def test_docker_ps_and_kubectl_get_can_parallelize(self) -> None:
        # Different external resources don't serialise.
        d = self._infer("docker ps")
        k = self._infer("kubectl get pods")
        self.assertIsNotNone(d)
        self.assertIsNotNone(k)
        self.assertFalse(access_conflicts(d, k))

    def test_unknown_docker_subcommand_falls_through(self) -> None:
        # Unknown subcommand falls through to caller (returns None,
        # caller typically falls back to unknown_workspace).
        self.assertIsNone(self._infer("docker frobnicate"))

    def test_unknown_kubectl_verb_falls_through(self) -> None:
        self.assertIsNone(self._infer("kubectl frobnicate pods"))


class CompositeShellDecompositionTests(unittest.TestCase):
    """ && / || / ; decomposition into per-segment access union."""

    def _process(self, command: str, *, workdir: str = "."):
        from high_agent.tools.core import _process_resource_access
        return _process_resource_access(
            {"command": command, "workdir": workdir}, root="/repo", kind="terminal"
        )

    def test_mkdir_then_touch_unions_writes(self) -> None:
        access = self._process("mkdir a && touch a/x")
        self.assertFalse(access.unknown)
        # Both writes present in the union.
        self.assertTrue(any(item.endswith("/a") and item.startswith("dir:") for item in access.writes))
        self.assertTrue(any(item.endswith("/a/x") and item.startswith("file:") for item in access.writes))

    def test_two_readonly_segments_yield_readonly(self) -> None:
        access = self._process("ls; pwd")
        self.assertFalse(access.unknown)
        self.assertEqual(access.writes, frozenset())

    def test_git_status_and_log_readonly(self) -> None:
        access = self._process("git status && git log --oneline")
        self.assertFalse(access.unknown)
        self.assertEqual(access.writes, frozenset())

    def test_npm_build_and_test_unions_writes(self) -> None:
        access = self._process("npm run build && npm test")
        self.assertFalse(access.unknown)
        # build creates dist/build outputs.
        self.assertTrue(any("dist" in item or "build" in item for item in access.writes))

    def test_rm_then_mkdir_unions(self) -> None:
        access = self._process("rm -rf x; mkdir x")
        self.assertFalse(access.unknown)
        self.assertTrue(any(item.endswith("/x") for item in access.writes))

    def test_composite_with_unknown_segment_falls_back(self) -> None:
        # `cd x && npm install` — `cd` is not classified, decomposition fails
        # → caller falls back to unknown_workspace.
        access = self._process("cd x && npm install")
        self.assertTrue(access.unknown)

    def test_composite_with_redirect_in_segment_falls_back(self) -> None:
        access = self._process("ls && echo done > out.txt")
        self.assertTrue(access.unknown)

    def test_or_separator_decomposed(self) -> None:
        access = self._process("ls || pwd")
        self.assertFalse(access.unknown)
        self.assertEqual(access.writes, frozenset())

    def test_quoted_separator_not_split(self) -> None:
        # Quote-aware splitter: a `;` inside quotes does NOT count as a top-level
        # composite separator. We test the splitter directly because
        # _has_unsafe_shell_syntax (called downstream) is intentionally not
        # quote-aware — so the full pipeline still falls back to unknown for
        # commands with quoted shell metas. Splitter-level correctness is
        # required so e.g. `mkdir 'a;b'; touch x` doesn't get split inside the
        # quoted dirname.
        from high_agent.tools.core import _split_composite_command
        # No top-level separator → splitter returns None.
        self.assertIsNone(_split_composite_command("git log --grep='a;b'"))
        # Top-level `;` after the quoted segment IS a separator.
        segments = _split_composite_command("mkdir 'a;b'; touch x")
        self.assertEqual(segments, ["mkdir 'a;b'", "touch x"])

    def test_background_marker_still_unsafe(self) -> None:
        # Bare `&` (background) remains unsafe even after.
        access = self._process("ls & pwd")
        self.assertTrue(access.unknown)


class LocalScriptInvocationTests(unittest.TestCase):
    """ classify ./script.sh + bash script.sh + bare python script.py."""

    def _process(self, command: str, *, workdir: str = "."):
        from high_agent.tools.core import _process_resource_access
        return _process_resource_access(
            {"command": command, "workdir": workdir}, root="/repo", kind="terminal"
        )

    def test_relative_shell_script_classifies_as_scoped(self) -> None:
        access = self._process("./build.sh")
        self.assertFalse(access.unknown)
        self.assertTrue(any(item.endswith("build.sh") and item.startswith("file:") for item in access.reads))
        self.assertTrue(any(item.startswith("dir:") for item in access.writes))
        self.assertEqual(access.side_effect_level, "local")

    def test_absolute_shell_script_classifies_as_scoped(self) -> None:
        access = self._process("/usr/local/bin/deploy.sh")
        self.assertFalse(access.unknown)
        self.assertTrue(any("deploy.sh" in item and item.startswith("file:") for item in access.reads))

    def test_bash_script_classifies_as_scoped(self) -> None:
        access = self._process("bash ci/test.sh")
        self.assertFalse(access.unknown)
        self.assertTrue(any("test.sh" in item and item.startswith("file:") for item in access.reads))
        self.assertTrue(any(item.startswith("dir:") for item in access.writes))

    def test_bash_dash_c_falls_back_to_unknown(self) -> None:
        # `bash -c "..."` body is opaque; the heuristic must defer to unknown.
        access = self._process("bash -c 'echo hi'")
        self.assertTrue(access.unknown)

    def test_bare_python_script_classifies_as_scoped(self) -> None:
        access = self._process("python deploy.py")
        self.assertFalse(access.unknown)
        self.assertTrue(any(item.endswith("deploy.py") and item.startswith("file:") for item in access.reads))
        self.assertTrue(any(item.startswith("dir:") for item in access.writes))

    def test_python3_script_classifies_as_scoped(self) -> None:
        access = self._process("python3 tools/run.py")
        self.assertFalse(access.unknown)
        self.assertTrue(any("run.py" in item and item.startswith("file:") for item in access.reads))

    def test_python_dash_c_falls_back_to_unknown(self) -> None:
        access = self._process("python -c 'print(1)'")
        self.assertTrue(access.unknown)

    def test_python_dash_m_unknown_target_falls_back(self) -> None:
        # Non-allowlisted -m targets defer to unknown (could mutate).
        access = self._process("python -m mypackage.main")
        self.assertTrue(access.unknown)

    def test_python_dash_m_json_tool_remains_readonly(self) -> None:
        # Allowlisted readonly -m targets are classified upstream as readonly.
        access = self._process("python -m json.tool data.json")
        self.assertFalse(access.unknown)
        self.assertEqual(access.writes, frozenset())

    def test_two_scripts_in_same_dir_serialise(self) -> None:
        # Both scripts write `dir:.` so they must conflict.
        a = self._process("./a.sh")
        b = self._process("./b.sh")
        self.assertTrue(access_conflicts(a, b))

    def test_script_and_unrelated_readonly_can_parallelise(self) -> None:
        # `git status` is readonly, the script writes dir:.; they overlap on
        # `dir:.` (read vs write) so they must serialise — verify the model
        # is conservative-but-correct rather than wrongly parallel.
        script = self._process("./build.sh")
        git = self._process("git status")
        # script writes dir:., git reads dir:.; access_conflicts must return True.
        self.assertTrue(access_conflicts(script, git))


if __name__ == "__main__":
    unittest.main()
