from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.runtime import CausalRuntime, DependencyPredicate
from high_agent.tools import ToolsetRegistry, create_core_registry


class ToolAndAgentTests(unittest.TestCase):
    def test_core_registry_requires_and_emits_resource_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = create_core_registry()
            path = Path(tmp) / "a.txt"
            task = registry.task_from_call(
                "write_file",
                {"path": str(path), "content": "hello"},
                workspace_root=tmp,
                task_id="write-a",
            )
            self.assertEqual(task.task_id, "write-a")
            self.assertIn(f"file:{path}", task.writes)

            rel_task = registry.task_from_call(
                "write_file",
                {"path": "rel.txt", "content": "hello"},
                workspace_root=tmp,
                task_id="write-rel",
            )
            self.assertIn(f"file:{Path(tmp) / 'rel.txt'}", rel_task.writes)

    def test_delegate_task_is_registered_in_core_registry(self) -> None:
        registry = create_core_registry()
        self.assertIn("delegate_task", registry.names())

    def test_terminal_is_blocked_without_yes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = create_core_registry()
            runtime = CausalRuntime(workspace_root=tmp, delivery_debounce=0.0)
            self.addCleanup(runtime.shutdown)
            task = registry.task_from_call("terminal", {"command": "echo hi"}, workspace_root=tmp, task_id="term")
            runtime.submit([task])
            self.assertTrue(runtime.wait_all(timeout=1))
            result = runtime.collect({"term"})["term"]
            self.assertEqual(result.status, "blocked")

    def test_extended_file_tools_edit_search_move_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = create_core_registry()
            runtime = CausalRuntime(workspace_root=tmp, delivery_debounce=0.0)
            self.addCleanup(runtime.shutdown)
            tasks = [
                registry.task_from_call("write_file", {"path": "a.txt", "content": "alpha\n"}, workspace_root=tmp, task_id="write"),
                registry.task_from_call("append_file", {"path": "a.txt", "content": "beta\n"}, workspace_root=tmp, task_id="append"),
                registry.task_from_call("replace_in_file", {"path": "a.txt", "old": "beta", "new": "gamma"}, workspace_root=tmp, task_id="replace"),
                registry.task_from_call("search_files", {"pattern": "gamma", "path": ".", "max_results": 5}, workspace_root=tmp, task_id="search"),
                registry.task_from_call("move_path", {"src": "a.txt", "dst": "b.txt"}, workspace_root=tmp, task_id="move"),
                registry.task_from_call("delete_path", {"path": "b.txt"}, workspace_root=tmp, task_id="delete"),
            ]
            for before, after in zip(tasks, tasks[1:]):
                after.dependencies.append(DependencyPredicate.task_completed(before.task_id))
            runtime.submit(tasks)
            self.assertTrue(runtime.wait_all(timeout=2))
            self.assertFalse((Path(tmp) / "a.txt").exists())
            self.assertFalse((Path(tmp) / "b.txt").exists())
            self.assertIn("gamma", runtime.collect({"search"})["search"].summary)


if __name__ == "__main__":
    unittest.main()
