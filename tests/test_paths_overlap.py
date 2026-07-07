from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from high_agent.runtime.resource_access import (
    ResourceAccess,
    access_conflicts,
    resources_overlap,
)


class PathsOverlapTests(unittest.TestCase):
    def test_sibling_files_do_not_overlap(self) -> None:
        self.assertFalse(resources_overlap("file:/a/b.txt", "file:/a/c.txt"))

    def test_identical_files_overlap(self) -> None:
        self.assertTrue(resources_overlap("file:/a/b.txt", "file:/a/b.txt"))

    def test_distinct_files_with_prefix_relation_do_not_overlap(self) -> None:
        # file:/a/b 与 file:/a/b/c 是两个不同的文件路径，不应被视为冲突
        self.assertFalse(resources_overlap("file:/a/b", "file:/a/b/c"))

    def test_directory_covers_descendant_file(self) -> None:
        self.assertTrue(resources_overlap("dir:/a", "file:/a/b.txt"))
        self.assertTrue(resources_overlap("file:/a/b.txt", "dir:/a"))

    def test_directory_covers_nested_directory(self) -> None:
        self.assertTrue(resources_overlap("dir:/a", "dir:/a/b/c"))
        self.assertTrue(resources_overlap("dir:/a/b/c", "dir:/a"))

    def test_unrelated_directories_do_not_overlap(self) -> None:
        self.assertFalse(resources_overlap("dir:/a", "dir:/b"))

    def test_directory_with_sibling_file_does_not_overlap(self) -> None:
        self.assertFalse(resources_overlap("dir:/a/b", "file:/a/c.txt"))

    def test_file_and_directory_at_same_path_overlap(self) -> None:
        # 同一个路径既被当作 file 又被当作 dir，应视为冲突
        self.assertTrue(resources_overlap("file:/a", "dir:/a"))

    def test_workspace_wildcard_overlaps_everything(self) -> None:
        self.assertTrue(resources_overlap("workspace:*", "file:/anywhere/x.txt"))
        self.assertTrue(resources_overlap("file:/anywhere/x.txt", "workspace:*"))

    def test_two_writes_to_sibling_files_can_run_concurrently(self) -> None:
        a = ResourceAccess.write("file:/work/a.txt")
        b = ResourceAccess.write("file:/work/b.txt")
        self.assertFalse(access_conflicts(a, b))

    def test_directory_write_blocks_child_file_write(self) -> None:
        parent = ResourceAccess.write("dir:/work")
        child = ResourceAccess.write("file:/work/a.txt")
        self.assertTrue(access_conflicts(parent, child))


if __name__ == "__main__":
    unittest.main()
