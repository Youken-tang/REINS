import gc
import os
import tempfile

from app.importers.csv_importer import import_rows


def _open_fds():
    # Linux-only — read /proc/self/fd
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:
        return 0


def test_no_handle_leak_after_many_imports():
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        f.write("a,b\n1,2\n3,4\n")
        tmp = f.name
    gc.collect()
    before = _open_fds()
    for _ in range(50):
        import_rows(tmp)
    gc.collect()
    after = _open_fds()
    # Allow tiny variance from interpreter; importing 50× shouldn't grow >5.
    assert after - before <= 5, f"leaked {after - before} fds"
