import tempfile, os
from app.ingest.file_reader import read_lines


def test_invalid_byte_replaced_not_dropped():
    with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".log") as f:
        f.write(b"hello\xffworld\n")
        p = f.name
    lines = read_lines(p)
    os.unlink(p)
    assert lines, "lost the line entirely"
    assert "hello" in lines[0] and "world" in lines[0]
