import pytest
from app.api.upload import handle_upload


def test_oversized_rejected():
    with pytest.raises(ValueError):
        handle_upload("a.png", 100 * 1024 * 1024, "image/png")

def test_path_traversal_rejected():
    with pytest.raises(ValueError):
        handle_upload("../../etc/passwd", 100, "image/png")

def test_disallowed_mime_rejected():
    with pytest.raises(ValueError):
        handle_upload("a.exe", 100, "application/x-msdownload")

def test_happy_path():
    assert handle_upload("photo.png", 1024, "image/png")["ok"]
