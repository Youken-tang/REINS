import os
from app.utils.tempfile_helper import create_temp


def test_create_temp_returns_existing_path():
    p = create_temp()
    try:
        assert os.path.exists(p)
        # On a TOCTOU-safe impl, two calls return distinct paths.
        p2 = create_temp()
        try:
            assert p != p2
        finally:
            os.unlink(p2)
    finally:
        if os.path.exists(p): os.unlink(p)
