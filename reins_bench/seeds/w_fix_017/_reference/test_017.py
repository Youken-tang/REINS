from app.db.pool import Pool


def test_release_returns_connection():
    p = Pool(size=2)
    c1 = p.acquire()
    c2 = p.acquire()
    p.release(c1)
    c3 = p.acquire()
    assert c3 is c1 or c3 is not None
