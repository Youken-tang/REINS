from app.services.profile_cache import ProfileCache


def test_update_invalidates_cache():
    pc = ProfileCache()
    pc._store[1] = {"name": "alice"}
    assert pc.get(1)["name"] == "alice"
    pc.update(1, name="bob")
    assert pc.get(1)["name"] == "bob"
