from app.api.users import serialise_user


def test_serialiser_includes_email_verified():
    u = {"id": 1, "name": "a", "email": "a@x.com", "email_verified": True}
    out = serialise_user(u)
    assert out["email_verified"] is True

def test_serialiser_defaults_email_verified_false():
    u = {"id": 2, "name": "b", "email": "b@x.com"}
    out = serialise_user(u)
    assert out["email_verified"] is False
