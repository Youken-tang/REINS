"""User serialiser missing the new `email_verified` field."""


def serialise_user(user: dict) -> dict:
    # BUG: model now has email_verified but serialiser doesn't expose it.
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
    }
