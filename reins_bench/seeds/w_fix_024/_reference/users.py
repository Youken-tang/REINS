def serialise_user(user: dict) -> dict:
    return {
        "id": user["id"],
        "name": user["name"],
        "email": user["email"],
        "email_verified": bool(user.get("email_verified", False)),
    }
