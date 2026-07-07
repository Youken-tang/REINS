"""Profile update endpoint — no CSRF token check."""


class ProfileService:
    def __init__(self):
        self.profiles: dict[int, dict] = {}

    def update(self, uid: int, csrf_token: str | None, fields: dict) -> dict:
        # BUG: csrf_token unused → CSRF attacks succeed.
        p = self.profiles.setdefault(uid, {})
        p.update(fields)
        return p
