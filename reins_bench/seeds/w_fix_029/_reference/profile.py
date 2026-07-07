import hmac


class ProfileService:
    def __init__(self, expected_token: str = "valid"):
        self.profiles: dict[int, dict] = {}
        self._expected = expected_token

    def update(self, uid: int, csrf_token: str | None, fields: dict) -> dict:
        if not csrf_token or not hmac.compare_digest(csrf_token, self._expected):
            raise PermissionError("missing or invalid CSRF token")
        p = self.profiles.setdefault(uid, {})
        p.update(fields)
        return p
