"""Login endpoint with no rate limit — password brute-force possible."""


class LoginService:
    def __init__(self, password_check):
        self._check = password_check

    def login(self, ip: str, user: str, password: str) -> bool:
        # BUG: no per-IP throttling.
        return self._check(user, password)
