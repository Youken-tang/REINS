import time

_MAX_ATTEMPTS = 5
_WINDOW_SECONDS = 60


class LoginService:
    def __init__(self, password_check):
        self._check = password_check
        self._attempts: dict[str, list[float]] = {}

    def login(self, ip: str, user: str, password: str) -> bool:
        now = time.monotonic()
        history = [t for t in self._attempts.get(ip, []) if now - t < _WINDOW_SECONDS]
        if len(history) >= _MAX_ATTEMPTS:
            raise PermissionError("rate limited")
        history.append(now)
        self._attempts[ip] = history
        return self._check(user, password)
