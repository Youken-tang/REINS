"""Signed-token verifier — expiry not checked."""
import hmac, hashlib, json, base64, time


def issue(payload: dict, secret: str, ttl_seconds: int = 60) -> str:
    payload = dict(payload, exp=int(time.time()) + ttl_seconds)
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()
    sig = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    return f"{body}.{sig}"


def verify(token: str, secret: str) -> dict | None:
    body, sig = token.rsplit(".", 1)
    expected = hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, expected):
        return None
    # BUG: signature ok but exp not checked — expired tokens still pass.
    return json.loads(base64.urlsafe_b64decode(body.encode()).decode())
