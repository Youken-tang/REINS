import hashlib
import hmac
import json


def sign(secret: str, payload: dict) -> str:
    body = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
