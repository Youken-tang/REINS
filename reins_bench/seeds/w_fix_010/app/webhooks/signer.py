"""Webhook payload signer with wrong JSON canonicalisation."""
import hashlib
import hmac
import json


def sign(secret: str, payload: dict) -> str:
    # BUG: json.dumps uses default separators with spaces;
    # consumer canonicalises without spaces → signatures mismatch.
    body = json.dumps(payload)
    return hmac.new(secret.encode(), body.encode(), hashlib.sha256).hexdigest()
