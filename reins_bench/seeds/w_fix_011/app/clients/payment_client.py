"""Payment client retry loop swallows non-retriable errors."""


class TransientError(Exception):
    pass

class FatalError(Exception):
    pass


def charge(do_call, max_retries: int = 3):
    attempts = 0
    last_exc = None
    while attempts < max_retries:
        try:
            return do_call()
        except Exception as e:  # BUG: catches FatalError too
            last_exc = e
            attempts += 1
    raise last_exc
