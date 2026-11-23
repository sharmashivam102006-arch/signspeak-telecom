"""Security helpers: rate limiting.

Policy table (PATH_LIMITS) is the single source of truth for per-route
rate limits. It is enforced centrally by ``RateLimitMiddleware`` (see
below) so that limits also cover routes whose files are owned by other
agents (``/api/auth/login``, ``/api/auth/guest``,
``/api/ai/sign/predict``, ``/api/ai/sign/collect``,
``/api/ai/sign/extract``).

* ``limiter`` — the slowapi ``Limiter``, registered as
  ``app.state.limiter`` in ``app/main.py`` so route owners CAN use
  ``@limiter.limit(...)`` decorators on their own routes. Do NOT add
  such a decorator to a route already listed in PATH_LIMITS: the
  request would be counted against two independent budgets with the
  same limit (harmless, but redundant and confusing).
* ``RateLimitMiddleware`` — a ``BaseHTTPMiddleware`` that enforces
  PATH_LIMITS with a fixed-window strategy (the same strategy slowapi
  uses by default), keyed per client IP via slowapi's
  ``get_remote_address`` (X-Forwarded-For aware). Exceeding a limit
  returns HTTP 429 ``{"detail": "Rate limit exceeded: ..."}``.
  The check runs before auth dependencies, so unauthenticated
  requests also consume budget — but the budget is per-IP, so an
  attacker can only exhaust their own IP's allowance, never another
  user's.
* WebSocket rate limiting is out of scope (see audit): /ws/call and
  /ws/captions are not limited by this module. Both sockets already
  enforce a 64KB per-message cap and drop malformed messages.

Tests use ``reset_rate_limits()`` to clear the in-memory counters
between scenarios (all TestClient requests share one client IP).
"""
import re

from limits import parse as _parse_limit
from limits.storage import MemoryStorage
from limits.strategies import FixedWindowRateLimiter
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

# The slowapi limiter (for @limiter.limit decorators on routes whose
# owners choose to use them). Registered as app.state.limiter.
limiter = Limiter(key_func=get_remote_address)

# (HTTP method, path regex, slowapi-style limit string)
PATH_LIMITS = [
    ("POST", re.compile(r"^/api/auth/login$"), "10/minute"),
    ("POST", re.compile(r"^/api/auth/guest$"), "10/minute"),
    ("POST", re.compile(r"^/api/ai/tts$"), "30/minute"),
    ("POST", re.compile(r"^/api/ai/transcribe/audio$"), "20/minute"),
    ("POST", re.compile(r"^/api/ai/sign/extract$"), "30/minute"),
    ("POST", re.compile(r"^/api/ai/sign/predict$"), "60/minute"),
    ("POST", re.compile(r"^/api/ai/sign/collect$"), "30/minute"),
]

_storage = MemoryStorage()
_strategy = FixedWindowRateLimiter(_storage)
_PARSED_LIMITS = [
    (method, pattern, _parse_limit(limit_str))
    for method, pattern, limit_str in PATH_LIMITS
]


def reset_rate_limits() -> None:
    """Clear all in-memory rate-limit counters (used by tests)."""
    _storage.reset()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Enforce PATH_LIMITS per client IP. 429 when a limit is exceeded."""

    async def dispatch(self, request, call_next):
        for method, pattern, limit in _PARSED_LIMITS:
            if request.method == method and pattern.match(request.url.path):
                key = get_remote_address(request)
                if not _strategy.hit(limit, key):
                    return JSONResponse(
                        status_code=429,
                        content={"detail": (
                            f"Rate limit exceeded: {limit} "
                            f"for {request.method} {request.url.path}")},
                    )
                break
        return await call_next(request)
