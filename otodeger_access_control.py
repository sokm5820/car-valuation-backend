"""Commercial access, device-seat and usage controls for OtoDeğer.

This module intentionally sits *outside* the Gold decision engine.  The engine
receives only a server-authorised audience/company context after these checks
have passed.

Production model
----------------
* Anonymous/unauthenticated traffic may use a tightly limited Personal trial.
* PERSONAL and PERSONAL_PLUS are authenticated consumer entitlements.
* BUSINESS is an authenticated organisation entitlement.
* One Business seat == one registered device.  A new device consumes a seat
  until an organisation admin explicitly deactivates it.
* Client supplied tier/company values are ignored when security mode is
  ``enforced``.
* Redis is required in ``enforced`` mode so limits work across workers.

Authentication provider integration
-----------------------------------
Clerk verifies browser session JWTs directly on the Flask request. OtoDeğer then
resolves product entitlements server-side; browser-supplied tier/company values
never grant paid access. Clerk proves identity and organisation membership, while
OtoDeğer remains authoritative for Personal+, Business company mapping, device
seats and usage limits.

The older OtoDeğer HMAC session-token helpers remain only for local regression
tests and controlled migration. Paid launch uses Clerk + ``enforced`` mode.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

try:
    from clerk_backend_api import AuthenticateRequestOptions, authenticate_request
except ImportError:  # Local/offline tests can still exercise non-Clerk paths.
    AuthenticateRequestOptions = None
    authenticate_request = None


VALID_SECURITY_MODES = {"legacy", "shadow", "enforced"}
VALID_TIERS = {"ANONYMOUS", "PERSONAL", "PERSONAL_PLUS", "BUSINESS"}


class AccessControlError(RuntimeError):
    code = "ACCESS_DENIED"
    status_code = 403

    def __init__(self, message: str = "Access denied", *, retry_after: int = 0, metadata: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.retry_after = max(0, int(retry_after or 0))
        self.metadata = dict(metadata or {})


class AuthenticationRequired(AccessControlError):
    code = "AUTHENTICATION_REQUIRED"
    status_code = 401


class BusinessAccessRequired(AccessControlError):
    code = "BUSINESS_ACCESS_REQUIRED"
    status_code = 403


class DeviceSeatLimitReached(AccessControlError):
    code = "DEVICE_SEAT_LIMIT_REACHED"
    status_code = 403


class UsageLimitReached(AccessControlError):
    code = "USAGE_LIMIT_REACHED"
    status_code = 429


class ConcurrentRequestLimitReached(AccessControlError):
    code = "CONCURRENT_REQUEST_LIMIT_REACHED"
    status_code = 429


class SecurityConfigurationError(AccessControlError):
    code = "SECURITY_CONFIGURATION_ERROR"
    status_code = 503


@dataclass(frozen=True)
class AccessContext:
    authenticated: bool
    user_id: str
    tier: str
    device_id: str
    org_id: str = ""
    org_name: str = ""
    org_role: str = "member"
    seat_limit: int = 0
    source: str = "anonymous"

    @property
    def business_entitled(self) -> bool:
        return self.authenticated and self.tier == "BUSINESS" and bool(self.org_id and self.org_name)

    @property
    def engine_access_tier(self) -> str:
        return "BUSINESS" if self.business_entitled else "PERSONAL"

    def public_payload(self) -> Dict[str, Any]:
        return {
            "authenticated": bool(self.authenticated),
            "tier": self.tier,
            "business_entitled": bool(self.business_entitled),
            "business_company": self.org_name if self.business_entitled else None,
            "organisation_id": self.org_id if self.business_entitled else None,
            "organisation_role": self.org_role if self.business_entitled else None,
            "seat_limit": int(self.seat_limit or 0) if self.business_entitled else 0,
            "auth_source": self.source,
        }


@dataclass(frozen=True)
class RequestLease:
    key: str
    token: str


DEFAULT_LIMITS: Dict[str, Dict[str, int]] = {
    # These are intentionally conservative launch defaults and can be raised by
    # environment variable without code changes after real usage is measured.
    "ANONYMOUS": {"ten_min": 4, "day": 8, "concurrent": 1},
    "PERSONAL": {"ten_min": 10, "day": 35, "concurrent": 1},
    "PERSONAL_PLUS": {"ten_min": 30, "day": 150, "concurrent": 2},
    "BUSINESS": {"ten_min": 40, "day": 250, "concurrent": 2},
}


def _env_int(name: str, default: int, *, minimum: int = 1, maximum: int = 1_000_000) -> int:
    try:
        value = int(str(os.getenv(name, default)).strip())
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(value, maximum))


def _env_csv(name: str, default: str = "") -> Tuple[str, ...]:
    raw = str(os.getenv(name, default) or "")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def _env_json_object(name: str) -> Dict[str, Any]:
    raw = str(os.getenv(name) or "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SecurityConfigurationError(f"{name} must contain valid JSON") from exc
    if not isinstance(value, dict):
        raise SecurityConfigurationError(f"{name} must be a JSON object")
    return value


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    value = str(value or "")
    padding = "=" * ((4 - len(value) % 4) % 4)
    return base64.urlsafe_b64decode((value + padding).encode("ascii"))


def _json_b64(value: Mapping[str, Any]) -> str:
    return _b64url_encode(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _safe_identifier(value: Any, *, max_len: int = 160) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Identifiers are never rendered directly as HTML.  Still constrain the
    # character set so they are safe Redis key ingredients after hashing.
    return text[:max_len]


def _normalise_tier(value: Any) -> str:
    value = str(value or "").strip().upper()
    aliases = {
        "FREE": "ANONYMOUS",
        "TRIAL": "ANONYMOUS",
        "PERSONAL": "PERSONAL",
        "PERSONAL+": "PERSONAL_PLUS",
        "PERSONAL_PLUS": "PERSONAL_PLUS",
        "PLUS": "PERSONAL_PLUS",
        "BUSINESS": "BUSINESS",
        "DEALER": "BUSINESS",
        "GALLERY": "BUSINESS",
    }
    return aliases.get(value, "PERSONAL")


def _hash_key(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()[:32]


class _MemoryBackend:
    """Test/local fallback.  Never considered sufficient for enforced mode."""

    def __init__(self):
        self.lock = threading.RLock()
        self.counters: Dict[str, Tuple[int, float]] = {}
        self.values: Dict[str, Tuple[str, float]] = {}
        self.sets: Dict[str, set] = {}
        self.hashes: Dict[str, Dict[str, str]] = {}

    def incr_window(self, key: str, ttl: int) -> Tuple[int, int]:
        now = time.time()
        with self.lock:
            count, expires = self.counters.get(key, (0, now + ttl))
            if expires <= now:
                count, expires = 0, now + ttl
            count += 1
            self.counters[key] = (count, expires)
            return count, max(1, int(expires - now))

    def set_nx(self, key: str, value: str, ttl: int) -> bool:
        now = time.time()
        with self.lock:
            current = self.values.get(key)
            if current and current[1] > now:
                return False
            self.values[key] = (value, now + ttl)
            return True

    def compare_delete(self, key: str, value: str) -> None:
        with self.lock:
            current = self.values.get(key)
            if current and current[0] == value:
                self.values.pop(key, None)

    def sadd(self, key: str, member: str) -> int:
        with self.lock:
            bucket = self.sets.setdefault(key, set())
            before = len(bucket)
            bucket.add(member)
            return 1 if len(bucket) > before else 0

    def srem(self, key: str, member: str) -> int:
        with self.lock:
            bucket = self.sets.setdefault(key, set())
            existed = member in bucket
            bucket.discard(member)
            return 1 if existed else 0

    def smembers(self, key: str) -> Iterable[str]:
        with self.lock:
            return set(self.sets.get(key, set()))

    def hset(self, key: str, mapping: Mapping[str, str]) -> None:
        with self.lock:
            self.hashes.setdefault(key, {}).update({str(k): str(v) for k, v in mapping.items()})

    def hgetall(self, key: str) -> Dict[str, str]:
        with self.lock:
            return dict(self.hashes.get(key, {}))

    def delete(self, key: str) -> None:
        with self.lock:
            self.hashes.pop(key, None)
            self.sets.pop(key, None)
            self.values.pop(key, None)
            self.counters.pop(key, None)

    def ping(self) -> bool:
        return True


class _RedisBackend:
    _INCR_SCRIPT = """
local n = redis.call('INCR', KEYS[1])
if n == 1 then redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1])) end
local ttl = redis.call('TTL', KEYS[1])
return {n, ttl}
"""
    _COMPARE_DELETE = """
if redis.call('GET', KEYS[1]) == ARGV[1] then
  return redis.call('DEL', KEYS[1])
end
return 0
"""

    def __init__(self, redis_url: str):
        import redis  # type: ignore

        self.redis = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=3,
            socket_timeout=3,
            health_check_interval=30,
        )
        self._incr = self.redis.register_script(self._INCR_SCRIPT)
        self._compare_delete = self.redis.register_script(self._COMPARE_DELETE)

    def incr_window(self, key: str, ttl: int) -> Tuple[int, int]:
        result = self._incr(keys=[key], args=[int(ttl)])
        return int(result[0]), max(1, int(result[1]))

    def set_nx(self, key: str, value: str, ttl: int) -> bool:
        return bool(self.redis.set(key, value, ex=int(ttl), nx=True))

    def compare_delete(self, key: str, value: str) -> None:
        self._compare_delete(keys=[key], args=[value])

    def sadd(self, key: str, member: str) -> int:
        return int(self.redis.sadd(key, member))

    def srem(self, key: str, member: str) -> int:
        return int(self.redis.srem(key, member))

    def smembers(self, key: str) -> Iterable[str]:
        return set(self.redis.smembers(key) or [])

    def hset(self, key: str, mapping: Mapping[str, str]) -> None:
        self.redis.hset(key, mapping=dict(mapping))

    def hgetall(self, key: str) -> Dict[str, str]:
        return dict(self.redis.hgetall(key) or {})

    def delete(self, key: str) -> None:
        self.redis.delete(key)

    def ping(self) -> bool:
        return bool(self.redis.ping())


class CommercialAccessManager:
    COOKIE_NAME = "otodeger_session"

    def __init__(self):
        mode = str(os.getenv("COMMERCIAL_SECURITY_MODE", "legacy")).strip().casefold()
        self.mode = mode if mode in VALID_SECURITY_MODES else "legacy"
        # Clerk is the identity provider. OtoDeğer still owns paid entitlements.
        self.clerk_secret_key = str(os.getenv("CLERK_SECRET_KEY") or "").strip()
        self.clerk_jwt_key = str(os.getenv("CLERK_JWT_KEY") or "").replace("\\n", "\n").strip()
        self.clerk_authorized_parties = _env_csv(
            "CLERK_AUTHORIZED_PARTIES",
            "http://localhost:5173,https://otodeger.online,https://www.otodeger.online",
        )
        self.personal_plus_users = set(_env_csv("OTODEGER_PERSONAL_PLUS_USERS"))
        self.business_orgs = self._load_business_orgs(_env_json_object("OTODEGER_BUSINESS_ORGS_JSON"))

        # Legacy signed sessions are retained only for local regression/migration.
        self.session_secret = str(os.getenv("OTODEGER_SESSION_SECRET") or "").strip()
        self.session_ttl_seconds = _env_int("OTODEGER_SESSION_TTL_SECONDS", 7 * 24 * 60 * 60, minimum=300)
        self.concurrent_ttl_seconds = _env_int("ASSISTANT_CONCURRENCY_TTL_SECONDS", 120, minimum=30, maximum=600)
        self.namespace = str(os.getenv("COMMERCIAL_REDIS_NAMESPACE", "otodeger:commercial:v1")).strip() or "otodeger:commercial:v1"
        redis_url = str(os.getenv("REDIS_URL") or "").strip()
        self.backend = _RedisBackend(redis_url) if redis_url else _MemoryBackend()
        self.has_shared_backend = bool(redis_url)

        if self.mode == "enforced":
            if authenticate_request is None or AuthenticateRequestOptions is None:
                raise SecurityConfigurationError("clerk-backend-api is required in enforced mode")
            if not self.clerk_secret_key:
                raise SecurityConfigurationError("CLERK_SECRET_KEY is required in enforced mode")
            if not self.clerk_authorized_parties:
                raise SecurityConfigurationError("CLERK_AUTHORIZED_PARTIES is required in enforced mode")
            if not self.has_shared_backend:
                raise SecurityConfigurationError("REDIS_URL is required in enforced mode")
            if not self.backend.ping():
                raise SecurityConfigurationError("Redis is unavailable in enforced mode")

        self.limits = {}
        for tier, defaults in DEFAULT_LIMITS.items():
            prefix = tier.upper()
            self.limits[tier] = {
                "ten_min": _env_int(f"{prefix}_ASSISTANT_10MIN", defaults["ten_min"]),
                "day": _env_int(f"{prefix}_ASSISTANT_DAY", defaults["day"]),
                "concurrent": _env_int(f"{prefix}_ASSISTANT_CONCURRENT", defaults["concurrent"], maximum=20),
            }

        self.ai_global_hour = _env_int("COMMERCIAL_AI_GLOBAL_CALLS_HOUR", 250)
        self.ai_global_day = _env_int("COMMERCIAL_AI_GLOBAL_CALLS_DAY", 1000)
        self.ai_identity_hour = _env_int("COMMERCIAL_AI_IDENTITY_CALLS_HOUR", 60)

    @property
    def enforcement_enabled(self) -> bool:
        return self.mode == "enforced"

    @property
    def clerk_configured(self) -> bool:
        return bool(self.clerk_secret_key and self.clerk_authorized_parties and authenticate_request is not None)

    @staticmethod
    def _load_business_orgs(raw: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Normalize the server-owned Clerk-org -> OtoDeğer Business map.

        Accepted form:
        {"org_x": {"name": "Gallery Name", "seat_limit": 2}}
        A plain string value is also accepted and defaults to one device seat.
        """
        result: Dict[str, Dict[str, Any]] = {}
        for raw_org_id, raw_config in dict(raw or {}).items():
            org_id = _safe_identifier(raw_org_id)
            if not org_id:
                continue
            if isinstance(raw_config, str):
                name = raw_config.strip()[:160]
                seat_limit = 1
            elif isinstance(raw_config, Mapping):
                name = str(raw_config.get("name") or raw_config.get("company") or "").strip()[:160]
                try:
                    seat_limit = int(raw_config.get("seat_limit") or raw_config.get("seats") or 1)
                except (TypeError, ValueError):
                    seat_limit = 1
            else:
                continue
            if not name:
                continue
            result[org_id] = {
                "name": name,
                "seat_limit": max(1, min(seat_limit, 10_000)),
            }
        return result

    def _key(self, *parts: str) -> str:
        clean = [self.namespace] + [str(p).strip(":") for p in parts if str(p)]
        return ":".join(clean)

    # ---------- signed session -------------------------------------------------
    def issue_session_token(
        self,
        *,
        user_id: str,
        tier: str,
        org_id: str = "",
        org_name: str = "",
        org_role: str = "member",
        seat_limit: int = 0,
        ttl_seconds: Optional[int] = None,
    ) -> str:
        if not self.session_secret:
            raise SecurityConfigurationError("OTODEGER_SESSION_SECRET is not configured")
        user_id = _safe_identifier(user_id)
        if not user_id:
            raise ValueError("user_id is required")
        tier = _normalise_tier(tier)
        now = int(time.time())
        ttl = int(ttl_seconds or self.session_ttl_seconds)
        payload = {
            "sub": user_id,
            "tier": tier,
            "org_id": _safe_identifier(org_id),
            "org_name": str(org_name or "").strip()[:160],
            "org_role": str(org_role or "member").strip().lower()[:40],
            "seat_limit": max(0, int(seat_limit or 0)),
            "iat": now,
            "exp": now + ttl,
            "jti": secrets.token_urlsafe(12),
            "v": 1,
        }
        header = {"alg": "HS256", "typ": "JWT"}
        encoded_header = _json_b64(header)
        encoded_payload = _json_b64(payload)
        signing_input = f"{encoded_header}.{encoded_payload}".encode("ascii")
        signature = hmac.new(self.session_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
        return f"{encoded_header}.{encoded_payload}.{_b64url_encode(signature)}"

    def _decode_session_token(self, token: str) -> Optional[Mapping[str, Any]]:
        if not token or not self.session_secret:
            return None
        try:
            header_b64, payload_b64, signature_b64 = token.split(".", 2)
            signing_input = f"{header_b64}.{payload_b64}".encode("ascii")
            expected = hmac.new(self.session_secret.encode("utf-8"), signing_input, hashlib.sha256).digest()
            provided = _b64url_decode(signature_b64)
            if not hmac.compare_digest(expected, provided):
                return None
            header = json.loads(_b64url_decode(header_b64).decode("utf-8"))
            payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
            if header.get("alg") != "HS256" or int(payload.get("v") or 0) != 1:
                return None
            now = int(time.time())
            if int(payload.get("exp") or 0) <= now or int(payload.get("iat") or 0) > now + 60:
                return None
            return payload
        except Exception:
            return None

    def _request_token(self, request_obj: Any) -> str:
        auth = str(request_obj.headers.get("Authorization") or "").strip()
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        try:
            return str(request_obj.cookies.get(self.COOKIE_NAME) or "").strip()
        except Exception:
            return ""

    def _verify_clerk_claims(self, request_obj: Any) -> Optional[Mapping[str, Any]]:
        auth = str(request_obj.headers.get("Authorization") or "").strip()
        if not auth.lower().startswith("bearer "):
            return None
        if not self.clerk_configured:
            return None

        try:
            state = authenticate_request(
                request_obj,
                AuthenticateRequestOptions(
                    secret_key=self.clerk_secret_key,
                    jwt_key=self.clerk_jwt_key or None,
                    authorized_parties=list(self.clerk_authorized_parties),
                    accepts_token=["session_token"],
                ),
            )
        except Exception as exc:
            if self.mode == "enforced":
                raise AuthenticationRequired("Invalid or unavailable Clerk session") from exc
            return None

        if not getattr(state, "is_signed_in", False):
            if self.mode != "enforced":
                return None
            reason_obj = getattr(state, "reason", None)
            reason = getattr(reason_obj, "name", None) or "UNAUTHORIZED"
            raise AuthenticationRequired(f"Clerk session rejected: {reason}")

        payload = getattr(state, "payload", None) or {}
        return payload if isinstance(payload, Mapping) else {}

    def _context_from_clerk_claims(self, claims: Mapping[str, Any], *, device_id: str, client_ip: str) -> AccessContext:
        user_id = _safe_identifier(claims.get("sub"))
        if not user_id:
            raise AuthenticationRequired("Clerk session does not contain a user id")

        clerk_org_id = _safe_identifier(claims.get("org_id"))
        role = str(claims.get("org_role") or "member").strip().lower()[:40]
        business = self.business_orgs.get(clerk_org_id) if clerk_org_id else None

        if business:
            return AccessContext(
                authenticated=True,
                user_id=user_id,
                tier="BUSINESS",
                device_id=device_id,
                org_id=clerk_org_id,
                org_name=str(business["name"]),
                org_role=role,
                seat_limit=int(business["seat_limit"]),
                source="clerk_session",
            )

        tier = "PERSONAL_PLUS" if user_id in self.personal_plus_users else "PERSONAL"
        return AccessContext(
            authenticated=True,
            user_id=user_id,
            tier=tier,
            device_id=device_id,
            source="clerk_session",
        )

    def resolve_context(
        self,
        request_obj: Any,
        *,
        client_tier: Any = None,
        client_company: Any = None,
        client_ip: str = "",
    ) -> AccessContext:
        device_id = _safe_identifier(request_obj.headers.get("X-Otodeger-Device"), max_len=120)
        if not device_id:
            # Anonymous users still need a stable-ish quota scope.  IP is never a
            # Business device seat and is hashed before storage.
            device_id = f"anon-{_hash_key(client_ip or 'unknown')}"

        token = self._request_token(request_obj)

        # Controlled migration path: accept an OtoDeğer-signed token if present.
        # This is not the paid-launch browser auth path; Clerk is.
        claims = self._decode_session_token(token)
        if claims:
            tier = _normalise_tier(claims.get("tier"))
            user_id = _safe_identifier(claims.get("sub"))
            org_id = _safe_identifier(claims.get("org_id"))
            org_name = str(claims.get("org_name") or "").strip()[:160]
            role = str(claims.get("org_role") or "member").strip().lower()[:40]
            seat_limit = max(0, int(claims.get("seat_limit") or 0))
            if tier == "BUSINESS" and (not org_id or not org_name or seat_limit < 1):
                tier = "PERSONAL"
                org_id = org_name = ""
                seat_limit = 0
            return AccessContext(
                authenticated=bool(user_id),
                user_id=user_id or f"anon-{_hash_key(client_ip)}",
                tier=tier,
                device_id=device_id,
                org_id=org_id,
                org_name=org_name,
                org_role=role,
                seat_limit=seat_limit,
                source="signed_session",
            )

        clerk_claims = self._verify_clerk_claims(request_obj)
        if clerk_claims is not None:
            return self._context_from_clerk_claims(
                clerk_claims,
                device_id=device_id,
                client_ip=client_ip,
            )

        # If a bearer token was supplied but neither accepted auth path verified it,
        # fail closed once paid-launch enforcement is enabled.
        auth_header = str(request_obj.headers.get("Authorization") or "").strip()
        if self.mode == "enforced" and auth_header.lower().startswith("bearer "):
            raise AuthenticationRequired("Invalid authentication token")

        if self.mode in {"legacy", "shadow"}:
            tier = _normalise_tier(client_tier or "PERSONAL")
            company = str(client_company or "").strip()[:160]
            # Legacy business is intentionally marked unauthenticated.  The app
            # may keep owner testing working, but paid-launch code can distinguish
            # it from a real entitlement.
            return AccessContext(
                authenticated=False,
                user_id=f"legacy-{_hash_key(client_ip or device_id)}",
                tier=tier,
                device_id=device_id,
                org_id=f"legacy-{_hash_key(company)}" if tier == "BUSINESS" and company else "",
                org_name=company if tier == "BUSINESS" else "",
                seat_limit=999 if tier == "BUSINESS" and company else 0,
                source="legacy_client",
            )

        return AccessContext(
            authenticated=False,
            user_id=f"anon-{_hash_key(client_ip or device_id)}",
            tier="ANONYMOUS",
            device_id=device_id,
            source="anonymous",
        )

    # ---------- Business device seats -----------------------------------------
    def _org_devices_key(self, org_id: str) -> str:
        return self._key("org", _hash_key(org_id), "devices")

    def _device_meta_key(self, org_id: str, device_id: str) -> str:
        return self._key("org", _hash_key(org_id), "device", _hash_key(device_id))

    def activate_business_device(self, context: AccessContext) -> Dict[str, Any]:
        if not context.business_entitled:
            raise BusinessAccessRequired("A verified Business organisation is required")
        if not context.device_id or context.device_id.startswith("anon-"):
            raise AccessControlError("A device identifier is required for Business access")

        devices_key = self._org_devices_key(context.org_id)
        existing = set(self.backend.smembers(devices_key))
        device_hash = _hash_key(context.device_id)
        already_registered = device_hash in existing
        if not already_registered and len(existing) >= int(context.seat_limit):
            raise DeviceSeatLimitReached(
                "No Business device seats are available",
                metadata={"seat_limit": int(context.seat_limit), "seats_used": len(existing)},
            )
        if not already_registered:
            self.backend.sadd(devices_key, device_hash)

        now = int(time.time())
        self.backend.hset(self._device_meta_key(context.org_id, context.device_id), {
            "device_hash": device_hash,
            "user_id_hash": _hash_key(context.user_id),
            "first_seen": str(now) if not already_registered else self.backend.hgetall(self._device_meta_key(context.org_id, context.device_id)).get("first_seen", str(now)),
            "last_seen": str(now),
        })
        return {
            "seat_limit": int(context.seat_limit),
            "seats_used": len(existing) if already_registered else len(existing) + 1,
            "device_registered": True,
        }

    def list_business_devices(self, context: AccessContext) -> Dict[str, Any]:
        if not context.business_entitled:
            raise BusinessAccessRequired()
        if context.org_role not in {"admin", "owner"}:
            raise AccessControlError("Organisation admin access is required")
        members = sorted(self.backend.smembers(self._org_devices_key(context.org_id)))
        devices = []
        # Metadata keys cannot be recovered from hashes alone, intentionally.  We
        # therefore return privacy-preserving identifiers; the frontend can label
        # the current device locally.  A future durable DB can add friendly names.
        current_hash = _hash_key(context.device_id)
        for member in members:
            devices.append({
                "device_id": member,
                "current": member == current_hash,
            })
        return {"seat_limit": int(context.seat_limit), "seats_used": len(members), "devices": devices}

    def deactivate_business_device(self, context: AccessContext, device_hash: str) -> Dict[str, Any]:
        if not context.business_entitled:
            raise BusinessAccessRequired()
        if context.org_role not in {"admin", "owner"}:
            raise AccessControlError("Organisation admin access is required")
        member = str(device_hash or "").strip().lower()
        if not re_full_hash(member):
            raise AccessControlError("Invalid device identifier")
        removed = self.backend.srem(self._org_devices_key(context.org_id), member)
        return {"removed": bool(removed), "device_id": member}

    # ---------- quotas / concurrency ------------------------------------------
    def _identity_scope(self, context: AccessContext, client_ip: str) -> str:
        if context.authenticated:
            return _hash_key(f"user:{context.user_id}")
        return _hash_key(f"anon:{context.device_id}:{client_ip}")

    def consume_assistant_quota(self, context: AccessContext, *, client_ip: str) -> Dict[str, Any]:
        if self.mode != "enforced":
            return {"tier": context.tier, "remaining_day": None}

        limits = self.limits.get(context.tier, self.limits["ANONYMOUS"])
        scope = self._identity_scope(context, client_ip)
        count_10, retry_10 = self.backend.incr_window(self._key("quota", scope, "10m"), 600)
        if count_10 > limits["ten_min"]:
            raise UsageLimitReached("Short-term usage limit reached", retry_after=retry_10, metadata={"window": "10m"})
        count_day, retry_day = self.backend.incr_window(self._key("quota", scope, "day"), 86400)
        if count_day > limits["day"]:
            raise UsageLimitReached("Daily usage limit reached", retry_after=retry_day, metadata={"window": "day"})
        return {
            "tier": context.tier,
            "remaining_10min": max(0, limits["ten_min"] - count_10),
            "remaining_day": max(0, limits["day"] - count_day),
        }

    def acquire_concurrency(self, context: AccessContext, *, client_ip: str) -> RequestLease:
        if self.mode != "enforced":
            return RequestLease("", "")
        limits = self.limits.get(context.tier, self.limits["ANONYMOUS"])
        scope = self._identity_scope(context, client_ip)
        # Each concurrency slot is a separate NX key.  This works across Render
        # workers and self-heals after the request TTL if a process crashes.
        for slot in range(int(limits["concurrent"])):
            key = self._key("inflight", scope, str(slot))
            token = secrets.token_urlsafe(16)
            if self.backend.set_nx(key, token, self.concurrent_ttl_seconds):
                return RequestLease(key, token)
        raise ConcurrentRequestLimitReached("Too many simultaneous assistant requests", retry_after=3)

    def release_concurrency(self, lease: Optional[RequestLease]) -> None:
        if not lease or not lease.key or not lease.token:
            return
        try:
            self.backend.compare_delete(lease.key, lease.token)
        except Exception:
            pass

    def consume_ai_call_budget(self, context: AccessContext, *, client_ip: str) -> None:
        if self.mode != "enforced":
            return
        global_h, retry_h = self.backend.incr_window(self._key("ai", "global", "hour"), 3600)
        if global_h > self.ai_global_hour:
            raise UsageLimitReached("Global AI hourly circuit breaker reached", retry_after=retry_h)
        global_d, retry_d = self.backend.incr_window(self._key("ai", "global", "day"), 86400)
        if global_d > self.ai_global_day:
            raise UsageLimitReached("Global AI daily circuit breaker reached", retry_after=retry_d)
        scope = self._identity_scope(context, client_ip)
        identity_h, retry_i = self.backend.incr_window(self._key("ai", scope, "hour"), 3600)
        if identity_h > self.ai_identity_hour:
            raise UsageLimitReached("AI usage circuit breaker reached", retry_after=retry_i)

    def status_payload(self, context: AccessContext) -> Dict[str, Any]:
        payload = context.public_payload()
        payload.update({
            "security_mode": self.mode,
            "business_access": bool(context.business_entitled) if self.enforcement_enabled else True,
            "device_seat_enforced": bool(self.enforcement_enabled),
        })
        if context.business_entitled:
            devices = set(self.backend.smembers(self._org_devices_key(context.org_id)))
            payload["seats_used"] = len(devices)
        else:
            payload["seats_used"] = 0
        return payload


def re_full_hash(value: str) -> bool:
    if len(value) != 32:
        return False
    return all(ch in "0123456789abcdef" for ch in value)


_MANAGER: Optional[CommercialAccessManager] = None
_MANAGER_LOCK = threading.Lock()


def get_access_manager() -> CommercialAccessManager:
    global _MANAGER
    with _MANAGER_LOCK:
        if _MANAGER is None:
            _MANAGER = CommercialAccessManager()
        return _MANAGER


def reset_access_manager_for_tests() -> None:
    global _MANAGER
    with _MANAGER_LOCK:
        _MANAGER = None


def issue_session_token(**kwargs: Any) -> str:
    return get_access_manager().issue_session_token(**kwargs)


__all__ = [
    "AccessContext",
    "RequestLease",
    "AccessControlError",
    "AuthenticationRequired",
    "BusinessAccessRequired",
    "DeviceSeatLimitReached",
    "UsageLimitReached",
    "ConcurrentRequestLimitReached",
    "SecurityConfigurationError",
    "CommercialAccessManager",
    "get_access_manager",
    "reset_access_manager_for_tests",
    "issue_session_token",
]
