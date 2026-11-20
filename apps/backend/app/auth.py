"""Provider-neutral verification: the API accepts tokens but never issues account credentials."""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPublicKey
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .config import Settings, get_settings

bearer = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class Principal:
    user_id: str


@lru_cache(maxsize=4)
def _public_key(pem: str):
    key = serialization.load_pem_public_key(pem.encode())
    if not isinstance(key, RSAPublicKey) or key.key_size < 2048:
        raise ValueError("An RSA public key of at least 2048 bits is required")
    return key


def validate_auth_configuration(settings: Settings):
    if not (settings.auth_public_key and settings.auth_issuer and settings.auth_audience):
        raise ValueError("Authentication issuer, audience and RSA public key must be configured")
    return _public_key(settings.auth_public_key)


def verify_token(token: str, settings: Settings) -> Principal:
    try:
        key = validate_auth_configuration(settings)
    except (ValueError, TypeError):
        raise HTTPException(503, "Authentication is not configured") from None
    try:
        if not token or len(token) > 8192:
            raise ValueError("Invalid token length")
        claims = jwt.decode(
            token, key, algorithms=["RS256"], issuer=settings.auth_issuer,
            audience=settings.auth_audience,
            options={"require": ["exp", "iat", "sub", "iss", "aud"]},
        )
        subject = claims["sub"]
        if not isinstance(subject, str) or not subject.strip() or len(subject) > 255:
            raise ValueError("Invalid subject")
        if any(isinstance(claims[k], bool) or not isinstance(claims[k], (int, float)) for k in ("iat", "exp")):
            raise ValueError("Invalid token timestamps")
        if not 0 < claims["exp"] - claims["iat"] <= settings.auth_max_token_lifetime_seconds:
            raise ValueError("Invalid token lifetime")
        identity = json.dumps([claims["iss"], subject], separators=(",", ":"), ensure_ascii=False).encode()
    except (jwt.InvalidTokenError, ValueError, TypeError, KeyError):
        raise HTTPException(401, "Invalid or expired credentials", headers={"WWW-Authenticate": "Bearer"}) from None
    return Principal(user_id=hashlib.sha256(identity).hexdigest())


async def require_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    settings: Settings = Depends(get_settings),
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(401, "Bearer credentials required", headers={"WWW-Authenticate": "Bearer"})
    principal = verify_token(credentials.credentials, settings)
    from .db import consume_rate_limit
    if not await consume_rate_limit(principal.user_id):
        raise HTTPException(429, "Request budget reached", headers={"Retry-After": "60"})
    return principal
