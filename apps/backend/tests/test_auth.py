"""Authentication checks use ephemeral RSA keys, never configured application secrets."""
import time

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
@pytest.mark.parametrize("method,path", [
    ("GET", "/api/documents"), ("GET", "/api/stats"),
    ("POST", "/api/sessions"), ("POST", "/api/embeddings/generate"),
])
async def test_private_endpoints_require_credentials(method, path):
    from app import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.request(method, path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


@pytest.fixture
def signing_keys():
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives import serialization
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key, key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    ).decode()


@pytest.mark.parametrize("mutation", ["expired", "audience", "issuer", "missing_exp", "future", "algorithm", "subject", "lifetime"])
def test_rejects_invalid_tokens(signing_keys, mutation):
    import jwt
    from fastapi import HTTPException
    from app.auth import verify_token
    from app.config import Settings
    key, public = signing_keys
    settings = Settings(auth_public_key=public, auth_issuer="https://issuer.test", auth_audience="forge")
    now = int(time.time())
    claims = dict(sub="user-a", iss=settings.auth_issuer, aud="forge", exp=now + 300, iat=now)
    if mutation == "expired": claims["exp"] = now - 1
    if mutation == "audience": claims["aud"] = "another-api"
    if mutation == "issuer": claims["iss"] = "https://attacker.test"
    if mutation == "missing_exp": del claims["exp"]
    if mutation == "future": claims["iat"] = now + 600
    if mutation == "subject": claims["sub"] = ""
    if mutation == "lifetime": claims["exp"] = now + 100000
    token = jwt.encode(claims, "attacker-controlled-symmetric-key-32-bytes" if mutation == "algorithm" else key,
                       algorithm="HS256" if mutation == "algorithm" else "RS256")
    with pytest.raises(HTTPException) as exc:
        verify_token(token, settings)
    assert exc.value.status_code == 401


def test_verified_identity_is_bound_to_issuer_and_subject(signing_keys):
    import jwt
    from app.auth import verify_token
    from app.config import Settings
    key, public = signing_keys
    now = int(time.time())
    cfg = dict(auth_public_key=public, auth_audience="forge")
    def principal(issuer, sub):
        claims = dict(sub=sub, iss=issuer, aud="forge", iat=now, exp=now+300)
        return verify_token(jwt.encode(claims, key, algorithm="RS256"), Settings(auth_issuer=issuer, **cfg))
    assert principal("one", "alice") == principal("one", "alice")
    assert principal("one", "alice") != principal("two", "alice")
    assert principal("one", "alice") != principal("one", "bob")


def test_invalid_unicode_in_decoded_subject_is_rejected(signing_keys, monkeypatch):
    import jwt
    from fastapi import HTTPException
    from app.auth import verify_token
    from app.config import Settings

    _, public = signing_keys
    settings = Settings(auth_public_key=public, auth_issuer="https://issuer.test", auth_audience="forge")
    claims = {"iss": settings.auth_issuer, "aud": settings.auth_audience, "sub": "\ud800", "iat": 100, "exp": 400}
    monkeypatch.setattr(jwt, "decode", lambda *args, **kwargs: claims)

    with pytest.raises(HTTPException) as error:
        verify_token("opaque", settings)
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_private_sse_refusal_has_allowed_origin_and_no_store(signing_keys):
    import jwt
    from app import app
    from app.config import Settings, get_settings

    key, public = signing_keys
    settings = Settings(auth_public_key=public, auth_issuer="https://issuer.test", auth_audience="forge")
    app.dependency_overrides[get_settings] = lambda: settings
    now = int(time.time())
    claims = dict(sub="test-user", iss=settings.auth_issuer, aud=settings.auth_audience, iat=now, exp=now + 300)
    token = jwt.encode(claims, key, algorithm="RS256")
    origin = settings.allowed_origins
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post(
            "/api/chat/stream",
            headers={"Authorization": f"Bearer {token}", "Origin": origin, "Accept": "text/event-stream"},
            json={"session_id": "empty-session", "query": "What does the manual say?"},
        )

    assert response.status_code == 200
    assert response.headers.get_list("cache-control") == ["no-store"]
    assert response.headers["access-control-allow-origin"] == origin
    assert "event: complete" in response.text
    assert '"refusal": true' in response.text


def test_missing_auth_configuration_fails_closed():
    from fastapi import HTTPException
    from app.auth import verify_token
    from app.config import Settings
    with pytest.raises(HTTPException) as exc:
        verify_token("opaque", Settings())
    assert exc.value.status_code == 503
