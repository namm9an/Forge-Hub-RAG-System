"""Bounded application regressions; proxy/identity-provider assurance is tracked separately."""
import base64
import hashlib
import hmac
import json
import time

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import HTTPException, UploadFile

from app.auth import verify_token
from app.config import Settings, get_settings


@pytest.fixture
def identity():
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo).decode()
    settings = Settings(auth_public_key=public, auth_issuer="security-test-issuer", auth_audience="forge-api")
    def claims(subject="owner", **extra):
        now = int(time.time())
        return dict(sub=subject, iss=settings.auth_issuer, aud=settings.auth_audience, exp=now+300, iat=now, **extra)
    def sign(subject="owner", **extra):
        return jwt.encode(claims(subject, **extra), key, algorithm="RS256")
    return key, settings, claims, sign


@pytest.mark.parametrize("length", [51, 52, 53, 72, 73])
def test_long_subject_does_not_truncate_or_bypass_signature(identity, length):
    _, settings, _, sign = identity
    first = verify_token(sign("x"*length+"a"), settings)
    second = verify_token(sign("x"*length+"b"), settings)
    assert first != second
    # A display/login name alone cannot replace signed credentials.
    with pytest.raises(HTTPException) as error:
        verify_token("x"*length, settings)
    assert error.value.status_code == 401


@pytest.mark.parametrize("first,second", [("alice", "\u0430lice"), ("jos\u00e9", "jose\u0301"), ("\u00e9"*40+"a", "\u00e9"*40+"b")])
def test_unicode_subjects_cannot_clone_another_opaque_identity(identity, first, second):
    _, settings, _, sign = identity
    assert verify_token(sign(first), settings) != verify_token(sign(second), settings)


def test_email_claim_cannot_replace_signed_subject(identity):
    _, settings, _, sign = identity
    assert verify_token(sign("attacker", email="victim@example.test"), settings) != verify_token(sign("victim", email="victim@example.test"), settings)
    assert verify_token(sign("victim", email="old@example.test"), settings) == verify_token(sign("victim", email="new@example.test"), settings)


@pytest.mark.parametrize("attack", ["none", "public_key_hmac", "wrong_rsa_key", "wrong_algorithm", "tamper"])
def test_signature_verification_rejects_algorithm_confusion(identity, attack):
    key, settings, claims, sign = identity
    if attack == "none":
        token = jwt.encode(claims(), "", algorithm="none")
    elif attack == "public_key_hmac":
        # Real confusion construction: HMAC signed using the verifier's public DER key as the secret.
        secret = key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
        def encode(value):
            return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=")
        signed = encode({"alg": "HS256", "typ": "JWT"}) + b"." + encode(claims())
        token = (signed + b"." + base64.urlsafe_b64encode(hmac.new(secret, signed, hashlib.sha256).digest()).rstrip(b"=")).decode()
    elif attack == "wrong_rsa_key":
        wrong = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        token = jwt.encode(claims(), wrong, algorithm="RS256")
    elif attack == "wrong_algorithm":
        token = jwt.encode(claims(), key, algorithm="RS512")
    else:
        segments = sign().split(".")
        changed = claims("victim")
        segments[1] = base64.urlsafe_b64encode(json.dumps(changed).encode()).rstrip(b"=").decode()
        token = ".".join(segments)
    with pytest.raises(HTTPException) as error:
        verify_token(token, settings)
    assert error.value.status_code == 401
    assert error.value.detail == "Invalid or expired credentials"


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), True, "not-a-time"])
def test_invalid_numeric_dates_have_generic_authentication_failure(identity, value):
    key, settings, claims, _ = identity
    payload = claims()
    payload["exp"] = value
    token = jwt.encode(payload, key, algorithm="RS256")
    with pytest.raises(HTTPException) as error:
        verify_token(token, settings)
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_spoofed_ip_and_email_headers_cannot_bypass_user_rate_budget(identity, monkeypatch):
    from app import app, db
    _, settings, _, sign = identity
    settings.requests_per_minute = 2
    app.dependency_overrides[get_settings] = lambda: settings
    monkeypatch.setattr(db, "get_settings", lambda: settings)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        spoof = {"X-User-ID": "victim", "X-User-Email": "victim@example.test", "X-Forwarded-For": "127.0.0.1"}
        assert (await client.get("/api/stats", headers=spoof)).status_code == 401
        for address in ("127.0.0.1", "192.0.2.1", "198.51.100.2"):
            response = await client.get("/api/stats", headers={"Authorization": "Bearer "+sign(), "X-Forwarded-For": address})
        assert response.status_code == 429


@pytest.mark.asyncio
async def test_authenticated_responses_are_not_cacheable(identity):
    from app import app
    _, settings, _, sign = identity
    app.dependency_overrides[get_settings] = lambda: settings
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/stats", headers={"Authorization": "Bearer "+sign(), "X-Forwarded-Host": "attacker.test"})
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert "attacker.test" not in response.text


@pytest.mark.parametrize("filename", ["safe\u202ename.pdf", "safe\u2066name.pdf", "safe\u200bname.pdf"])
def test_invisible_filename_controls_cannot_disguise_source_names(filename):
    from app.documents import _validate_file_metadata
    from starlette.datastructures import Headers
    file = UploadFile(filename=filename, file=None, headers=Headers({"content-type": "application/pdf"}))
    with pytest.raises(HTTPException) as error:
        _validate_file_metadata(file)
    assert error.value.status_code == 400


def test_unsupported_auth_and_interpreter_features_are_not_exposed():
    from app import app
    from starlette.routing import WebSocketRoute
    paths = {route.path.lower() for route in app.routes if hasattr(route, "path")}
    assert not any(any(part in path for part in ("saml", "graphql", "ldap", "otp", "password", "email")) for path in paths)
    assert not any(isinstance(route, WebSocketRoute) for route in app.routes)


def test_deeply_nested_unsigned_json_cannot_raise_a_server_error(identity):
    _, settings, _, _ = identity
    header = b'{"alg":"RS256","padding":' + b'['*1200 + b'0' + b']'*1200 + b'}'
    token = base64.urlsafe_b64encode(header).rstrip(b'=').decode() + '.e30.c2ln'
    with pytest.raises(HTTPException) as error:
        verify_token(token, settings)
    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_xml_disguised_as_pdf_is_rejected_before_parser_or_model(monkeypatch, tmp_path):
    from app import app, db, documents
    from app.auth import Principal, require_user
    app.dependency_overrides[require_user] = lambda: Principal("xml-test-owner")
    def forbidden(*args, **kwargs):
        raise AssertionError("XML must never reach PDF parsing or inference")
    monkeypatch.setattr(documents, "_extract_pdf", forbidden)
    monkeypatch.setattr(documents, "get_model_async", forbidden)
    canary = tmp_path / "synthetic-canary.txt"
    canary.write_text("synthetic-xxe-canary", encoding="utf-8")
    payload = ('<?xml version="1.0"?><!DOCTYPE x [<!ENTITY leak SYSTEM "' + canary.as_uri() + '">]><x>&leak;</x>').encode()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        response = await client.post("/api/documents/upload", params={"session_id": "xml-test-session"},
            files={"files": ("disguised.pdf", payload, "application/pdf")})
    assert response.status_code == 422
    assert "synthetic-xxe-canary" not in response.text
    assert not db._memory_store.documents
    assert not db._memory_store.sessions


@pytest.mark.asyncio
async def test_cross_origin_preflight_and_cookies_do_not_create_authentication():
    from app import app
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
        preflight = await client.options("/api/chat/stream", headers={"Origin": "https://attacker.test",
            "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "authorization"})
        assert preflight.status_code == 400
        assert "access-control-allow-origin" not in preflight.headers
        response = await client.get("/api/stats", headers={"Origin": "https://attacker.test", "Cookie": "session_id=victim; user=victim"})
    assert response.status_code == 401
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("scope", ["\u0430lice", "\u0065\u0301", "user\u202ename", "name\u200b"])
@pytest.mark.asyncio
async def test_unicode_session_substitutions_are_rejected(scope):
    from app.sessions import authorize_session
    with pytest.raises(HTTPException) as error:
        await authorize_session(scope, "owner", create=True)
    assert error.value.status_code == 422


def test_json_metaproperties_cannot_assign_owner_or_runtime_configuration():
    from app.schemas import ChatRequest
    request = ChatRequest.model_validate({"query": "{{7*7}} __import__('os')", "session_id": "safe-session",
        "user_id": "victim", "__proto__": {"admin": True}, "constructor": {"prototype": {"admin": True}},
        "llm_provider": "attacker", "auth_public_key": "attacker"})
    assert request.query == "{{7*7}} __import__('os')"
    assert set(request.model_dump()).isdisjoint({"user_id", "__proto__", "constructor", "llm_provider", "auth_public_key"})


def test_unclosed_citation_marker_storm_is_rejected():
    from app.chat import valid_answer_citations
    from app.schemas import Source
    source = Source(id="chunk", filename="safe.pdf", page_number=1)
    assert not valid_answer_citations("[Doc:"*12000, [source])


@pytest.mark.asyncio
async def test_production_host_allowlist_rejects_poisoned_host(monkeypatch):
    import app.main as main
    settings = Settings(environment="production", allowed_hosts="api.forge.test")
    monkeypatch.setattr(main, "get_settings", lambda: settings)
    application = main.create_app()
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="https://api.forge.test") as client:
        assert (await client.get("/health")).status_code == 200
        response = await client.get("/health", headers={"Host": "attacker.test", "X-Forwarded-Host": "api.forge.test"})
    assert response.status_code == 400
