import asyncio
import pytest


async def invoke(middleware, chunks, headers=()):
    queue = list(chunks)
    output = []
    async def receive():
        return {"type": "http.request", "body": queue.pop(0), "more_body": bool(queue)}
    async def send(message):
        output.append(message)
    await middleware({"type": "http", "method": "POST", "path": "/api/test", "headers": headers}, receive, send)
    return output


@pytest.mark.asyncio
async def test_actual_body_limit_prevents_parser_execution():
    from app.middleware import RequestLimits
    invoked = []
    async def parser(scope, receive, send):
        invoked.append(True)
    output = await invoke(RequestLimits(parser, max_body_bytes=4, max_inflight=1), [b"123", b"45"])
    assert output[0]["status"] == 413
    assert not invoked


@pytest.mark.asyncio
async def test_small_body_replays_exactly_and_adds_security_headers():
    from app.middleware import RequestLimits
    async def parser(scope, receive, send):
        message = await receive()
        assert message["body"] == b"1234"
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"ok"})
    output = await invoke(RequestLimits(parser, max_body_bytes=4, max_inflight=1), [b"12", b"34"])
    assert output[0]["status"] == 200
    assert (b"x-content-type-options", b"nosniff") in output[0]["headers"]


@pytest.mark.asyncio
async def test_concurrency_limit_rejects_without_queuing_and_recovers():
    from app.middleware import RequestLimits
    entered, release = asyncio.Event(), asyncio.Event()
    async def parser(scope, receive, send):
        entered.set()
        await release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})
    middleware = RequestLimits(parser, max_body_bytes=4, max_inflight=1)
    first = asyncio.create_task(invoke(middleware, [b""]))
    await entered.wait()
    second = await invoke(middleware, [b""])
    assert second[0]["status"] == 503
    release.set()
    await first
    assert (await invoke(middleware, [b""]))[0]["status"] == 200


@pytest.mark.asyncio
@pytest.mark.parametrize("headers,chunks", [
    ([(b"content-length", b"1"), (b"content-length", b"2")], [b"x"]),
    ([(b"content-length", b"1"), (b"transfer-encoding", b"chunked")], [b"x"]),
    ([(b"content-length", b"+1")], [b"x"]),
    ([(b"content-length", b" 1")], [b"x"]),
    ([(b"content-length", b"2")], [b"x"]),
    ([(b"authorization", b"Bearer one"), (b"authorization", b"Bearer two")], [b""]),
])
async def test_ambiguous_framing_and_identity_headers_are_rejected_before_handler(headers, chunks):
    from app.middleware import RequestLimits
    called = []
    async def handler(scope, receive, send):
        called.append(True)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b""})
    output = await invoke(RequestLimits(handler, max_body_bytes=10, max_inflight=1), chunks, headers)
    assert output[0]["status"] == 400
    assert not called


@pytest.mark.asyncio
async def test_early_limit_failures_are_also_not_cacheable():
    from app.middleware import RequestLimits
    async def handler(scope, receive, send):
        raise AssertionError("must not run")
    output = await invoke(RequestLimits(handler, max_body_bytes=1, max_inflight=1), [b"xx"])
    assert output[0]["status"] == 413
    assert (b"cache-control", b"no-store") in output[0]["headers"]


@pytest.mark.asyncio
async def test_endpoint_cannot_override_protected_response_cache_policy():
    from app.middleware import RequestLimits
    async def handler(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"cache-control", b"public, max-age=86400")]})
        await send({"type": "http.response.body", "body": b"private"})
    output = await invoke(RequestLimits(handler, max_body_bytes=1, max_inflight=1), [b""])
    values = [value for name, value in output[0]["headers"] if name == b"cache-control"]
    assert values == [b"no-store"]
