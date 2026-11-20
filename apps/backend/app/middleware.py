"""Bound ingress before multipart parsing; reject overload instead of an unbounded queue."""
import asyncio
import tempfile
from time import monotonic

from starlette.responses import JSONResponse


class RequestLimits:
    def __init__(self, app, max_body_bytes: int, max_inflight: int):
        self.app, self.max_body_bytes, self.max_inflight = app, max_body_bytes, max_inflight
        self.inflight = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        original_send = send

        async def send(message):
            if message["type"] == "http.response.start":
                message = dict(message)
                response_headers = [(name, value) for name, value in message.get("headers", [])
                                    if name.lower() not in {b"cache-control", b"x-content-type-options"}]
                response_headers.extend([(b"x-content-type-options", b"nosniff"), (b"cache-control", b"no-store")])
                message["headers"] = response_headers
            await original_send(message)

        async def reject(detail, status, headers=None):
            rejection_headers = dict(headers or {})
            if scope.get("http_version", "1.1").startswith("1."):
                # A framing rejection must not leave the parser free to dispatch trailing bytes.
                rejection_headers["Connection"] = "close"
            return await JSONResponse({"detail": detail}, status, headers=rejection_headers)(scope, receive, send)

        if self.inflight >= self.max_inflight:
            return await reject("Service capacity reached", 503, {"Retry-After": "1"})
        self.inflight += 1
        try:
            header_pairs = [(name.lower(), value) for name, value in scope.get("headers", [])]
            names = [name for name, _ in header_pairs]
            if (any(names.count(name) > 1 for name in (b"content-length", b"transfer-encoding", b"authorization", b"host"))
                    or (b"content-length" in names and b"transfer-encoding" in names)):
                return await reject("Ambiguous request headers", 400)
            headers = dict(header_pairs)
            transfer = headers.get(b"transfer-encoding")
            if transfer is not None and (transfer.lower() != b"chunked" or scope.get("http_version") == "2"):
                return await reject("Invalid Transfer-Encoding", 400)
            length = headers.get(b"content-length")
            try:
                if length is not None and (not length.isdigit() or len(length) > 20):
                    raise ValueError()
                declared_size = int(length) if length is not None else 0
                if declared_size < 0:
                    raise ValueError()
            except ValueError:
                return await reject("Invalid Content-Length", 400)
            if declared_size > self.max_body_bytes:
                return await reject("Request body limit exceeded", 413)
            # Spool uploads to disk rather than retaining a body-sized buffer per active request.
            with tempfile.SpooledTemporaryFile(max_size=1024 * 1024) as body:
                size = 0
                deadline = monotonic() + 30
                while True:
                    try:
                        message = await asyncio.wait_for(receive(), timeout=max(0, deadline-monotonic()))
                    except asyncio.TimeoutError:
                        return await reject("Request body timed out", 408)
                    if message["type"] == "http.disconnect":
                        return
                    chunk = message.get("body", b"")
                    size += len(chunk)
                    if size > self.max_body_bytes:
                        return await reject("Request body limit exceeded", 413)
                    body.write(chunk)
                    if not message.get("more_body", False):
                        break
                if length is not None and declared_size != size:
                    return await reject("Request body length mismatch", 400)
                body.seek(0)
                remaining = size
                consumed = False

                async def replay():
                    nonlocal remaining, consumed
                    if consumed:
                        return await receive()  # Preserve disconnect notification for StreamingResponse.
                    data = body.read(min(65536, remaining))
                    remaining -= len(data)
                    consumed = remaining == 0
                    return {"type": "http.request", "body": data, "more_body": not consumed}

                await self.app(scope, replay, send)
        finally:
            self.inflight -= 1
