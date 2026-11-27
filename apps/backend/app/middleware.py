"""Limit concurrent ingress and admission wait before parsing request bodies."""
import asyncio
import tempfile
import weakref
from time import monotonic

from starlette.responses import JSONResponse


ADMISSION_WAIT_SECONDS = 2


class _Admission:
    def __init__(self, capacity):
        self.slots = asyncio.BoundedSemaphore(capacity)
        self.queue_capacity = 2 * capacity
        self.waiting = 0

    async def acquire(self):
        # Bound the number of waiting connections as well as their wait.
        if self.slots.locked() and self.waiting >= self.queue_capacity:
            return False
        self.waiting += 1
        try:
            async with asyncio.timeout(ADMISSION_WAIT_SECONDS):
                await self.slots.acquire()
            return True
        except TimeoutError:
            return False
        finally:
            self.waiting -= 1


class RequestLimits:
    def __init__(self, app, max_body_bytes: int, max_inflight: int, allowed_origins=()):
        self.app, self.max_body_bytes, self.max_inflight = app, max_body_bytes, max_inflight
        self._admissions = weakref.WeakKeyDictionary()
        self.allowed_origins = frozenset(allowed_origins)

    def _loop_admission(self):
        loop = asyncio.get_running_loop()
        reference = self._admissions.get(loop)
        admission = reference() if reference else None
        if admission is None:
            admission = _Admission(self.max_inflight)
            # A quiescent middleware need not retain a stopped loop.
            self._admissions[loop] = weakref.ref(admission)
        return admission

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

        origin = headers.get(b"origin", b"").decode("latin-1")
        if (scope.get("method") == "OPTIONS" and origin and b"access-control-request-method" in headers
                and transfer is None and declared_size == 0):
            # CORS policy and host validation still run before the browser can send its actual request.
            return await self.app(scope, receive, send)

        admission = self._loop_admission()
        if not await admission.acquire():
            retry = {"Retry-After": "1"}
            if origin and "*" in self.allowed_origins:
                retry.update({"Access-Control-Allow-Origin": "*", "Access-Control-Expose-Headers": "Retry-After"})
            elif origin in self.allowed_origins:
                retry.update({"Access-Control-Allow-Origin": origin, "Access-Control-Expose-Headers": "Retry-After",
                              "Vary": "Origin"})
            return await reject("Service capacity reached", 429, retry)
        try:
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
            admission.slots.release()
