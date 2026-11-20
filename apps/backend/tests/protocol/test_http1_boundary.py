"""Opt-in raw HTTP/1 checks against an isolated loopback Uvicorn server.

These exercise the selected h11/httptools parser plus RequestLimits. They do
not certify a reverse proxy, TLS termination, HTTP/2, or Rapid Reset defenses.
"""

import json
import socket
import threading
from time import monotonic

import pytest
import uvicorn

from app.middleware import RequestLimits


class LocalHTTPServer:
    def __init__(self, parser):
        self.parser = parser
        self.paths = []
        self.lock = threading.Lock()
        self.errors = []
        self.listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.listener.bind(("127.0.0.1", 0))
        self.listener.listen(16)
        self.listener.setblocking(False)
        self.port = self.listener.getsockname()[1]

        async def handler(scope, receive, send):
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                if not message.get("more_body", False):
                    break
            with self.lock:
                self.paths.append(scope["path"])
            body = json.dumps({"path": scope["path"]}).encode()
            await send({
                "type": "http.response.start", "status": 200,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode()),
                    # The middleware must override an unsafe inner cache policy.
                    (b"cache-control", b"public, max-age=60"),
                ],
            })
            await send({"type": "http.response.body", "body": body})

        self.server = uvicorn.Server(uvicorn.Config(
            RequestLimits(handler, max_body_bytes=1024, max_inflight=2),
            host="127.0.0.1", port=self.port, http=parser, http2=False,
            ws="none", lifespan="off", loop="asyncio", proxy_headers=False,
            access_log=False, log_config=None, log_level="error",
            timeout_keep_alive=1, timeout_graceful_shutdown=1,
        ))

        def run():
            try:
                self.server.run(sockets=[self.listener])
            except BaseException as error:
                self.errors.append(error)

        self.thread = threading.Thread(target=run, name=f"forge-protocol-{parser}", daemon=True)

    def start(self):
        self.thread.start()
        deadline = monotonic() + 5
        while not self.server.started and self.thread.is_alive() and monotonic() < deadline:
            threading.Event().wait(0.01)
        if not self.server.started:
            self.stop()
            pytest.fail(f"Loopback {self.parser} server did not start: {self.errors!r}")

    def stop(self):
        self.server.should_exit = True
        if self.thread.ident is not None:
            self.thread.join(timeout=3)
            if self.thread.is_alive():
                self.server.force_exit = True
                self.thread.join(timeout=2)
        self.listener.close()
        if self.thread.is_alive():
            pytest.fail(f"Loopback {self.parser} server did not stop within its teardown deadline")

    def dispatched(self):
        with self.lock:
            return list(self.paths)

    def exchange(self, payload, allowed_port):
        """Grant only this connection's exact loopback port; restore the guard afterward."""
        token = allowed_port.set(self.port)
        try:
            with socket.create_connection(("127.0.0.1", self.port), timeout=2) as connection:
                connection.sendall(payload)
                response = bytearray()
                deadline = monotonic() + 2
                closed = False
                while monotonic() < deadline:
                    connection.settimeout(max(0.001, deadline - monotonic()))
                    try:
                        chunk = connection.recv(min(8192, 65537 - len(response)))
                    except socket.timeout:
                        break
                    except ConnectionResetError:
                        closed = True
                        break
                    if not chunk:
                        closed = True
                        break
                    response.extend(chunk)
                    if len(response) > 65536:
                        pytest.fail("Loopback response exceeded the 64 KiB test budget")
                return bytes(response), closed
        finally:
            allowed_port.reset(token)


@pytest.fixture(scope="class", params=["h11", "httptools"])
def local_http_server(request):
    server = LocalHTTPServer(request.param)
    server.start()
    try:
        yield server
    finally:
        server.stop()


def response_headers(response):
    """Read the first bounded HTTP response's header fields without hiding duplicates."""
    head = response.split(b"\r\n\r\n", 1)[0]
    return [tuple(part.strip().lower() for part in line.split(b":", 1))
            for line in head.split(b"\r\n")[1:] if b":" in line]


def assert_rejected_without_dispatch(server, response, closed, *paths):
    # A parser may reject by closing without a response; Uvicorn normally emits 400.
    assert closed, "Ambiguous framing must not leave the connection reusable"
    assert not response or response.startswith(b"HTTP/1.1 400"), response[:200]
    for path in paths:
        assert path not in server.dispatched()
        assert path.encode() not in response


@pytest.mark.protocol
class TestHTTP1Boundary:
    def test_normal_pipelined_requests_are_dispatched(self, local_http_server, loopback_protocol_port):
        server = local_http_server
        response, closed = server.exchange(
            b"GET /normal-one HTTP/1.1\r\nHost: localhost\r\n\r\n"
            b"GET /normal-two HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            loopback_protocol_port,
        )
        assert closed
        assert response.count(b"HTTP/1.1 200") == 2
        assert server.dispatched()[-2:] == ["/normal-one", "/normal-two"]
        assert loopback_protocol_port.get() is None

    def test_conflicting_content_lengths_cannot_dispatch_following_request(self, local_http_server, loopback_protocol_port):
        server = local_http_server
        response, closed = server.exchange(
            b"POST /bad-duplicate-cl HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 4\r\nContent-Length: 5\r\n\r\nDATA"
            b"GET /injected-after-duplicate-cl HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            loopback_protocol_port,
        )
        assert_rejected_without_dispatch(server, response, closed, "/bad-duplicate-cl", "/injected-after-duplicate-cl")

    def test_content_length_transfer_encoding_conflict_closes_connection(self, local_http_server, loopback_protocol_port):
        server = local_http_server
        response, closed = server.exchange(
            b"POST /bad-cl-te HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Length: 5\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n"
            b"GET /injected-after-cl-te HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            loopback_protocol_port,
        )
        assert_rejected_without_dispatch(server, response, closed, "/bad-cl-te", "/injected-after-cl-te")

    @pytest.mark.parametrize("length", [b"+4", b"4x", b"-1"])
    def test_invalid_content_length_cannot_dispatch_following_request(self, local_http_server, loopback_protocol_port, length):
        server = local_http_server
        response, closed = server.exchange(
            b"POST /bad-cl-syntax HTTP/1.1\r\nHost: localhost\r\nContent-Length: " + length + b"\r\n\r\nDATA"
            b"GET /injected-after-cl-syntax HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            loopback_protocol_port,
        )
        assert_rejected_without_dispatch(server, response, closed, "/bad-cl-syntax", "/injected-after-cl-syntax")

    def test_application_responses_have_one_no_store_policy(self, local_http_server, loopback_protocol_port):
        response, closed = local_http_server.exchange(
            b"GET /cache-policy HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n",
            loopback_protocol_port,
        )
        assert closed
        assert response.startswith(b"HTTP/1.1 200")
        headers = response_headers(response)
        assert [value for name, value in headers if name == b"cache-control"] == [b"no-store"]
        assert (b"x-content-type-options", b"nosniff") in headers

    def test_application_rejection_retains_no_store_policy(self, local_http_server, loopback_protocol_port):
        server = local_http_server
        response, closed = server.exchange(
            b"POST /body-too-large HTTP/1.1\r\nHost: localhost\r\nContent-Length: 1025\r\nConnection: close\r\n\r\n",
            loopback_protocol_port,
        )
        assert closed
        assert response.startswith(b"HTTP/1.1 413")
        assert (b"cache-control", b"no-store") in response_headers(response)
        assert "/body-too-large" not in server.dispatched()

    def test_parser_errors_are_a_distinct_header_boundary(self, local_http_server, loopback_protocol_port, record_property):
        server = local_http_server
        response, closed = server.exchange(
            b"POST /parser-error HTTP/1.1\r\nHost: localhost\r\nContent-Length: invalid\r\n\r\n",
            loopback_protocol_port,
        )
        assert_rejected_without_dispatch(server, response, closed, "/parser-error")
        # The HTTP parser can generate 400 before ASGI middleware runs. Record
        # its observed policy rather than pretending application headers cover it.
        policy = [value.decode("ascii", "replace") for name, value in response_headers(response) if name == b"cache-control"]
        record_property("http_parser", server.parser)
        record_property("parser_error_cache_control", policy or ["not supplied by parser"])
