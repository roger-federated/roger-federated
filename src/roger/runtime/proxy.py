"""proxy.py — transparent reverse proxy in front of a local OpenAI-compatible runtime server.

Runtimes never log message bodies, so the only provider-agnostic way to see the chats flowing through
them is to sit on the wire: dialect.plan() moves the real server to an ephemeral loopback port, and
`start()` listens on the port the user asked for, relaying every request byte-for-byte. Clients keep
talking to the usual address and never notice. Nothing here knows which runtime is behind it (that is
all in dialect.py).
"""
import sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Iterable

import httpx

from roger.runtime import capture

# Per RFC 7230 these describe the connection they arrived on, so forwarding them would lie to the
# other side. `expect` is here because the stdlib handler already answered 100-continue on our behalf
# (curl sends it for every long chat), and httpx can't honour it a second time.
_HOP_BY_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                         "te", "trailer", "transfer-encoding", "upgrade", "expect"})
_CHUNK = 64 * 1024
# The runtime usually needs a moment to bind after we spawn it; the first client request tends to
# arrive before that (`roger llama-server …` + an IDE plugin that polls). Short, bounded retries.
_CONNECT_RETRIES, _CONNECT_BACKOFF = 8, 0.25


# ---------------------------------------------------------------------------
# The proxy server
# ---------------------------------------------------------------------------

# on_exchange(path, request_body, response_content_type, status, response_body) — called once a chat
# request has been fully relayed with a 2xx, from the connection's own thread.
Sink = Callable[[str, bytes, str, int, bytes], None]


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second listener share a port that's already bound, which would
    # silently leave an already-running runtime answering clients instead of us. Fail loudly there.
    allow_reuse_address = sys.platform != "win32"

    def __init__(self, addr: tuple[str, int], backend: str, on_exchange: Sink | None):
        super().__init__(addr, _Handler)
        self.backend = backend.rstrip("/")
        self.on_exchange = on_exchange
        # One pooled client for every connection thread. No read/write timeout: a completion can
        # legitimately take minutes to produce its first byte on a small GPU.
        self.client = httpx.Client(timeout=httpx.Timeout(connect=5.0, read=None, write=None, pool=None))


def start(host: str, port: int, backend_url: str, on_exchange: Sink | None) -> ThreadingHTTPServer:
    """Bind (raises OSError when the port is taken) but don't serve yet — run `serve_forever` in a thread."""
    return _Server((host, port), backend_url, on_exchange)


def stop(server: ThreadingHTTPServer) -> None:
    server.shutdown()
    server.server_close()
    server.client.close()


def _read_n(rfile, n: int) -> Iterable[bytes]:
    """Stream a Content-Length body in chunks so multi-GB uploads (model blobs) never sit in memory."""
    while n > 0:
        chunk = rfile.read(min(n, _CHUNK))
        if not chunk:
            return
        n -= len(chunk)
        yield chunk


def _iter_chunked(rfile) -> Iterable[bytes]:
    """Decode a `Transfer-Encoding: chunked` request body; the stdlib handler leaves the framing to us."""
    while True:
        size = int(rfile.readline().split(b";")[0].strip() or b"0", 16)
        if size == 0:
            while rfile.readline() not in (b"\r\n", b"\n", b""):   # swallow trailers
                pass
            return
        yield rfile.read(size)
        rfile.readline()   # CRLF terminating the chunk payload


def _tee(chunks: Iterable[bytes], buf: bytearray) -> Iterable[bytes]:
    for c in chunks:
        buf += c
        yield c


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"   # keep-alive + chunked responses, which streaming clients expect

    def log_message(self, *_):      # the runtime already logs its own requests; don't double up
        pass

    def _forward(self):
        srv: _Server = self.server
        if self.headers.get("Upgrade"):
            # Websocket tunnelling through BaseHTTPRequestHandler isn't worth it: none of the
            # OpenAI-style chat APIs use it.
            self.send_error(501, "roger: protocol upgrades are not relayed")
            self.close_connection = True
            return

        length = self.headers.get("Content-Length")
        if length is not None:
            body = _read_n(self.rfile, int(length))
        elif "chunked" in self.headers.get("Transfer-Encoding", "").lower():
            body = _iter_chunked(self.rfile)
        else:
            body = None
        want = srv.on_exchange is not None and capture.wants(self.command, self.path)
        req_buf = bytearray()
        if want and body is not None:
            body = _tee(body, req_buf)   # chat bodies are small; we need the JSON anyway

        # Drop Host so httpx sets the loopback one the runtime expects; keep an explicit
        # Content-Length (httpx then sends the iterator body as-is instead of re-chunking it).
        headers = [(k, v) for k, v in self.headers.items()
                   if k.lower() not in _HOP_BY_HOP and k.lower() not in ("host", "accept-encoding")]
        # Capture needs plain text, and on loopback compression buys nothing — but never impose
        # an encoding the client didn't ask for on paths we don't read.
        headers.append(("Accept-Encoding", "identity" if want else self.headers.get("Accept-Encoding", "identity")))

        req = srv.client.build_request(self.command, srv.backend + self.path, headers=headers, content=body)
        resp = None
        for attempt in range(_CONNECT_RETRIES):
            try:
                resp = srv.client.send(req, stream=True)
                break
            except httpx.ConnectError:
                # Safe to retry: nothing of the body iterator is consumed before the connect succeeds.
                if attempt == _CONNECT_RETRIES - 1:
                    self.send_error(502, f"roger: runtime at {srv.backend} is not accepting connections yet")
                    self.close_connection = True
                    return
                time.sleep(_CONNECT_BACKOFF)
            except httpx.HTTPError as e:
                self.send_error(502, f"roger: {e.__class__.__name__} talking to {srv.backend}")
                self.close_connection = True
                return

        sent_headers = False
        try:
            self.send_response_only(resp.status_code)   # not send_response: upstream supplies Date/Server
            for k, v in resp.headers.multi_items():
                if k.lower() not in _HOP_BY_HOP:
                    self.send_header(k, v)
            no_body = self.command == "HEAD" or resp.status_code < 200 or resp.status_code in (204, 304)
            chunked = not no_body and "content-length" not in resp.headers
            if chunked:
                self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            sent_headers = True
            resp_buf = bytearray()
            if not no_body:
                # iter_raw, not iter_bytes: relay exactly what upstream sent so Content-Encoding and
                # Content-Length stay truthful. Flush per chunk so SSE tokens leave as they arrive.
                for chunk in resp.iter_raw():
                    if not chunk:
                        continue
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk) if chunked else chunk)
                    self.wfile.flush()
                    if want:
                        resp_buf += chunk
                if chunked:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            if want and 200 <= resp.status_code < 300:
                try:
                    srv.on_exchange(self.path, bytes(req_buf), resp.headers.get("content-type", ""),
                                    resp.status_code, bytes(resp_buf))
                except Exception as e:   # capture must never break relaying
                    print(f"roger: failed to save exchange: {e!r}", file=sys.stderr)
        except (BrokenPipeError, ConnectionResetError, httpx.HTTPError) as e:
            # Client went away mid-stream, or upstream died. Closing the upstream response below
            # cancels the runtime's generation; closing our side keeps a half-relayed body from
            # corrupting the next keep-alive request.
            if not sent_headers:
                self.send_error(502, f"roger: {e.__class__.__name__} talking to {srv.backend}")
            self.close_connection = True
        finally:
            resp.close()

    do_GET = do_POST = do_PUT = do_PATCH = do_DELETE = do_HEAD = do_OPTIONS = _forward
