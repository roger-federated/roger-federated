"""Lightweight MCP client adapter — bridges external MCP tool servers to
rollout()'s existing tools/tool_handlers interface without modifying the rollout loop.

Servers are declared in ~/.roger/mcp.json using the standard `mcpServers` schema
(copy-paste compatible with Claude Desktop / Cursor / Claude Code), e.g.:
    {"mcpServers": {
        "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
        "sentry":     {"type": "http", "url": "https://mcp.sentry.dev/mcp",
                       "headers": {"Authorization": "Bearer ..."}},
        "gmail":      {"serverUrl": "https://gmailmcp.googleapis.com/mcp/v1",
                       "oauth": {"clientId": "...", "clientSecret": "..."}}
    }}
Each entry is either a local stdio subprocess (`command`/`args`/`env`) or a remote server
(`url` (alias `serverUrl`) + optional `type` "sse"|"http" and `headers`). A remote server may
also carry an `oauth` block (`{clientId, clientSecret}`, the format Google's Gmail MCP docs
use): on first connect we run a browser-based Authorization Code + PKCE login and cache the
resulting tokens under ~/.roger/oauth/<server>.json (auto-refreshed), so later runs don't
re-prompt. The OAuth client must be registered with a loopback redirect URI (Google: a
"Desktop app" client, which allows any 127.0.0.1 port). Tools are namespaced
`mcp__<server>__<tool>` so same-named tools from different servers can't collide.

Usage with rollout():
    stack, tools, handlers = await connect_servers(load_mcp_config())
    try:
        await rollout(..., tools=tools, tool_handlers=handlers)
    finally:
        await stack.aclose()
"""

import contextlib, subprocess, base64, io, json, os, warnings
import asyncio, http.server, threading, urllib.parse, webbrowser
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from PIL import Image
from roger.agency.path_utils import state_dir


def _audio(b64: str):
    "base64 audio → pydub.AudioSegment (normalised to an array downstream by _parse_result), or None."
    try:
        from pydub import AudioSegment
        return AudioSegment.from_file(io.BytesIO(base64.b64decode(b64)))
    except Exception:   # pydub/ffmpeg missing or payload not decodable
        return None


def _mcp_block(c):
    """One MCP content block → PIL.Image | pydub.AudioSegment | str. Images/audio become media
    objects that _parse_result turns into model inputs; everything else (binary resources,
    resource_links, unknown types) degrades to a compact text placeholder, not a base64 blob."""
    t = getattr(c, "type", None)
    if t == "text":
        return c.text
    if t == "image":
        return Image.open(io.BytesIO(base64.b64decode(c.data)))
    if t == "audio":
        return _audio(c.data) or f"[audio {getattr(c, 'mimeType', '') or ''}]".strip()
    if t == "resource":   # EmbeddedResource: text or base64 blob (+ mimeType)
        r = c.resource
        mime, blob = str(getattr(r, "mimeType", "") or ""), getattr(r, "blob", None)
        if blob is not None and mime.startswith("image/"):
            return Image.open(io.BytesIO(base64.b64decode(blob)))
        if blob is not None and mime.startswith("audio/"):
            return _audio(blob) or f"[resource {getattr(r, 'uri', '') or ''} {mime}]".strip()
        text = getattr(r, "text", None)
        return text if text is not None else f"[resource {getattr(r, 'uri', '') or ''} {mime}]".strip()
    return f"[{t} {getattr(c, 'uri', '') or ''}]".strip()   # resource_link / unknown


def _strip_schema(schema: dict) -> dict:
    """Remove meta-fields ($schema, additionalProperties, etc.) that waste tokens
    in the prompt without helping the model produce valid tool calls."""
    skip = {"$schema", "additionalProperties"}
    out = {}
    for k, v in schema.items():
        if k in skip:
            continue
        if isinstance(v, dict):
            v = _strip_schema(v)
        elif isinstance(v, list):
            v = [_strip_schema(i) if isinstance(i, dict) else i for i in v]
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# OAuth 2.0 for remote servers — we wire up the mcp SDK's client-side flow
# (mcp.client.auth.OAuthClientProvider does Authorization Code + PKCE, metadata
# discovery and silent refresh) rather than implementing any OAuth crypto here.
# ---------------------------------------------------------------------------

def _remote_url(spec: dict) -> str:
    "Remote server URL, accepting Google's `serverUrl` alias for `url`."
    return spec.get("url") or spec["serverUrl"]


def _parse_callback(path: str) -> tuple[str | None, str | None]:
    "Pull (code, state) out of the OAuth redirect request path's query string."
    params = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
    return params.get("code", [None])[0], params.get("state", [None])[0]


def _oauth_cache_path(server_name: str) -> str:
    "Per-server token cache file ~/.roger/oauth/<name>.json (name sanitised for the filesystem)."
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in server_name)
    return os.path.join(state_dir(), "oauth", f"{safe}.json")


def _load_cache(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_cache(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


class _FileTokenStorage:
    """Thin adapter exposing the SDK's TokenStorage protocol (four awaited methods that must
    travel together as one object, one instance per server) over the plain _load_cache/_save_cache
    file functions. Persists to ~/.roger/oauth/<name>.json. The SDK calls set_tokens() after the
    first login *and* after every silent refresh, so the freshest token is always on disk.
    get_client_info() seeds the user-supplied clientId/clientSecret on first use, which makes
    OAuthClientProvider skip Dynamic Client Registration and use those pre-registered credentials
    (the Gmail path). Methods are async only because the protocol awaits them; the IO is sync."""

    def __init__(self, server_name: str, oauth: dict, redirect_uri: str):
        self._path = _oauth_cache_path(server_name)
        self._oauth = oauth
        self._redirect_uri = redirect_uri

    async def get_tokens(self) -> OAuthToken | None:
        tok = _load_cache(self._path).get("tokens")
        return OAuthToken.model_validate(tok) if tok else None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = _load_cache(self._path)
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        _save_cache(self._path, data)

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        info = _load_cache(self._path).get("client_info")
        if info:
            return OAuthClientInformationFull.model_validate(info)
        # No persisted registration yet: hand back the spec's pre-registered credentials so the
        # SDK uses them directly instead of attempting Dynamic Client Registration.
        return OAuthClientInformationFull(
            client_id=self._oauth["clientId"],
            client_secret=self._oauth.get("clientSecret"),
            redirect_uris=[self._redirect_uri],
            grant_types=["authorization_code", "refresh_token"],
            token_endpoint_auth_method="client_secret_post",
        )

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = _load_cache(self._path)
        data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
        _save_cache(self._path, data)


def _loopback_oauth_handlers(stack: contextlib.AsyncExitStack):
    """Bind a loopback HTTP listener to catch the OAuth redirect and return
    (redirect_uri, redirect_handler, callback_handler) for OAuthClientProvider.
    redirect_handler opens the system browser at the auth URL; callback_handler blocks until the
    provider's redirect lands on 127.0.0.1 and yields the (code, state). The listening socket is
    torn down via `stack` when the connection's exit stack unwinds."""
    loop = asyncio.get_running_loop()
    result: asyncio.Future = loop.create_future()

    class _Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            code, state = _parse_callback(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>Authentication complete. "
                             b"You may close this tab.</body></html>")
            if not result.done():   # hand the code back to the event-loop thread
                loop.call_soon_threadsafe(result.set_result, (code, state))

        def log_message(self, *a):  # silence BaseHTTPRequestHandler's stderr request logging
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), _Handler)
    redirect_uri = f"http://127.0.0.1:{server.server_address[1]}/callback"
    threading.Thread(target=server.serve_forever, daemon=True).start()
    stack.callback(lambda: (server.shutdown(), server.server_close()))

    async def redirect_handler(authorization_url: str) -> None:
        # Also print the URL: webbrowser.open() is a no-op on headless/remote boxes.
        print(f"\nOpening browser to authorise MCP access:\n{authorization_url}\n")
        webbrowser.open(authorization_url)

    async def callback_handler() -> tuple[str, str | None]:
        code, state = await result
        if not code:
            raise RuntimeError("OAuth redirect did not include an authorization code")
        return code, state

    return redirect_uri, redirect_handler, callback_handler


def _build_oauth_auth(name: str, spec: dict, stack: contextlib.AsyncExitStack):
    """Return an OAuthClientProvider for a remote spec carrying an `oauth` block, else None.
    The provider is an httpx.Auth the transports accept; it triggers the browser login lazily on
    the first 401 from the server."""
    oauth = spec.get("oauth")
    if not oauth:
        return None
    redirect_uri, redirect_handler, callback_handler = _loopback_oauth_handlers(stack)
    metadata = OAuthClientMetadata(
        client_name=f"roger ({name})",
        redirect_uris=[redirect_uri],
        grant_types=["authorization_code", "refresh_token"],
        response_types=["code"],
        token_endpoint_auth_method="client_secret_post",
        scope=oauth.get("scope"),   # usually None — the SDK derives scopes from server metadata
    )
    return OAuthClientProvider(
        server_url=_remote_url(spec),
        client_metadata=metadata,
        storage=_FileTokenStorage(name, oauth, redirect_uri),
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )


class MCPConnection:
    """Async context manager: connects to one MCP server (stdio subprocess or remote
    SSE / streamable-HTTP), discovers its tools, and exposes them in rollout()-compatible
    format with names namespaced `mcp__<server>__<tool>`."""

    def __init__(self, name: str, spec: dict, only: set[str] = None):
        """Args:
            name: server label, used as the `mcp__{name}__` tool-name prefix
            spec: one mcpServers entry — stdio (`command`/`args`/`env`) or remote
                  (`url` + optional `type` "sse"|"http" and `headers`)
            only: if set, only expose tools whose (un-prefixed) names are in this set
        """
        self._name = name
        self._spec = spec
        self._only = only
        self._stack = contextlib.AsyncExitStack()
        self._session: ClientSession = None
        self._discovered: list = []  # raw MCP Tool objects from list_tools(), after `only` filter
        # prefixed name -> original server name, so handlers can call_tool() with the real name
        self._orig: dict[str, str] = {}

    def _prefixed(self, tool_name: str) -> str:
        return f"mcp__{self._name}__{tool_name}"

    async def _open_transport(self):
        """Pick the transport from the spec and return its (read, write) streams.
        stdio/SSE yield a 2-tuple; streamable-HTTP yields (read, write, get_session_id)."""
        spec = self._spec
        if spec.get("command"):
            # errlog=DEVNULL avoids fileno() crash where sys.stderr isn't a real fd (e.g. Jupyter)
            params = StdioServerParameters(command=spec["command"],
                                           args=spec.get("args", []), env=spec.get("env"))
            read, write = await self._stack.enter_async_context(
                stdio_client(params, errlog=subprocess.DEVNULL))
            return read, write
        url, headers = _remote_url(spec), spec.get("headers")
        # `auth` is None unless the spec carries an `oauth` block; the transports accept it as
        # an httpx.Auth and the SDK runs the PKCE browser flow lazily on the first 401.
        auth = _build_oauth_auth(self._name, spec, self._stack)
        if spec.get("type") == "sse":
            read, write = await self._stack.enter_async_context(
                sse_client(url, headers=headers, auth=auth))
            return read, write
        # Default remote transport: streamable HTTP (the current MCP standard). Its context
        # manager yields a third value (a session-id getter) we don't need — drop it.
        read, write, _ = await self._stack.enter_async_context(
            streamablehttp_client(url, headers=headers, auth=auth))
        return read, write

    async def __aenter__(self):
        read, write = await self._open_transport()
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        tools = (await self._session.list_tools()).tools
        self._discovered = [t for t in tools if self._only is None or t.name in self._only]
        self._orig = {self._prefixed(t.name): t.name for t in self._discovered}
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    @property
    def tools(self) -> list[dict]:
        """Tool schemas in HF apply_chat_template format (list of JSON-schema dicts).
        MCP's inputSchema maps directly to HF's parameters field; names are namespaced."""
        return [
            {"type": "function", "function": {
                "name": self._prefixed(t.name),
                "description": t.description or "",
                "parameters": t.inputSchema,
            }}
            for t in self._discovered
        ]

    @property
    def handlers(self) -> dict:
        """prefixed-name → async callable mapping, drop-in for rollout(tool_handlers=...)."""
        return {p: self._make_handler(p) for p in self._orig}

    def _make_handler(self, prefixed: str):
        """Wrap a single MCP tool as an async callable that forwards kwargs to the server,
        calling it by its original (un-prefixed) name the server knows."""
        original = self._orig[prefixed]
        async def handler(**kwargs):
            result = await self._session.call_tool(original, kwargs)
            items = [_mcp_block(c) for c in result.content]
            # All-text (the common case) → one joined string; mixed/media → list for _parse_result.
            return "\n".join(items) if all(isinstance(x, str) for x in items) else items
        handler.__name__ = prefixed
        return handler


def load_mcp_config() -> dict:
    """Read ~/.roger/mcp.json and return its `mcpServers` mapping (name -> spec).
    Returns {} when the file is absent/empty; warns and returns {} on malformed JSON."""
    path = os.path.join(state_dir(), "mcp.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        warnings.warn(f"Ignoring {path}: could not parse ({e}).")
        return {}
    return data.get("mcpServers", {})


async def connect_servers(servers: dict) -> tuple[contextlib.AsyncExitStack, list[dict], dict]:
    """Open every MCP server in an `mcpServers` mapping and merge their tools/handlers.

    Args:
        servers: {name: spec, ...} as in mcp.json (see load_mcp_config / module docstring).
    Returns:
        (exit_stack, merged_tools, merged_handlers) — caller must await stack.aclose().

    A server that fails to start (bad command, unreachable URL, init error) is warned about
    and skipped so one broken entry can't abort the whole session; namespacing keeps the
    merge collision-free.
    """
    stack = contextlib.AsyncExitStack()
    all_tools, all_handlers = [], {}
    for name, spec in servers.items():
        try:
            conn = await stack.enter_async_context(MCPConnection(name, spec))
        except Exception as e:
            warnings.warn(f"MCP server '{name}' failed to connect; skipping. ({e})")
            continue
        all_tools.extend(conn.tools)
        all_handlers.update(conn.handlers)
    return stack, all_tools, all_handlers
