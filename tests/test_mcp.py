"""Tests for the MCP adapter (config parsing + tool-name namespacing).
Run with:  PYTHONPATH=src python -m pytest tests/test_mcp.py

No real MCP server is started: MCPConnection is exercised with a stub session so the
namespacing / handler-resolution logic can be checked in isolation.
"""

import asyncio, contextlib, json, threading, types
import roger.tools.mcp_utils as mcp_utils
from roger.tools.mcp_utils import (
    MCPConnection, load_mcp_config,
    _FileTokenStorage, _remote_url, _parse_callback, _build_oauth_auth,
)
from mcp.client.auth import OAuthClientProvider
from mcp.shared.auth import OAuthToken


# ---------------------------------------------------------------------------
# load_mcp_config — pure file parsing
# ---------------------------------------------------------------------------

def test_load_mcp_config_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_utils, "state_dir", lambda: str(tmp_path))
    assert load_mcp_config() == {}
    print("PASS test_load_mcp_config_absent")

def test_load_mcp_config_parses(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_utils, "state_dir", lambda: str(tmp_path))
    servers = {"filesystem": {"command": "npx", "args": ["-y", "x", "."]},
               "sentry": {"type": "http", "url": "https://example/mcp"}}
    (tmp_path / "mcp.json").write_text(json.dumps({"mcpServers": servers}), encoding="utf-8")
    assert load_mcp_config() == servers
    print("PASS test_load_mcp_config_parses")

def test_load_mcp_config_malformed(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_utils, "state_dir", lambda: str(tmp_path))
    (tmp_path / "mcp.json").write_text("{ not json", encoding="utf-8")
    assert load_mcp_config() == {}  # warns + returns {}, does not raise
    print("PASS test_load_mcp_config_malformed")


# ---------------------------------------------------------------------------
# Namespacing + handler resolution — stub session, no transport
# ---------------------------------------------------------------------------

class _StubTool:
    def __init__(self, name):
        self.name = name
        self.description = f"desc {name}"
        self.inputSchema = {"type": "object", "properties": {}}

class _StubResult:
    def __init__(self, text):
        self.content = [types.SimpleNamespace(type="text", text=text)]

class _StubSession:
    def __init__(self, names):
        self._names = names
        self.calls = []  # records (name, kwargs) the handler forwards
    async def call_tool(self, name, kwargs):
        self.calls.append((name, kwargs))
        return _StubResult(f"ran {name}")

def _make_conn(server_name, tool_names):
    """Build an MCPConnection wired to a stub session, bypassing __aenter__/transport."""
    conn = MCPConnection(server_name, {"command": "x"})
    conn._session = _StubSession(tool_names)
    conn._discovered = [_StubTool(n) for n in tool_names]
    conn._orig = {conn._prefixed(n): n for n in tool_names}
    return conn

def test_tools_are_namespaced():
    conn = _make_conn("filesystem", ["read_file", "list_directory"])
    names = [t["function"]["name"] for t in conn.tools]
    assert names == ["mcp__filesystem__read_file", "mcp__filesystem__list_directory"]
    # inputSchema maps straight to HF's parameters field
    assert conn.tools[0]["function"]["parameters"] == {"type": "object", "properties": {}}
    print("PASS test_tools_are_namespaced")

def test_handler_resolves_original_name():
    import asyncio
    conn = _make_conn("git", ["status"])
    handler = conn.handlers["mcp__git__status"]
    out = asyncio.run(handler(repo="."))
    # The server is called with the un-prefixed name, not the namespaced one
    assert conn._session.calls == [("status", {"repo": "."})]
    assert out == "ran status"
    print("PASS test_handler_resolves_original_name")

def test_namespacing_prevents_collision():
    # Two servers exposing a same-named tool must not clash once merged
    a = _make_conn("alpha", ["read"])
    b = _make_conn("beta", ["read"])
    merged = {**a.handlers, **b.handlers}
    assert set(merged) == {"mcp__alpha__read", "mcp__beta__read"}
    print("PASS test_namespacing_prevents_collision")


# ---------------------------------------------------------------------------
# OAuth 2.0 — offline (no real server, no browser, no network)
# ---------------------------------------------------------------------------

_OAUTH = {"clientId": "cid-123", "clientSecret": "secret-xyz"}

def test_remote_url_accepts_serverurl_alias():
    assert _remote_url({"url": "https://a/mcp"}) == "https://a/mcp"
    # Google's Gmail spec uses `serverUrl`, not `url`
    assert _remote_url({"serverUrl": "https://gmailmcp.googleapis.com/mcp/v1"}) \
        == "https://gmailmcp.googleapis.com/mcp/v1"
    print("PASS test_remote_url_accepts_serverurl_alias")

def test_parse_callback():
    assert _parse_callback("/callback?code=abc&state=xyz") == ("abc", "xyz")
    # state is optional; a missing code surfaces as None so callback_handler can reject it
    assert _parse_callback("/callback?code=abc") == ("abc", None)
    assert _parse_callback("/callback?error=access_denied") == (None, None)
    # full URL, as pasted from a browser address bar in the headless fallback
    assert _parse_callback("http://127.0.0.1:5000/callback?code=abc&state=xyz") == ("abc", "xyz")
    print("PASS test_parse_callback")

def test_needs_interactive_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_utils, "state_dir", lambda: str(tmp_path))
    plain = {"fs": {"command": "npx"}, "http": {"url": "https://a/mcp"}}
    oauthed = dict(plain, canva={"url": "https://mcp.canva.com/mcp", "oauth": {}})
    assert not mcp_utils.needs_interactive_auth(plain)
    # oauth block (even the empty DCR form), no cached tokens yet -> interactive login expected
    assert mcp_utils.needs_interactive_auth(oauthed)
    # once tokens are cached the login is silent, so no interactive connect expected
    asyncio.run(_FileTokenStorage("canva", _OAUTH, "http://127.0.0.1:1/callback")
                .set_tokens(OAuthToken(access_token="tok")))
    assert not mcp_utils.needs_interactive_auth(oauthed)
    print("PASS test_needs_interactive_auth")

def test_callback_handler_paste_fallback(monkeypatch, capsys):
    "Headless path: no browser opened, the user pastes the failed-redirect URL from their own box."
    monkeypatch.setattr(mcp_utils.webbrowser, "open", lambda url: False)
    pasted = iter(["not-a-url",   # junk first: handler must re-prompt, not crash
                   "http://127.0.0.1:5000/callback?code=abc&state=xyz"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(pasted))

    async def go():
        stack = contextlib.AsyncExitStack()
        try:
            _, redirect_handler, callback_handler = mcp_utils._loopback_oauth_handlers(stack)
            await redirect_handler("https://auth.example/authorize")
            assert await callback_handler() == ("abc", "xyz")
        finally:
            await stack.aclose()
    asyncio.run(go())
    out = capsys.readouterr().out
    assert "https://auth.example/authorize" in out          # URL printed for the local browser
    assert "Couldn't find an authorization code" in out     # junk line was rejected with guidance
    print("PASS test_callback_handler_paste_fallback")

def test_callback_handler_browser_path_untouched(monkeypatch):
    "When a browser opens locally, the handler must await the loopback redirect, never stdin."
    monkeypatch.setattr(mcp_utils.webbrowser, "open", lambda url: True)
    monkeypatch.setattr("builtins.input",
                        lambda prompt="": (_ for _ in ()).throw(AssertionError("stdin read")))

    async def go():
        stack = contextlib.AsyncExitStack()
        try:
            redirect_uri, redirect_handler, callback_handler = \
                mcp_utils._loopback_oauth_handlers(stack)
            await redirect_handler("https://auth.example/authorize")
            # Simulate the provider's redirect landing on the real loopback listener
            import urllib.request
            def hit():
                urllib.request.urlopen(f"{redirect_uri}?code=abc&state=xyz", timeout=5)
            threading.Thread(target=hit, daemon=True).start()
            assert await asyncio.wait_for(callback_handler(), 10) == ("abc", "xyz")
        finally:
            await stack.aclose()
    asyncio.run(go())
    print("PASS test_callback_handler_browser_path_untouched")

def test_token_storage_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_utils, "state_dir", lambda: str(tmp_path))
    store = _FileTokenStorage("gmail", _OAUTH, "http://127.0.0.1:5000/callback")

    async def go():
        # Fresh store: no tokens yet, but client_info is seeded from the spec creds so the SDK
        # skips Dynamic Client Registration and uses the pre-registered client.
        assert await store.get_tokens() is None
        seeded = await store.get_client_info()
        assert seeded.client_id == "cid-123" and seeded.client_secret == "secret-xyz"
        # set_tokens (called by the SDK after login and every refresh) persists to disk
        await store.set_tokens(OAuthToken(access_token="tok", refresh_token="ref"))
        reloaded = await store.get_tokens()
        assert reloaded.access_token == "tok" and reloaded.refresh_token == "ref"
        # Empty oauth block (no pre-registered creds): get_client_info must return None so the
        # SDK falls through to Dynamic Client Registration instead of KeyError'ing on clientId.
        dcr = _FileTokenStorage("canva", {}, "http://127.0.0.1:5000/callback")
        assert await dcr.get_client_info() is None

    asyncio.run(go())
    # Cache is keyed by server name under ~/.roger/oauth/<name>.json
    assert (tmp_path / "oauth" / "gmail.json").exists()
    print("PASS test_token_storage_roundtrip")

def test_token_storage_keyed_per_server(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_utils, "state_dir", lambda: str(tmp_path))
    async def go():
        await _FileTokenStorage("gmail", _OAUTH, "http://127.0.0.1:1/callback").set_tokens(
            OAuthToken(access_token="a"))
        await _FileTokenStorage("drive", _OAUTH, "http://127.0.0.1:1/callback").set_tokens(
            OAuthToken(access_token="b"))
    asyncio.run(go())
    # Two servers → two distinct cache files (no cross-contamination)
    assert (tmp_path / "oauth" / "gmail.json").exists()
    assert (tmp_path / "oauth" / "drive.json").exists()
    print("PASS test_token_storage_keyed_per_server")

def test_build_oauth_auth(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp_utils, "state_dir", lambda: str(tmp_path))
    async def go():
        stack = contextlib.AsyncExitStack()
        try:
            # No `oauth` block → no auth provider (existing header-only path is unaffected)
            assert _build_oauth_auth("plain", {"url": "https://a/mcp"}, stack) is None
            # Empty `oauth` block → provider with no seeded creds, i.e. the DCR path
            assert isinstance(_build_oauth_auth("canva", {"url": "https://a/mcp", "oauth": {}},
                                                stack), OAuthClientProvider)
            # With `oauth` → an OAuthClientProvider pointed at the resolved server URL
            prov = _build_oauth_auth(
                "gmail",
                {"serverUrl": "https://gmailmcp.googleapis.com/mcp/v1", "oauth": _OAUTH},
                stack)
            assert isinstance(prov, OAuthClientProvider)
            assert prov.context.server_url == "https://gmailmcp.googleapis.com/mcp/v1"
            # redirect URI is a bound loopback the OAuth client must allow
            assert str(prov.context.client_metadata.redirect_uris[0]).startswith("http://127.0.0.1:")
        finally:
            await stack.aclose()   # tears down the loopback listener
    asyncio.run(go())
    print("PASS test_build_oauth_auth")
