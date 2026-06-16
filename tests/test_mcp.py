"""Tests for the MCP adapter (config parsing + tool-name namespacing).
Run with:  PYTHONPATH=src python -m pytest tests/test_mcp.py

No real MCP server is started: MCPConnection is exercised with a stub session so the
namespacing / handler-resolution logic can be checked in isolation.
"""

import json, os, types
import roger.tools.mcp_utils as mcp_utils
from roger.tools.mcp_utils import MCPConnection, load_mcp_config


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
