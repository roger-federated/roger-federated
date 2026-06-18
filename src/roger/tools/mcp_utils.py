"""Lightweight MCP client adapter — bridges external MCP tool servers to
rollout()'s existing tools/tool_handlers interface without modifying the rollout loop.

Servers are declared in ~/.roger/mcp.json using the standard `mcpServers` schema
(copy-paste compatible with Claude Desktop / Cursor / Claude Code), e.g.:
    {"mcpServers": {
        "filesystem": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]},
        "sentry":     {"type": "http", "url": "https://mcp.sentry.dev/mcp",
                       "headers": {"Authorization": "Bearer ..."}}
    }}
Each entry is either a local stdio subprocess (`command`/`args`/`env`) or a remote server
(`url` + optional `type` "sse"|"http" and `headers`). Tools are namespaced `mcp__<server>__<tool>`
so same-named tools from different servers can't collide.

Usage with rollout():
    stack, tools, handlers = await connect_servers(load_mcp_config())
    try:
        await rollout(..., tools=tools, tool_handlers=handlers)
    finally:
        await stack.aclose()
"""

import contextlib, subprocess, base64, io, json, os, warnings
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
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
        url, headers = spec["url"], spec.get("headers")
        if spec.get("type") == "sse":
            read, write = await self._stack.enter_async_context(sse_client(url, headers=headers))
            return read, write
        # Default remote transport: streamable HTTP (the current MCP standard). Its context
        # manager yields a third value (a session-id getter) we don't need — drop it.
        read, write, _ = await self._stack.enter_async_context(
            streamablehttp_client(url, headers=headers))
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
