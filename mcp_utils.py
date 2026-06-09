"""Lightweight MCP client adapter — bridges external MCP tool servers to
rollout()'s existing tools/tool_handlers interface without modifying the rollout loop.

Usage with rollout():
    async with MCPConnection("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]) as mcp:
        await rollout(..., tools=local_fns + mcp.tools, tool_handlers=local_handlers | mcp.handlers)

For multiple servers, use connect_servers() which merges tools/handlers from all:
    stack, tools, handlers = await connect_servers([
        {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]},
        {"command": "python", "args": ["-m", "mcp_server_git", "--repo", "."]},
    ])
    # ... use tools/handlers in rollout ...
    await stack.aclose()
"""

import contextlib, subprocess, base64, io
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from PIL import Image


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
    """Async context manager: connects to one MCP server via stdio,
    discovers its tools, and exposes them in rollout()-compatible format."""

    def __init__(self, command: str, args: list[str] = [], env: dict = None,
                 only: set[str] = None):
        """Args:
            only: if set, only expose tools whose names are in this set (reduces prompt bloat)
        """
        self._params = StdioServerParameters(command=command, args=args, env=env)
        self._only = only
        self._stack = contextlib.AsyncExitStack()
        self._session: ClientSession = None
        self._discovered: list = []  # raw MCP Tool objects from list_tools()

    async def __aenter__(self):
        # stdio_client and ClientSession are nested async CMs; ExitStack manages both
        # errlog=DEVNULL avoids fileno() crash in Jupyter where sys.stderr isn't a real fd
        read, write = await self._stack.enter_async_context(
            stdio_client(self._params, errlog=subprocess.DEVNULL)
        )
        self._session = await self._stack.enter_async_context(ClientSession(read, write))
        await self._session.initialize()
        self._discovered = (await self._session.list_tools()).tools
        return self

    async def __aexit__(self, *exc):
        await self._stack.aclose()

    @property
    def tools(self) -> list[dict]:
        """Tool schemas in HF apply_chat_template format (list of JSON-schema dicts).
        MCP's inputSchema maps directly to HF's parameters field."""
        return [
            {"type": "function", "function": {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.inputSchema,
            }}
            for t in self._discovered
        ]

    @property
    def handlers(self) -> dict:
        """name → async callable mapping, drop-in for rollout(tool_handlers=...)."""
        return {t.name: self._make_handler(t.name) for t in self._discovered}

    def _make_handler(self, name: str):
        """Wrap a single MCP tool as an async callable that forwards kwargs to the server."""
        async def handler(**kwargs):
            result = await self._session.call_tool(name, kwargs)
            # Single image block → PIL.Image (e.g. Playwright screenshot); multi/text → joined str
            if len(result.content) == 1 and hasattr(result.content[0], "data"):
                c = result.content[0]
                return Image.open(io.BytesIO(base64.b64decode(c.data)))
            texts = [c.text for c in result.content if hasattr(c, "text")]
            return "\n".join(texts) if texts else str(result.content)
        handler.__name__ = name
        return handler


async def connect_servers(configs: list[dict]) -> tuple[contextlib.AsyncExitStack, list[dict], dict]:
    """Open multiple MCP servers and merge their tools/handlers.

    Args:
        configs: [{"command": str, "args": [...], "env": {...}}, ...]
    Returns:
        (exit_stack, merged_tools, merged_handlers) — caller must await stack.aclose()
    """
    stack = contextlib.AsyncExitStack()
    all_tools, all_handlers = [], {}
    for cfg in configs:
        conn = await stack.enter_async_context(
            MCPConnection(cfg["command"], cfg.get("args", []), cfg.get("env"))
        )
        all_tools.extend(conn.tools)
        all_handlers.update(conn.handlers)
    return stack, all_tools, all_handlers
