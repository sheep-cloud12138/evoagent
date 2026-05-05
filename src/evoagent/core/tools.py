from __future__ import annotations

import copy
import ipaddress
import json
import socket
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse
from urllib.error import URLError
from urllib.request import Request, urlopen

from evoagent.core.config import settings
from evoagent.core.weather import fetch_weather_report

ToolCallable = Callable[[dict[str, Any]], str]
SemanticRetrieverCallable = Callable[[str, int, bool], list[dict[str, Any]]]


_SEMANTIC_RETRIEVER: SemanticRetrieverCallable | None = None


def register_semantic_retriever(callback: SemanticRetrieverCallable | None) -> None:
    global _SEMANTIC_RETRIEVER
    _SEMANTIC_RETRIEVER = callback


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    source: str
    category: str
    schema: dict[str, Any]
    handler: ToolCallable


def _get_current_time(_args: dict[str, Any]) -> str:
    return datetime.now(UTC).isoformat()


def _get_weather(args: dict[str, Any]) -> str:
    location = str(args.get("location", "")).strip()
    if not location:
        return "error: missing location"
    return fetch_weather_report(location)


def _is_restricted_ip(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    )


def _validate_fetch_url(url: str) -> str | None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return "url scheme must be http or https"
    if not parsed.hostname:
        return "url host is missing"

    if settings.allow_private_network_fetch:
        return None

    host = parsed.hostname.strip().strip("[]")
    try:
        if _is_restricted_ip(ipaddress.ip_address(host)):
            return "private or local network targets are blocked"
        return None
    except ValueError:
        pass

    lowered = host.lower().rstrip(".")
    if lowered == "localhost" or lowered.endswith(".localhost"):
        return "private or local network targets are blocked"

    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return f"host resolution failed ({exc})"

    for info in infos:
        sockaddr = info[4]
        if not sockaddr:
            continue
        try:
            resolved_ip = ipaddress.ip_address(str(sockaddr[0]))
        except ValueError:
            continue
        if _is_restricted_ip(resolved_ip):
            return "private or local network targets are blocked"
    return None


def _fetch_url_text(args: dict[str, Any]) -> str:
    url = str(args.get("url", "")).strip()
    if not url:
        return "error: missing url"

    validation_error = _validate_fetch_url(url)
    if validation_error:
        return f"error: {validation_error}"

    max_chars_raw = args.get("max_chars", 2000)
    try:
        max_chars = max(200, min(int(max_chars_raw), 10000))
    except Exception:
        max_chars = 2000

    req = Request(
        url=url,
        headers={
            "User-Agent": "evoagent-tool-fetch-url",
            "Accept": "text/plain,text/html,application/json;q=0.9,*/*;q=0.1",
        },
        method="GET",
    )
    try:
        with urlopen(
            req, timeout=max(5, settings.subagent_api_timeout_seconds)
        ) as resp:  # nosec B310
            body = resp.read().decode("utf-8", errors="replace")
    except URLError as exc:
        return f"error: fetch failed ({exc})"

    text = body.strip().replace("\r\n", "\n")
    return text[:max_chars]


def _mcp_call_tool(args: dict[str, Any]) -> str:
    if not settings.mcp_server_url:
        return "error: MCP server is not configured (set MCP_SERVER_URL)"

    tool_name = str(args.get("tool_name", "")).strip()
    if not tool_name:
        return "error: missing tool_name"

    payload = {
        "jsonrpc": "2.0",
        "id": "evoagent-mcp-call",
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": args.get("arguments", {}),
        },
    }
    headers = {
        "Content-Type": "application/json",
        "User-Agent": "evoagent-mcp-bridge",
    }
    if settings.mcp_auth_token:
        headers["Authorization"] = f"Bearer {settings.mcp_auth_token}"

    data = json.dumps(payload).encode("utf-8")
    req = Request(
        url=settings.mcp_server_url, headers=headers, data=data, method="POST"
    )
    try:
        with urlopen(req, timeout=max(5, settings.mcp_timeout_seconds)) as resp:  # nosec B310
            body = resp.read().decode("utf-8", errors="replace")
    except URLError as exc:
        return f"error: mcp call failed ({exc})"

    return body[:12000]


def _semantic_memory_search(args: dict[str, Any]) -> str:
    if _SEMANTIC_RETRIEVER is None:
        return "error: semantic retriever is not configured"

    query = str(args.get("query", "")).strip()
    if not query:
        return "error: missing query"

    top_k_raw = args.get("top_k", 5)
    try:
        top_k = max(1, min(int(top_k_raw), 20))
    except Exception:
        top_k = 5

    use_negative = bool(args.get("use_negative", False))
    try:
        rows = _SEMANTIC_RETRIEVER(query, top_k, use_negative)
    except Exception as exc:
        return f"error: semantic retrieval failed ({exc})"

    return json.dumps(
        {
            "query": query,
            "top_k": top_k,
            "use_negative": use_negative,
            "hits": rows,
        },
        ensure_ascii=False,
    )


def _resolve_local_path(raw_path: str) -> tuple[Path | None, str | None]:
    text = raw_path.strip()
    if not text:
        return None, "missing path"
    cwd = Path.cwd().resolve()
    target = Path(text)
    resolved = (
        (cwd / target).resolve() if not target.is_absolute() else target.resolve()
    )
    try:
        resolved.relative_to(cwd)
    except Exception:
        return None, "path must stay inside current working directory"
    return resolved, None


def _write_text_file(args: dict[str, Any]) -> str:
    raw_path = str(args.get("path", ""))
    content = str(args.get("content", ""))
    overwrite = bool(args.get("overwrite", False))
    create_parents = bool(args.get("create_parents", True))

    target, err = _resolve_local_path(raw_path)
    if err or target is None:
        return f"error: {err or 'invalid path'}"

    if target.exists() and not overwrite:
        return "error: target exists (set overwrite=true to replace)"

    if create_parents:
        target.parent.mkdir(parents=True, exist_ok=True)

    try:
        target.write_text(content, encoding="utf-8")
    except Exception as exc:
        return f"error: write failed ({exc})"

    return f"ok: wrote {target} bytes={len(content.encode('utf-8'))}"


def _read_text_file(args: dict[str, Any]) -> str:
    raw_path = str(args.get("path", ""))
    max_chars_raw = args.get("max_chars", 4000)
    try:
        max_chars = max(200, min(int(max_chars_raw), 12000))
    except Exception:
        max_chars = 4000

    target, err = _resolve_local_path(raw_path)
    if err or target is None:
        return f"error: {err or 'invalid path'}"
    if not target.exists() or not target.is_file():
        return "error: target file not found"

    try:
        text = target.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"error: read failed ({exc})"
    return text[:max_chars]


def _list_directory(args: dict[str, Any]) -> str:
    raw_path = str(args.get("path", "."))
    target, err = _resolve_local_path(raw_path)
    if err or target is None:
        return f"error: {err or 'invalid path'}"
    if not target.exists() or not target.is_dir():
        return "error: directory not found"

    limit_raw = args.get("limit", 200)
    try:
        limit = max(1, min(int(limit_raw), 500))
    except Exception:
        limit = 200

    try:
        names = sorted([p.name + ("/" if p.is_dir() else "") for p in target.iterdir()])
    except Exception as exc:
        return f"error: list failed ({exc})"
    return "\n".join(names[:limit])


_TOOL_REGISTRY: dict[str, ToolDefinition] = {}
_TOOL_STATS: dict[str, dict[str, Any]] = {}


def _ensure_tool_stats(name: str, source: str, category: str) -> None:
    payload = _TOOL_STATS.get(name)
    if payload is None:
        _TOOL_STATS[name] = {
            "calls": 0,
            "success": 0,
            "failed": 0,
            "last_error": "",
            "source": source,
            "category": category,
        }
        return
    payload["source"] = source
    payload["category"] = category


def _register_tool(defn: ToolDefinition) -> None:
    _TOOL_REGISTRY[defn.name] = defn
    _ensure_tool_stats(defn.name, defn.source, defn.category)


def _register_builtin_tools() -> None:
    _register_tool(
        ToolDefinition(
            name="get_current_time",
            source="builtin",
            category="utility",
            schema={
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get current UTC datetime in ISO8601 format.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
            },
            handler=_get_current_time,
        )
    )
    _register_tool(
        ToolDefinition(
            name="fetch_url_text",
            source="builtin",
            category="web",
            schema={
                "type": "function",
                "function": {
                    "name": "fetch_url_text",
                    "description": "Fetch text content from a URL.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string", "description": "Absolute URL"},
                            "max_chars": {
                                "type": "integer",
                                "description": "Max response characters",
                            },
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                },
            },
            handler=_fetch_url_text,
        )
    )
    _register_tool(
        ToolDefinition(
            name="get_weather",
            source="builtin",
            category="web",
            schema={
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather and today's forecast for a location.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "City or place name, e.g. 珠海 or Zhuhai",
                            },
                        },
                        "required": ["location"],
                        "additionalProperties": False,
                    },
                },
            },
            handler=_get_weather,
        )
    )
    _register_tool(
        ToolDefinition(
            name="mcp_call_tool",
            source="mcp",
            category="integration",
            schema={
                "type": "function",
                "function": {
                    "name": "mcp_call_tool",
                    "description": "Call a remote MCP tool via MCP JSON-RPC gateway.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "tool_name": {
                                "type": "string",
                                "description": "MCP tool name",
                            },
                            "arguments": {
                                "type": "object",
                                "description": "Arguments object passed to MCP tool",
                            },
                        },
                        "required": ["tool_name"],
                        "additionalProperties": False,
                    },
                },
            },
            handler=_mcp_call_tool,
        )
    )
    _register_tool(
        ToolDefinition(
            name="semantic_memory_search",
            source="builtin",
            category="retrieval",
            schema={
                "type": "function",
                "function": {
                    "name": "semantic_memory_search",
                    "description": "Search semantic memory using BM25/embedding hybrid retrieval.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query text",
                            },
                            "top_k": {
                                "type": "integer",
                                "description": "Number of hits to return",
                            },
                            "use_negative": {
                                "type": "boolean",
                                "description": "Search negative semantic memory instead of positive memory",
                            },
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            handler=_semantic_memory_search,
        )
    )
    _register_tool(
        ToolDefinition(
            name="write_text_file",
            source="builtin",
            category="filesystem",
            schema={
                "type": "function",
                "function": {
                    "name": "write_text_file",
                    "description": "Write text content to a file in current working directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative file path",
                            },
                            "content": {
                                "type": "string",
                                "description": "Text to write",
                            },
                            "overwrite": {
                                "type": "boolean",
                                "description": "Overwrite existing file",
                            },
                            "create_parents": {
                                "type": "boolean",
                                "description": "Create missing parent directories",
                            },
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            handler=_write_text_file,
        )
    )
    _register_tool(
        ToolDefinition(
            name="read_text_file",
            source="builtin",
            category="filesystem",
            schema={
                "type": "function",
                "function": {
                    "name": "read_text_file",
                    "description": "Read text content from a file in current working directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative file path",
                            },
                            "max_chars": {
                                "type": "integer",
                                "description": "Maximum characters to return",
                            },
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            handler=_read_text_file,
        )
    )
    _register_tool(
        ToolDefinition(
            name="list_directory",
            source="builtin",
            category="filesystem",
            schema={
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List directory entries in current working directory.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Relative directory path",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum entries to return",
                            },
                        },
                        "additionalProperties": False,
                    },
                },
            },
            handler=_list_directory,
        )
    )


_register_builtin_tools()


def register_skill_tool(
    name: str,
    description: str,
    parameters: dict[str, Any],
    handler: ToolCallable,
    category: str = "skill",
) -> None:
    """Register a skill-exposed tool into the unified tool registry."""
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }
    _register_tool(
        ToolDefinition(
            name=name,
            source="skill",
            category=category,
            schema=schema,
            handler=handler,
        )
    )


def unregister_tools(
    source: str | None = None,
    category: str | None = None,
    name_prefix: str | None = None,
) -> list[str]:
    removed: list[str] = []
    for name, definition in list(_TOOL_REGISTRY.items()):
        if source is not None and definition.source != source:
            continue
        if category is not None and definition.category != category:
            continue
        if name_prefix is not None and not name.startswith(name_prefix):
            continue
        _TOOL_REGISTRY.pop(name, None)
        _TOOL_STATS.pop(name, None)
        removed.append(name)
    return removed


def resolve_tool_names(
    preferred_names: tuple[str, ...] = (),
    categories: tuple[str, ...] = (),
    min_success_rate: float = 0.0,
) -> tuple[str, ...]:
    if preferred_names:
        candidates = [name for name in preferred_names if name in _TOOL_REGISTRY]
    else:
        candidates = list(_TOOL_REGISTRY.keys())

    category_filter = set(categories)
    selected: list[str] = []
    for name in candidates:
        definition = _TOOL_REGISTRY.get(name)
        if definition is None:
            continue
        if category_filter and definition.category not in category_filter:
            continue

        stat = _TOOL_STATS.get(name, {})
        calls = int(stat.get("calls", 0))
        success = int(stat.get("success", 0))
        success_rate = 1.0 if calls == 0 else success / max(calls, 1)
        if success_rate + 1e-12 < min_success_rate:
            continue

        selected.append(name)

    return tuple(selected)


def get_tool_specs(names: tuple[str, ...]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for name in names:
        definition = _TOOL_REGISTRY.get(name)
        if definition is None:
            continue
        spec = copy.deepcopy(definition.schema)
        spec["source"] = definition.source
        spec["category"] = definition.category
        specs.append(spec)
    return specs


def get_tool_specs_for_llm(names: tuple[str, ...]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for spec in get_tool_specs(names):
        specs.append({"type": spec["type"], "function": spec["function"]})
    return specs


def execute_tool_with_meta(name: str, args: dict[str, Any]) -> dict[str, Any]:
    definition = _TOOL_REGISTRY.get(name)
    if definition is None:
        return {
            "name": name,
            "source": "unknown",
            "category": "unknown",
            "success": False,
            "result": f"error: unknown tool {name}",
            "error": f"unknown_tool:{name}",
            "success_rate": 0.0,
        }

    error = ""
    try:
        result = definition.handler(args)
        success = not str(result).strip().lower().startswith("error:")
        if not success:
            error = str(result)
    except Exception as exc:  # pragma: no cover
        result = f"error: tool execution failed ({exc})"
        success = False
        error = str(exc)

    stats = _TOOL_STATS[definition.name]
    stats["calls"] += 1
    if success:
        stats["success"] += 1
    else:
        stats["failed"] += 1
        stats["last_error"] = error[:240]

    calls = max(int(stats["calls"]), 1)
    success_rate = int(stats["success"]) / calls
    return {
        "name": definition.name,
        "source": definition.source,
        "category": definition.category,
        "success": success,
        "result": str(result),
        "error": error,
        "success_rate": success_rate,
    }


def execute_tool(name: str, args: dict[str, Any]) -> str:
    return execute_tool_with_meta(name, args)["result"]


def get_tool_stats() -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name, stats in _TOOL_STATS.items():
        calls = int(stats.get("calls", 0))
        success = int(stats.get("success", 0))
        success_rate = 1.0 if calls == 0 else success / max(calls, 1)
        report[name] = {
            "calls": calls,
            "success": success,
            "failed": int(stats.get("failed", 0)),
            "last_error": str(stats.get("last_error", "")),
            "source": str(stats.get("source", "unknown")),
            "category": str(stats.get("category", "unknown")),
            "success_rate": success_rate,
        }
    return report


def reset_tool_stats() -> None:
    for name, definition in _TOOL_REGISTRY.items():
        _TOOL_STATS[name] = {
            "calls": 0,
            "success": 0,
            "failed": 0,
            "last_error": "",
            "source": definition.source,
            "category": definition.category,
        }
