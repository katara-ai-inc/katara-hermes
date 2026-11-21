"""Shared test support: load the plugin package and fake Katara's MCP endpoint.

Run from the repository root with Hermes Agent importable, e.g.:

    PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/venv/bin/python -m unittest discover -s tests

Tests use only the standard library (unittest) and httpx (a Hermes core dependency), so they run inside
the Hermes venv without installing anything; pytest also collects them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

import httpx

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "hermes_katara"


def load_plugin():
    if PACKAGE not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            PACKAGE, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        spec.loader.exec_module(module)
    return sys.modules[PACKAGE]


ToolHandler = Callable[[dict[str, Any]], Any]


class FakeKatara:
    """In-memory stand-in for Katara's stateless Streamable HTTP MCP endpoint.

    ``tools`` maps tool name -> handler(arguments) returning a payload dict, or a ``(payload, True)``
    tuple for a tool error. ``sse=True`` answers as text/event-stream like the Go SDK can.
    """

    def __init__(self, tools: dict[str, ToolHandler] | None = None, *, sse: bool = False, token: str = "kt_test"):
        self.tools = tools or {}
        self.sse = sse
        self.token = token
        self.requests: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []
        self.fail_next: list[Exception | int] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def calls(self, name: str | None = None) -> list[dict[str, Any]]:
        out = [r["params"] for r in self.requests if r.get("method") == "tools/call"]
        return [c for c in out if name is None or c["name"] == name]

    def _reply(self, request_id: Any, result: dict[str, Any]) -> httpx.Response:
        message = {"jsonrpc": "2.0", "id": request_id, "result": result}
        if self.sse:
            body = f"event: message\ndata: {json.dumps(message)}\n\n"
            return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})
        return httpx.Response(200, json=message)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if self.fail_next:
            failure = self.fail_next.pop(0)
            if isinstance(failure, Exception):
                raise failure
            return httpx.Response(failure, text="denied by fake")
        if request.headers.get("authorization") != f"Bearer {self.token}":
            return httpx.Response(401, text="no bearer token")
        body = json.loads(request.content or b"{}")
        self.requests.append(body)
        self.headers.append(request.headers)
        method = body.get("method")
        if "id" not in body:  # notification
            return httpx.Response(202)
        if method == "initialize":
            return self._reply(body["id"], {"protocolVersion": "2025-06-18", "capabilities": {"tools": {}},
                                            "serverInfo": {"name": "katara", "version": "test"}})
        if method == "tools/list":
            return self._reply(body["id"], {"tools": [{"name": n, "inputSchema": {"type": "object"}} for n in self.tools]})
        if method == "tools/call":
            name = body["params"]["name"]
            handler = self.tools.get(name)
            if handler is None:
                return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"],
                                                 "error": {"code": -32602, "message": f"unknown tool {name}"}})
            outcome = handler(body["params"].get("arguments") or {})
            payload, is_error = outcome if isinstance(outcome, tuple) else (outcome, False)
            text = json.dumps(payload)
            return self._reply(body["id"], {"content": [{"type": "text", "text": text}],
                                            "structuredContent": payload, "isError": is_error})
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": body["id"],
                                         "error": {"code": -32601, "message": "method not found"}})


RECALL_PAYLOAD = {
    "protocol_version": 1,
    "facts": [
        {"id": 1, "fact_id": "f-1", "fact": "Prefers concise status updates", "kind": "preference",
         "entity_slug": None, "provenance": "conversation 2026-11-01"},
        {"id": 2, "fact_id": "f-2", "fact": "Owns the Katara backend repo", "kind": "fact",
         "entity_slug": "people/vcastellm", "provenance": "user said in chat"},
    ],
    "total": 2,
    "results": [
        {"slug": "projects/katara-mcp", "title": "Katara MCP", "chunk": "Stateless Streamable HTTP endpoint.",
         "evidence": "high_vector_match", "create_safety": "exists", "provenance": "projects/katara-mcp"},
    ],
    "katara": {"backend": "brain", "retrieval_status": {"state": "complete"}},
    "budget_tokens": 1200, "budget_used": 80, "dropped_count": 0,
}
