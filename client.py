"""Minimal client for Katara's MCP memory endpoint (Streamable HTTP, JSON-RPC 2.0).

Katara serves its memory verbs (``recall``, ``remember``, ``forget``, ``entity``) as MCP tools on a
stateless Streamable HTTP endpoint (default ``https://mcp.katara.ai/mcp``). An organization-scoped
Katara API token is sent as a bearer credential; the token carries its organization, so no org
header is needed unless the caller pins one.

Only ``httpx`` (a Hermes core dependency) is used: no MCP SDK, no event loop, no background
threads. Every call is synchronous and bounded by a timeout.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

DEFAULT_MCP_URL = "https://mcp.katara.ai/mcp"
MCP_PROTOCOL_VERSION = "2025-06-18"
CLIENT_NAME = "katara-hermes"
CLIENT_VERSION = "0.1.0"

# Server-supplied text is echoed to the model/user; keep it short.
_MAX_ERROR_TEXT = 300


class KataraError(RuntimeError):
    """A failed Katara call. ``code`` is a stable machine label; ``payload`` the server envelope."""

    def __init__(self, message: str, *, code: str = "error", status: int = 0,
                 payload: Optional[dict[str, Any]] = None):
        super().__init__(message)
        self.code = code
        self.status = status
        self.payload = payload or {}


# Failures a retry cannot fix: the user has to change the token, its grants or the source setting.
_CONFIG_ERROR_CODES = frozenset({"unauthorized", "forbidden", "endpoint_not_found", "redirect", "scope_denied"})
_CONFIG_ERROR_DETAILS = frozenset({"DEFAULT_SOURCE_REQUIRED"})


def is_config_error(exc: "KataraError") -> bool:
    return exc.code in _CONFIG_ERROR_CODES or exc.payload.get("detail") in _CONFIG_ERROR_DETAILS


def _short(text: str) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= _MAX_ERROR_TEXT else text[: _MAX_ERROR_TEXT - 1] + "…"


def _http_error(status: int, body: str) -> KataraError:
    hint = {
        401: ("unauthorized", "Katara rejected the API key (401). Check KATARA_API_KEY and that the token "
                              "has not expired or been revoked."),
        403: ("forbidden", "Katara denied access (403). The API token needs the Agent read and Memory read "
                           "grants (and Memory write to remember/forget) for this organization."),
        404: ("endpoint_not_found", "Katara MCP endpoint not found (404). Check the configured MCP URL."),
        413: ("too_large", "Request too large for Katara (413)."),
        428: ("precondition", "Katara organization is not ready yet (428)."),
        429: ("rate_limited", "Katara rate limit reached (429). Retry later."),
    }.get(status)
    if hint:
        code, message = hint
    else:
        code, message = ("unavailable" if status >= 500 else "http_error"), f"Katara returned HTTP {status}."
    detail = _short(body)
    return KataraError(f"{message} {detail}".strip() if detail else message, code=code, status=status)


def _parse_sse(text: str, request_id: int) -> Optional[dict[str, Any]]:
    """The JSON-RPC response for ``request_id`` from an SSE body (``data:`` lines per event)."""
    data_lines: list[str] = []
    found: Optional[dict[str, Any]] = None

    def flush() -> None:
        nonlocal found
        if not data_lines:
            return
        raw = "\n".join(data_lines)
        data_lines.clear()
        try:
            message = json.loads(raw)
        except ValueError:
            return
        if isinstance(message, dict) and message.get("id") == request_id and found is None:
            found = message

    for line in text.splitlines():
        if not line.strip():
            flush()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip(" "))
    flush()
    return found


class KataraMCPClient:
    """Synchronous JSON-RPC over Katara's Streamable HTTP MCP endpoint.

    One instance per provider instance (never shared across profiles). Thread-safe: the
    request-id counter and lazy initialization are guarded by a lock; httpx.Client is
    thread-safe for concurrent requests.
    """

    def __init__(self, api_key: str, url: str = DEFAULT_MCP_URL, *, org_id: str = "",
                 timeout: float = 20.0, transport: Optional[httpx.BaseTransport] = None):
        if not api_key:
            raise ValueError("Katara API key is required")
        self._url = (url or DEFAULT_MCP_URL).rstrip("/")
        if not self._url.startswith("https://") and not self._url.startswith("http://localhost") \
                and not self._url.startswith("http://127.0.0.1"):
            raise ValueError("Katara MCP URL must use https:// (http:// is allowed only for localhost)")
        headers = {
            "Authorization": f"Bearer {api_key.strip()}",
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "User-Agent": f"{CLIENT_NAME}/{CLIENT_VERSION}",
        }
        if org_id:
            headers["X-Katara-Org-Id"] = org_id.strip()
        # follow_redirects stays off: a redirect must never carry the bearer token elsewhere.
        self._http = httpx.Client(headers=headers, timeout=timeout, transport=transport,
                                  follow_redirects=False)
        self._timeout = timeout
        self._lock = threading.Lock()
        self._next_id = 0
        self._initialized = False
        self._session_id = ""
        self._protocol_version = MCP_PROTOCOL_VERSION

    # -- JSON-RPC plumbing ---------------------------------------------------------------------

    def _new_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def _headers(self) -> dict[str, str]:
        headers = {"MCP-Protocol-Version": self._protocol_version}
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    def _post(self, payload: dict[str, Any], timeout: Optional[float]) -> httpx.Response:
        try:
            return self._http.post(self._url, json=payload, headers=self._headers(),
                                   timeout=timeout if timeout is not None else self._timeout)
        except httpx.TimeoutException as exc:
            raise KataraError("Katara request timed out.", code="timeout") from exc
        except httpx.HTTPError as exc:
            raise KataraError(f"Could not reach Katara: {type(exc).__name__}.", code="network") from exc

    def _rpc(self, method: str, params: dict[str, Any], *, timeout: Optional[float] = None) -> dict[str, Any]:
        request_id = self._new_id()
        response = self._post({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}, timeout)
        if response.is_redirect:
            raise KataraError("Katara answered with a redirect; refusing to follow it with credentials. "
                              "Check the configured MCP URL.", code="redirect", status=response.status_code)
        if response.status_code >= 400:
            raise _http_error(response.status_code, response.text)
        session_id = response.headers.get("mcp-session-id")
        if session_id:
            self._session_id = session_id
        content_type = response.headers.get("content-type", "")
        if "text/event-stream" in content_type:
            message = _parse_sse(response.text, request_id)
        else:
            try:
                message = response.json()
            except ValueError:
                message = None
        if not isinstance(message, dict):
            raise KataraError("Katara returned an unreadable response.", code="protocol")
        if message.get("error"):
            error = message["error"] if isinstance(message["error"], dict) else {}
            raise KataraError(f"Katara error: {_short(error.get('message') or 'unknown error')}",
                              code="rpc_error", payload=error)
        result = message.get("result")
        if not isinstance(result, dict):
            raise KataraError("Katara returned no result.", code="protocol")
        return result

    def _notify(self, method: str) -> None:
        response = self._post({"jsonrpc": "2.0", "method": method}, timeout=None)
        if response.status_code >= 400:
            logger.debug("Katara notification %s answered HTTP %s", method, response.status_code)

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        result = self._rpc("initialize", {
            "protocolVersion": MCP_PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": {"name": CLIENT_NAME, "version": CLIENT_VERSION},
        })
        negotiated = result.get("protocolVersion")
        if isinstance(negotiated, str) and negotiated:
            self._protocol_version = negotiated
        try:
            self._notify("notifications/initialized")
        except KataraError as exc:  # stateless server: the notification is advisory
            logger.debug("Katara initialized notification failed: %s", exc)
        self._initialized = True

    # -- Public API ----------------------------------------------------------------------------

    def call_tool(self, name: str, arguments: dict[str, Any], *, timeout: Optional[float] = None) -> dict[str, Any]:
        """Call one Katara MCP tool and return its structured payload.

        Raises ``KataraError`` for transport/auth failures and for tool errors (``isError``); a tool
        error keeps Katara's safe envelope (``error``, ``message``, ``suggestion``) in ``payload``.
        """
        self._ensure_initialized()
        result = self._rpc("tools/call", {"name": name, "arguments": arguments}, timeout=timeout)
        payload = result.get("structuredContent")
        if not isinstance(payload, dict):
            payload = {}
            for item in result.get("content") or []:
                if isinstance(item, dict) and item.get("type") == "text":
                    try:
                        parsed = json.loads(item.get("text") or "")
                    except ValueError:
                        parsed = {"text": item.get("text") or ""}
                    if isinstance(parsed, dict):
                        payload = parsed
                        break
        if result.get("isError"):
            message = payload.get("message") or payload.get("error") or "Katara tool call failed"
            suggestion = payload.get("suggestion")
            text = f"{message} {suggestion}" if suggestion else str(message)
            raise KataraError(_short(text), code=str(payload.get("error") or "tool_error"), payload=payload)
        return payload

    def list_tools(self, *, timeout: Optional[float] = None) -> list[str]:
        """Tool names the token can see (used by the connection check)."""
        self._ensure_initialized()
        result = self._rpc("tools/list", {}, timeout=timeout)
        return [t.get("name") for t in result.get("tools") or [] if isinstance(t, dict) and t.get("name")]

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:  # pragma: no cover - best effort
            pass
