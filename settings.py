"""Katara provider settings: profile-scoped env > ``$HERMES_HOME/katara/config.json`` > defaults.

Kept free of provider imports so ``cli.py`` can use it without loading the provider package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from agent.secret_scope import get_secret

from .client import DEFAULT_MCP_URL

API_KEY_ENV = "KATARA_API_KEY"
EVIDENCE_KINDS = ("brain", "wiki", "original_sources")
_CONFIG_DIR = "katara"
_CONFIG_FILE = "config.json"
_BOOL_WORDS = {**dict.fromkeys(("true", "1", "yes", "y", "on"), True),
               **dict.fromkeys(("false", "0", "no", "n", "off"), False)}

DEFAULTS: dict[str, Any] = {
    "mcp_url": DEFAULT_MCP_URL,
    "source_id": "",
    "org_id": "",
    "auto_recall": True,
    "recall_limit": 8,
    "recall_budget_tokens": 1200,
    "recall_timeout": 6.0,
    "evidence_kind": "",
    "mirror_memory_writes": True,
    "read_only": False,
}
# Non-secret keys with an env override (read through the profile-scoped secret store).
ENV_OVERRIDES = {"mcp_url": "KATARA_MCP_URL", "source_id": "KATARA_SOURCE_ID", "org_id": "KATARA_ORG_ID"}


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _BOOL_WORDS.get(value.strip().lower(), default)
    return default


def clamp(value: Any, default: Any, lo: float, hi: float, cast: Callable[[Any], Any]) -> Any:
    try:
        return max(lo, min(hi, cast(value)))
    except (TypeError, ValueError):
        return default


def config_path(hermes_home: str | Path) -> Path:
    return Path(hermes_home) / _CONFIG_DIR / _CONFIG_FILE


def read_config_file(hermes_home: str | Path) -> dict[str, Any]:
    path = config_path(hermes_home)
    try:
        from utils import read_json_or_empty
        data = read_json_or_empty(path)
    except ImportError:  # pragma: no cover - very old Hermes
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = {}
    return data if isinstance(data, dict) else {}


def write_config_file(hermes_home: str | Path, values: dict[str, Any]) -> None:
    """Merge known non-secret keys into ``config.json`` (atomic write)."""
    path = config_path(hermes_home)
    current = read_config_file(hermes_home)
    current.update({k: v for k, v in (values or {}).items() if k in DEFAULTS})
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from utils import atomic_json_write
        atomic_json_write(path, current)
    except ImportError:  # pragma: no cover - very old Hermes
        path.write_text(json.dumps(current, indent=2), encoding="utf-8")


def resolve_settings(hermes_home: str | Path) -> dict[str, Any]:
    stored = read_config_file(hermes_home)
    raw = {**DEFAULTS, **{k: v for k, v in stored.items() if k in DEFAULTS and v not in (None, "")}}
    for key, env in ENV_OVERRIDES.items():
        value = (get_secret(env, "") or "").strip()
        if value:
            raw[key] = value
    evidence = str(raw.get("evidence_kind") or "").strip().lower()
    return {
        "mcp_url": str(raw["mcp_url"] or DEFAULT_MCP_URL).strip().rstrip("/") or DEFAULT_MCP_URL,
        "source_id": str(raw["source_id"] or "").strip(),
        "org_id": str(raw["org_id"] or "").strip(),
        "auto_recall": as_bool(raw["auto_recall"], True),
        "recall_limit": clamp(raw["recall_limit"], 8, 1, 50, int),
        "recall_budget_tokens": clamp(raw["recall_budget_tokens"], 1200, 100, 8000, int),
        "recall_timeout": clamp(raw["recall_timeout"], 6.0, 1.0, 30.0, float),
        "evidence_kind": evidence if evidence in EVIDENCE_KINDS else "",
        "mirror_memory_writes": as_bool(raw["mirror_memory_writes"], True),
        "read_only": as_bool(raw["read_only"], False),
    }
