"""``hermes katara`` subcommands (shown while Katara is the active memory provider)."""

from __future__ import annotations


def _status(args) -> int:
    from agent.secret_scope import get_secret
    from hermes_constants import get_hermes_home

    from .client import KataraError, KataraMCPClient
    from .settings import API_KEY_ENV, resolve_settings

    settings = resolve_settings(get_hermes_home())
    api_key = (get_secret(API_KEY_ENV, "") or "").strip()
    print("Katara memory")
    print(f"  endpoint:  {settings['mcp_url']}")
    print(f"  source:    {settings['source_id'] or '(token default)'}")
    print(f"  api key:   {'set' if api_key else f'missing — set {API_KEY_ENV}'}")
    print(f"  recall:    {'on' if settings['auto_recall'] else 'off'} · mirror writes: "
          f"{'on' if settings['mirror_memory_writes'] else 'off'} · read only: {'yes' if settings['read_only'] else 'no'}")
    if not api_key:
        return 1
    client = KataraMCPClient(api_key, settings["mcp_url"], org_id=settings["org_id"], timeout=15.0)
    try:
        tools = client.list_tools()
        memory = [t for t in ("recall", "entity", "remember", "forget") if t in tools]
        print(f"  connected: yes · memory verbs visible to this token: {', '.join(memory) or 'none'}")
        # A read-only one-fact recall proves the token can reach a memory source.
        recall_args = {"limit": 1}
        if settings["source_id"]:
            recall_args["source_id"] = settings["source_id"]
        payload = client.call_tool("recall", recall_args)
        print(f"  source:    ok · recall returned {payload.get('total', 0)} fact(s)")
        return 0
    except KataraError as exc:
        print(f"  error:     {exc}")
        if exc.code == "scope_denied":
            print("  fix:       Give the token the Agent read, Memory read and Document read grants (Memory write\n"
                  "             to save facts) in Katara → Organization → API access. The token's owner must be able\n"
                  "             to view the Brain (edit it for writes); a Personal Brain works only with a token owned\n"
                  "             by its owner.")
        if exc.payload.get("detail") == "DEFAULT_SOURCE_REQUIRED":
            print("  fix:       In Katara, open Organization → API access and choose this token's default memory\n"
                  "             source (a Brain), or set source_id with `hermes memory setup` / KATARA_SOURCE_ID.")
        return 1
    finally:
        client.close()


def katara_command(args) -> int:
    sub = getattr(args, "katara_command", None)
    if sub == "status":
        return _status(args)
    print("Usage: hermes katara status")
    return 0


def register_cli(subparser) -> None:
    subs = subparser.add_subparsers(dest="katara_command")
    subs.add_parser("status", help="Show Katara memory configuration and test the token and memory source")
    subparser.set_defaults(func=katara_command)
