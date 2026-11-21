"""KataraMemoryProvider lifecycle, tools, recall injection and mirroring (fake Katara endpoint)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx

from _support import RECALL_PAYLOAD, FakeKatara, load_plugin

katara = load_plugin()
from hermes_katara.settings import resolve_settings, write_config_file  # noqa: E402

_KATARA_ENV = ("KATARA_API_KEY", "KATARA_MCP_URL", "KATARA_SOURCE_ID", "KATARA_ORG_ID")


class ProviderTestCase(unittest.TestCase):
    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="katara-home-"))
        env = {k: v for k, v in os.environ.items() if k not in _KATARA_ENV}
        env["KATARA_API_KEY"] = "kt_test"
        patcher = mock.patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.remembered: list[dict] = []
        self.fake = FakeKatara({
            "recall": lambda a: RECALL_PAYLOAD,
            "entity": lambda a: {"protocol_version": 1, "found": True, "latency_ms": 3,
                                 "card": {"entity": {"slug": "people/alice", "title": "Alice", "type": "person"}}},
            "remember": self._remember,
            "forget": self._forget,
        })

    # Shapes recorded from the live endpoint: the fact id is top-level ``id``; status/fact_id live in the
    # committed write receipt's outcome.
    def _remember(self, args):
        self.remembered.append(args)
        fact_id = f"f-{len(self.remembered) + 10}"
        return {"protocol_version": 1, "id": fact_id, "entity_slug": None, "degraded_dedup": False,
                "katara": {"write_request": {
                    "request_id": args.get("request_id"), "state": "committed", "retry_after_ms": None,
                    "outcome": {"fact_id": fact_id, "legacy_id": 1, "created": True, "status": "inserted",
                                "status_text": "Fact stored.", "entity_slug": None, "valid_until": None,
                                "expired": False, "protocol_version": 1},
                    "persistence": {"mode": "database"}, "compacted": False}}}

    def _forget(self, args):
        return {"protocol_version": 1, "id": args["id"], "expired": True, "reason": args.get("reason"),
                "katara": {"write_request": {
                    "request_id": args.get("request_id"), "state": "committed", "retry_after_ms": None,
                    "outcome": {"fact_id": args["id"], "withdrawn": True, "expired": True,
                                "reason": args.get("reason")}}}}

    def provider(self, *, transport=None, **init_kwargs):
        provider = katara.KataraMemoryProvider(transport=transport or self.fake.transport)
        provider.initialize("session-1", hermes_home=str(self.home), platform="cli",
                            agent_identity="xavi", **init_kwargs)
        self.addCleanup(provider.shutdown)
        return provider


class AvailabilityTests(ProviderTestCase):
    def test_available_only_with_api_key(self):
        provider = katara.KataraMemoryProvider()
        self.assertTrue(provider.is_available())
        del os.environ["KATARA_API_KEY"]
        self.assertFalse(provider.is_available())
        self.assertIn("KATARA_API_KEY", provider.unavailable_reason())

    def test_initialize_without_key_stays_inert(self):
        del os.environ["KATARA_API_KEY"]
        provider = self.provider()
        self.assertEqual(provider.system_prompt_block(), "")
        self.assertEqual(provider.prefetch("what do I prefer?"), "")
        self.assertIn("not connected", json.loads(provider.handle_tool_call("katara_recall", {"query": "x"}))["error"])
        self.assertEqual(self.fake.requests, [])

    def test_register_hands_out_a_new_instance_each_time(self):
        seen = []
        ctx = mock.Mock(register_memory_provider=seen.append)
        katara.register(ctx)
        katara.register(ctx)
        self.assertEqual(len(seen), 2)
        self.assertIsNot(seen[0], seen[1])
        self.assertEqual(seen[0].name, "katara")


class SettingsTests(ProviderTestCase):
    def test_defaults(self):
        settings = resolve_settings(self.home)
        self.assertEqual(settings["mcp_url"], "https://mcp.katara.ai/mcp")
        self.assertTrue(settings["auto_recall"])
        self.assertTrue(settings["mirror_memory_writes"])
        self.assertFalse(settings["read_only"])

    def test_file_then_env_precedence_and_coercion(self):
        write_config_file(self.home, {"source_id": "file-src", "auto_recall": "false", "recall_limit": "500",
                                      "evidence_kind": "WIKI", "unknown_key": "dropped", "api_key": "never-stored"})
        stored = json.loads((self.home / "katara" / "config.json").read_text())
        self.assertNotIn("api_key", stored)
        self.assertNotIn("unknown_key", stored)
        settings = resolve_settings(self.home)
        self.assertEqual(settings["source_id"], "file-src")
        self.assertFalse(settings["auto_recall"])
        self.assertEqual(settings["recall_limit"], 50)
        self.assertEqual(settings["evidence_kind"], "wiki")
        os.environ["KATARA_SOURCE_ID"] = "env-src"
        self.assertEqual(resolve_settings(self.home)["source_id"], "env-src")

    def test_save_config_merges(self):
        provider = katara.KataraMemoryProvider()
        provider.save_config({"source_id": "a"}, str(self.home))
        provider.save_config({"read_only": True}, str(self.home))
        self.assertEqual(json.loads((self.home / "katara" / "config.json").read_text()),
                         {"source_id": "a", "read_only": True})

    def test_config_schema_declares_api_key_secret(self):
        schema = katara.KataraMemoryProvider().get_config_schema()
        api_key = next(f for f in schema if f["key"] == "api_key")
        self.assertTrue(api_key["secret"])
        self.assertEqual(api_key["env_var"], "KATARA_API_KEY")


class RecallTests(ProviderTestCase):
    def test_prefetch_injects_formatted_block_and_status(self):
        provider = self.provider()
        text = provider.prefetch("how should I report status?", session_id="session-1")
        self.assertTrue(text.startswith("<katara-memory>"))
        self.assertTrue(text.endswith("</katara-memory>"))
        self.assertIn("[preference] Prefers concise status updates [fact_id: f-1]", text)
        self.assertIn("(about people/vcastellm)", text)
        self.assertIn("Katara MCP (projects/katara-mcp): Stateless Streamable HTTP endpoint.", text)
        status = provider.recall_status()
        self.assertEqual((status.provider_label, status.count), ("Katara", 3))
        args = self.fake.calls("recall")[0]["arguments"]
        self.assertEqual(args, {"query": "how should I report status?", "limit": 8, "budget_tokens": 1200})

    def test_prefetch_passes_configured_source_and_lane(self):
        write_config_file(self.home, {"source_id": "brain-alias", "evidence_kind": "wiki"})
        self.provider().prefetch("q")
        args = self.fake.calls("recall")[0]["arguments"]
        self.assertEqual((args["source_id"], args["evidence_kind"]), ("brain-alias", "wiki"))

    def test_prefetch_disabled_or_failing_returns_empty_and_clears_status(self):
        provider = self.provider()
        provider.prefetch("first")
        self.assertIsNotNone(provider.recall_status())
        self.fake.fail_next = [500]
        self.assertEqual(provider.prefetch("second"), "")
        self.assertIsNone(provider.recall_status())
        write_config_file(self.home, {"auto_recall": False})
        quiet = self.provider()
        self.assertEqual(quiet.prefetch("third"), "")
        # The fake records only answered calls: "first" succeeded, "second" got a 500, "third" never left.
        self.assertEqual(len(self.fake.calls("recall")), 1)

    def test_missing_default_source_warns_once_then_stays_quiet(self):
        envelope = {"error": "invalid_params", "message": "The operation arguments are invalid.",
                    "detail": "DEFAULT_SOURCE_REQUIRED", "protocol_version": 1,
                    "suggestion": "Configure an authorized Brain-backed default QnA alias for this client/credential."}
        self.fake.tools["recall"] = lambda a: (envelope, True)
        provider = self.provider()
        with self.assertLogs(katara.logger, level="DEBUG") as logs:
            self.assertEqual(provider.prefetch("first"), "")
            self.assertEqual(provider.prefetch("second"), "")
        warnings = [r for r in logs.records if r.levelname == "WARNING"]
        self.assertEqual(len(warnings), 1)
        self.assertIn("hermes katara status", warnings[0].getMessage())
        self.assertIn("default QnA alias", warnings[0].getMessage())

    def test_transient_recall_failure_is_not_a_warning(self):
        provider = self.provider()
        self.fake.fail_next = [503]
        with self.assertLogs(katara.logger, level="DEBUG") as logs:
            self.assertEqual(provider.prefetch("q"), "")
        self.assertEqual([r.levelname for r in logs.records], ["DEBUG"])

    def _deny_original_sources(self):
        denied = {"error": "scope_denied", "message": "The current credential cannot perform this operation.",
                  "detail": "PERMISSION_DENIED", "protocol_version": 1}
        self.fake.tools["recall"] = lambda a: (
            (denied, True) if a.get("query") and a.get("evidence_kind") in (None, "brain", "original_sources")
            else RECALL_PAYLOAD)

    def test_token_without_document_read_falls_back_to_wiki_once(self):
        self._deny_original_sources()
        provider = self.provider()
        with self.assertLogs(katara.logger, level="WARNING") as logs:
            self.assertIn("<katara-memory>", provider.prefetch("first question"))
            self.assertIn("<katara-memory>", provider.prefetch("second question"))
        self.assertEqual(len(logs.records), 1)
        self.assertIn("Document read", logs.records[0].getMessage())
        lanes = [c["arguments"].get("evidence_kind") for c in self.fake.calls("recall")]
        self.assertEqual(lanes, [None, "wiki", "wiki"])  # denied once, then straight to wiki

    def test_explicit_lane_is_never_narrowed(self):
        self._deny_original_sources()
        out = json.loads(self.provider().handle_tool_call(
            "katara_recall", {"query": "x", "evidence_kind": "original_sources"}))
        self.assertEqual(out["code"], "scope_denied")
        self.assertEqual(len(self.fake.calls("recall")), 1)

    def test_recall_tool_notes_the_wiki_fallback(self):
        self._deny_original_sources()
        out = json.loads(self.provider().handle_tool_call("katara_recall", {"query": "x"}))
        self.assertEqual(out["total"], 2)
        self.assertIn("Wiki only", out["note"])

    def test_empty_recall_injects_nothing(self):
        self.fake.tools["recall"] = lambda a: {"protocol_version": 1, "facts": [], "results": [], "total": 0}
        provider = self.provider()
        self.assertEqual(provider.prefetch("anything"), "")
        self.assertIsNone(provider.recall_status())

    def test_prefetch_block_is_bounded(self):
        many = {"facts": [{"fact_id": f"f{i}", "fact": "x" * 1500, "kind": "fact"} for i in range(40)], "total": 40}
        text, count = katara.format_recall_block(many, max_chars=5000)
        self.assertLessEqual(len(text), 5200)
        self.assertIn("more results truncated", text)
        self.assertEqual(count, 40)


class ToolTests(ProviderTestCase):
    def call(self, provider, name, args):
        return json.loads(provider.handle_tool_call(name, args))

    def test_tool_surface_and_read_only(self):
        names = [s["name"] for s in self.provider().get_tool_schemas()]
        self.assertEqual(names, ["katara_recall", "katara_entity", "katara_remember", "katara_forget"])
        write_config_file(self.home, {"read_only": True})
        provider = self.provider()
        self.assertEqual([s["name"] for s in provider.get_tool_schemas()], ["katara_recall", "katara_entity"])
        self.assertIn("Unknown tool", self.call(provider, "katara_remember", {"fact": "x"})["error"])
        self.assertNotIn("katara_remember", provider.system_prompt_block())
        self.assertEqual(self.remembered, [])

    def test_recall_tool_trims_payload(self):
        out = self.call(self.provider(), "katara_recall", {"query": "repo", "limit": 3, "evidence_kind": "wiki"})
        self.assertEqual(out["facts"][1], {"fact_id": "f-2", "fact": "Owns the Katara backend repo", "kind": "fact",
                                           "entity_slug": "people/vcastellm", "provenance": "user said in chat"})
        self.assertEqual(out["results"][0]["slug"], "projects/katara-mcp")
        self.assertNotIn("katara", out)
        self.assertEqual(self.fake.calls("recall")[0]["arguments"],
                         {"query": "repo", "limit": 3, "evidence_kind": "wiki"})

    def test_recall_tool_validates_arguments(self):
        provider = self.provider()
        self.assertIn("query or entity", self.call(provider, "katara_recall", {})["error"])
        self.assertIn("evidence_kind", self.call(provider, "katara_recall", {"query": "x", "evidence_kind": "web"})["error"])
        self.assertEqual(self.fake.calls(), [])

    def test_entity_tool(self):
        out = self.call(self.provider(), "katara_entity", {"name": "Alice"})
        self.assertTrue(out["found"])
        self.assertNotIn("protocol_version", out)

    def test_remember_tool_sends_provenance_and_request_id(self):
        out = self.call(self.provider(), "katara_remember",
                        {"fact": "Ships on Fridays", "entity": "Katara", "kind": "commitment", "ttl": "30d"})
        self.assertEqual(out, {"status": "inserted", "fact_id": "f-11"})
        sent = self.remembered[0]
        self.assertEqual((sent["fact"], sent["entity"], sent["kind"], sent["ttl"]),
                         ("Ships on Fridays", "Katara", "commitment", "30d"))
        self.assertTrue(sent["provenance"].startswith("Hermes Agent (xavi) conversation on cli, "))
        self.assertEqual(len(sent["request_id"]), 36)
        self.assertNotIn("visibility", sent)

    def test_remember_rejects_bad_kind(self):
        self.assertIn("kind must be", self.call(self.provider(), "katara_remember", {"fact": "x", "kind": "idea"})["error"])

    def test_write_timeout_retries_once_with_same_request_id(self):
        provider = self.provider()
        provider.prefetch("warm up the session")  # initialize first so the failure hits tools/call
        self.fake.fail_next = [httpx.ReadTimeout("slow")]
        out = self.call(provider, "katara_remember", {"fact": "Retry me"})
        self.assertEqual(out["status"], "inserted")
        self.assertEqual(len(self.remembered), 1)

    def test_forget_tool(self):
        out = self.call(self.provider(), "katara_forget", {"fact_id": "f-1", "reason": "outdated"})
        self.assertEqual(out, {"fact_id": "f-1", "expired": True, "reason": "outdated"})
        self.assertIn("request_id", self.fake.calls("forget")[0]["arguments"])

    def test_katara_tool_error_is_reported_with_suggestion(self):
        envelope = {"error": "scope_denied", "message": "Access denied.", "detail": "DEFAULT_SOURCE_REQUIRED",
                    "suggestion": "Configure an authorized Brain-backed default QnA alias."}
        self.fake.tools["recall"] = lambda a: (envelope, True)
        out = self.call(self.provider(), "katara_recall", {"query": "x"})
        self.assertEqual(out["code"], "scope_denied")
        self.assertEqual(out["detail"], "DEFAULT_SOURCE_REQUIRED")
        self.assertIn("Brain-backed default", out["suggestion"])


class MirrorTests(ProviderTestCase):
    def _drain(self, provider):
        for thread in list(provider._threads):
            thread.join(timeout=5)

    def test_builtin_memory_add_and_replace_are_mirrored(self):
        provider = self.provider()
        provider.on_memory_write("add", "user", "User prefers dark mode")
        provider.on_memory_write("replace", "memory", "Repo lives at ~/katara-app", {"previous_content": "old"})
        provider.on_memory_write("remove", "memory", "Gone", {"previous_content": "Gone"})
        self._drain(provider)
        self.assertEqual(len(self.remembered), 2)
        by_fact = {r["fact"]: r for r in self.remembered}
        self.assertEqual(by_fact["User prefers dark mode"]["kind"], "preference")
        self.assertIn("built-in user profile add", by_fact["User prefers dark mode"]["provenance"])
        self.assertEqual(by_fact["Repo lives at ~/katara-app"]["kind"], "fact")

    def test_mirror_skipped_for_subagents_read_only_or_disabled(self):
        sub = self.provider(agent_context="subagent")
        sub.on_memory_write("add", "memory", "from a subagent")
        self._drain(sub)
        write_config_file(self.home, {"mirror_memory_writes": False})
        off = self.provider()
        off.on_memory_write("add", "memory", "mirror disabled")
        self._drain(off)
        write_config_file(self.home, {"mirror_memory_writes": True, "read_only": True})
        ro = self.provider()
        ro.on_memory_write("add", "memory", "read only")
        self._drain(ro)
        self.assertEqual(self.remembered, [])

    def test_mirror_failure_never_raises(self):
        self.fake.tools["remember"] = lambda a: ({"error": "scope_denied", "message": "no write grant"}, True)
        provider = self.provider()
        provider.on_memory_write("add", "memory", "will fail")
        self._drain(provider)  # no exception escapes the background thread


class MemoryManagerIntegrationTests(ProviderTestCase):
    def test_routes_through_hermes_memory_manager(self):
        from agent.memory_manager import MemoryManager

        # The manager only serves callers from the profile the provider was initialized for.
        os.environ["HERMES_HOME"] = str(self.home)
        provider = katara.KataraMemoryProvider(transport=self.fake.transport)
        manager = MemoryManager()
        manager.add_provider(provider)
        manager.initialize_all(session_id="mm-1", platform="cli", hermes_home=str(self.home))
        self.addCleanup(manager.shutdown_all)
        tool_names = {s["name"] for s in manager.get_all_tool_schemas()}
        self.assertTrue({"katara_recall", "katara_remember"} <= tool_names)
        out = json.loads(manager.handle_tool_call("katara_recall", {"query": "status"}))
        self.assertEqual(out["total"], 2)
        context = manager.prefetch_all("how do I like status updates?", session_id="mm-1")
        self.assertIn("<katara-memory>", context)


if __name__ == "__main__":
    unittest.main()
