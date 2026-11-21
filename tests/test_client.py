"""KataraMCPClient against a fake Streamable HTTP MCP endpoint."""

from __future__ import annotations

import unittest

import httpx

from _support import RECALL_PAYLOAD, FakeKatara, load_plugin

load_plugin()
from hermes_katara.client import KataraError, KataraMCPClient  # noqa: E402


class ClientTests(unittest.TestCase):
    def _client(self, fake: FakeKatara, **kwargs) -> KataraMCPClient:
        client = KataraMCPClient(fake.token, "https://mcp.example.test/mcp", transport=fake.transport, **kwargs)
        self.addCleanup(client.close)
        return client

    def test_initializes_once_then_calls_tools(self):
        fake = FakeKatara({"recall": lambda a: RECALL_PAYLOAD})
        client = self._client(fake)
        self.assertEqual(client.call_tool("recall", {"query": "x"})["total"], 2)
        client.call_tool("recall", {"query": "y"})
        methods = [r["method"] for r in fake.requests]
        self.assertEqual(methods, ["initialize", "notifications/initialized", "tools/call", "tools/call"])
        init = fake.requests[0]["params"]
        self.assertEqual(init["clientInfo"]["name"], "katara-hermes")
        self.assertEqual(fake.headers[-1]["mcp-protocol-version"], "2025-06-18")
        self.assertIn("text/event-stream", fake.headers[-1]["accept"])
        self.assertNotIn("x-katara-org-id", fake.headers[-1])

    def test_sse_responses_are_parsed(self):
        fake = FakeKatara({"recall": lambda a: RECALL_PAYLOAD}, sse=True)
        self.assertEqual(self._client(fake).call_tool("recall", {})["facts"][0]["fact_id"], "f-1")

    def test_text_content_fallback_when_no_structured_content(self):
        fake = FakeKatara({"entity": lambda a: {"found": False}})
        original = fake._reply

        def strip_structured(request_id, result):
            result = {k: v for k, v in result.items() if k != "structuredContent"}
            return original(request_id, result)

        fake._reply = strip_structured
        self.assertEqual(self._client(fake).call_tool("entity", {"name": "x"}), {"found": False})

    def test_org_header_only_when_pinned(self):
        fake = FakeKatara({"recall": lambda a: RECALL_PAYLOAD})
        self._client(fake, org_id="org_123").call_tool("recall", {})
        self.assertEqual(fake.headers[-1]["x-katara-org-id"], "org_123")

    def test_tool_error_envelope_becomes_katara_error(self):
        envelope = {"error": "scope_denied", "message": "No default source.", "detail": "DEFAULT_SOURCE_REQUIRED",
                    "suggestion": "Configure a default Brain alias for this credential.", "protocol_version": 1}
        fake = FakeKatara({"recall": lambda a: (envelope, True)})
        with self.assertRaises(KataraError) as ctx:
            self._client(fake).call_tool("recall", {})
        self.assertEqual(ctx.exception.code, "scope_denied")
        self.assertIn("Configure a default Brain alias", str(ctx.exception))
        self.assertEqual(ctx.exception.payload["detail"], "DEFAULT_SOURCE_REQUIRED")

    def test_bad_token_maps_to_unauthorized(self):
        fake = FakeKatara()
        client = KataraMCPClient("wrong", "https://mcp.example.test/mcp", transport=fake.transport)
        self.addCleanup(client.close)
        with self.assertRaises(KataraError) as ctx:
            client.call_tool("recall", {})
        self.assertEqual((ctx.exception.code, ctx.exception.status), ("unauthorized", 401))
        self.assertIn("KATARA_API_KEY", str(ctx.exception))

    def test_forbidden_mentions_memory_grant(self):
        fake = FakeKatara()
        fake.fail_next = [403]
        with self.assertRaises(KataraError) as ctx:
            self._client(fake).call_tool("recall", {})
        self.assertEqual(ctx.exception.code, "forbidden")
        self.assertIn("Memory read", str(ctx.exception))

    def test_timeout_and_network_errors_are_classified(self):
        fake = FakeKatara()
        fake.fail_next = [httpx.ReadTimeout("slow")]
        with self.assertRaises(KataraError) as ctx:
            self._client(fake).call_tool("recall", {})
        self.assertEqual(ctx.exception.code, "timeout")
        fake.fail_next = [httpx.ConnectError("down")]
        with self.assertRaises(KataraError) as ctx:
            self._client(fake).call_tool("recall", {})
        self.assertEqual(ctx.exception.code, "network")

    def test_redirects_are_not_followed(self):
        def redirect(request):
            return httpx.Response(307, headers={"location": "https://evil.example/collect"})

        client = KataraMCPClient("kt", "https://mcp.example.test/mcp", transport=httpx.MockTransport(redirect))
        self.addCleanup(client.close)
        with self.assertRaises(KataraError) as ctx:
            client.call_tool("recall", {})
        self.assertEqual(ctx.exception.code, "redirect")

    def test_rejects_plain_http_except_localhost(self):
        with self.assertRaises(ValueError):
            KataraMCPClient("kt", "http://mcp.katara.ai/mcp")
        KataraMCPClient("kt", "http://localhost:8080/mcp").close()
        with self.assertRaises(ValueError):
            KataraMCPClient("", "https://mcp.katara.ai/mcp")

    def test_list_tools(self):
        fake = FakeKatara({"recall": lambda a: {}, "remember": lambda a: {}})
        self.assertEqual(self._client(fake).list_tools(), ["recall", "remember"])


if __name__ == "__main__":
    unittest.main()
