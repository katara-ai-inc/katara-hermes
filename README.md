# Katara memory for Hermes Agent

Use [Katara](https://katara.ai) as the long-term memory of [Hermes Agent](https://hermes-agent.nousresearch.com).
Before each turn, Hermes recalls relevant facts and Brain/Wiki knowledge from your Katara organization. The
agent gets tools to search, save and expire facts, and whatever Hermes writes to its built-in memory is
mirrored into Katara, so every agent connected to the same Brain sees it.

The plugin talks to Katara's MCP memory endpoint (`https://mcp.katara.ai/mcp`) with an organization API token.
It has no Python dependencies beyond what Hermes already ships (`httpx`).

## What it does

| Hermes surface | Katara behaviour |
|---|---|
| Automatic recall (each non-trivial prompt) | `recall` with your prompt as the query and a server-side token budget; the result is injected inside `<katara-memory>` tags with a `🧠 Katara — recalled N` indicator |
| `katara_recall` tool | Search facts plus Brain/Wiki knowledge (optionally one entity, one evidence lane, or facts since a date) |
| `katara_entity` tool | One person / company / project card (summary, aliases, open threads); no LLM call |
| `katara_remember` tool | Save one durable fact (kind, entity, optional TTL), with automatic provenance `Hermes Agent (<profile>) conversation on <platform>, <date>` |
| `katara_forget` tool | Expire a fact by `fact_id` (Katara keeps the audit trail) |
| Built-in memory writes | Additions and edits Hermes makes to `MEMORY.md` / `USER.md` are saved as Katara facts (`preference` for the user profile, `fact` otherwise). Removals are not mirrored |

Writes carry a `request_id`. If the network drops, the plugin retries once with the same id, and Katara
deduplicates the retry. Subagents, cron runs and other non-primary agents never write automatically.

## Requirements

- Hermes Agent `0.22.0` or newer.
- A Katara organization with a **Brain**.
- A Katara **API token** for that organization:
  1. In Katara, open **Organization → API access** and create a token.
  2. Grant **Agent read**, **Memory read** and **Document read**, plus **Memory write** if the agent should
     save and forget facts. Without write, set `read_only` (see below). Without Document read, recall
     covers facts and Wiki pages only (the plugin falls back automatically and logs it once).
  3. The token acts on behalf of the person who owns it, so that person must be able to **view** the Brain
     (and **edit** it for writes). A **Personal** Brain works only with a token owned by the Brain's owner.
  4. Select the token's **default memory source** (a Brain-backed source) in the same screen, or set
     `source_id` in the plugin config. Katara refuses memory calls when neither is set.

## Install

```bash
hermes plugins install katara          # once listed in the Hermes plugin catalog
# or, before that:
hermes plugins install katara-ai-inc/katara-hermes
```

Then select Katara as the memory provider and enter the token:

```bash
hermes memory setup                    # pick "katara", paste the API token
hermes katara status                  # checks the token, its grants and the memory source
```

`hermes memory setup` saves the token to the active profile's `.env` as `KATARA_API_KEY`. Each Hermes profile
has its own token and settings. You can also use the Desktop: **Capabilities → Tools → Memory → Katara**.

## Configuration

Settings live in `$HERMES_HOME/katara/config.json` (written by `hermes memory setup` and the dashboard panel).
The environment variables in the table override the file.

| Key | Env override | Default | Meaning |
|---|---|---|---|
| API token | `KATARA_API_KEY` | (required) | Organization API token with Agent read + Memory read + Document read (+ Memory write) |
| `source_id` | `KATARA_SOURCE_ID` | token default | Brain-backed source alias used for every call |
| `auto_recall` | | `true` | Recall before each turn |
| `mirror_memory_writes` | | `true` | Mirror built-in memory additions/edits as Katara facts |
| `read_only` | | `false` | Hide `katara_remember` / `katara_forget` and never write |
| `recall_limit` | | `8` | Facts and knowledge results per recall (1–50) |
| `recall_budget_tokens` | | `1200` | Server-side packing budget for automatic recall (100–8000) |
| `recall_timeout` | | `6` | Seconds before automatic recall gives up (1–30); Hermes also bounds it |
| `evidence_kind` | | Brain | `wiki` or `original_sources` to restrict the knowledge lane |
| `mcp_url` | `KATARA_MCP_URL` | `https://mcp.katara.ai/mcp` | Katara MCP endpoint (`https://` only; `http://localhost` allowed for development) |
| `org_id` | `KATARA_ORG_ID` | from token | Pin an organization (sent as `X-Katara-Org-Id`); API tokens already carry one |

## Privacy and data flow

These are the things to know before you install:

- **Network calls go only to Katara**, at the configured MCP endpoint (`mcp.katara.ai` by default), over HTTPS,
  with your API token as a bearer credential. The plugin never follows redirects with the token.
- **Every non-trivial prompt you send is used as a recall query**, sent to Katara (up to 2,000 characters).
  Turn this off with `auto_recall: false`.
- **What Hermes saves to its built-in memory is copied to Katara** as facts, with `world` visibility, which
  means every agent connected to the same Brain can read them. Turn this off with `mirror_memory_writes: false`.
  The plugin does not upload conversation transcripts.
- **Facts the agent saves with `katara_remember` are stored in your Katara organization** under the same visibility.
- The API token is read from the active Hermes profile's secret store, never from files. The plugin stores no
  credentials and keeps no local data besides `$HERMES_HOME/katara/config.json`.
- No telemetry, no shell commands, no background processes. Mirroring uses short-lived background threads.

## Troubleshooting

| Message | Fix |
|---|---|
| `Katara rejected the API key (401)` | The token is wrong, expired or revoked; create a new one |
| `Katara denied access (403)` | Add the Agent read, Memory read and Document read (and Memory write) grants to the token |
| Warning: "cannot search original sources" | The token lacks Document read; recall still covers facts and Wiki pages |
| `scope_denied` / `SCOPE_DENIED` ("The current credential cannot perform this operation") | The token is missing Agent read, Memory read or (for writes) Memory write, or its owner cannot view/edit the Brain (Personal Brains: owner only) |
| `DEFAULT_SOURCE_REQUIRED` ("Configure an authorized Brain-backed default QnA alias…") | The token has no default memory source: choose one for the token in Katara (Organization → API access), or set `source_id` |
| `Katara rate limit reached (429)` | Lower usage or wait; automatic recall skips silently on errors |
| No `<katara-memory>` block | Short acknowledgements ("ok", "thanks") skip recall on purpose; run `hermes katara status`; configuration errors are also logged once as a warning |

## Development

The repository root is the plugin directory. Tests use only `unittest` and `httpx`, so they run inside the Hermes
virtualenv with nothing to install (pytest collects them too):

```bash
PYTHONPATH=/path/to/hermes-agent /path/to/hermes-agent/venv/bin/python -m unittest discover -s tests
hermes plugins validate .              # the same check the plugin catalog CI runs
```

## License

MIT; see [LICENSE](LICENSE).
