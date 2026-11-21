"""Katara's config panel in the Hermes dashboard and Desktop.

Non-secret values land in ``$HERMES_HOME/katara/config.json`` (flat JSON storage); the API key goes to
the profile's ``.env`` as ``KATARA_API_KEY``. Loaded by path: imports only the pure-data schema module.
"""

from plugins.memory.config_schema import (
    KIND_BOOL,
    KIND_NUMBER,
    KIND_SECRET,
    KIND_SELECT,
    KIND_TEXT,
    ProviderConfigSchema,
    ProviderField,
    ProviderFieldOption,
)

CONFIG_SCHEMA = ProviderConfigSchema(
    name="katara",
    label="Katara",
    docs_url="https://github.com/katara-ai-inc/katara-hermes#configuration",
    fields=(
        ProviderField(
            key="api_key",
            label="API token",
            kind=KIND_SECRET,
            env_key="KATARA_API_KEY",
            placeholder="Enter Katara API token",
            description="Katara → Organization → API access. Grant Memory read (and Memory write to save facts).",
            inline=True,
        ),
        ProviderField(
            key="source_id",
            label="Memory source",
            kind=KIND_TEXT,
            placeholder="token default",
            description="Katara Brain-backed source alias. Blank uses the default saved for this token.",
            env_fallbacks=("KATARA_SOURCE_ID",),
            inline=True,
        ),
        ProviderField(key="auto_recall", label="Recall before each turn", kind=KIND_BOOL, default="true", inline=True),
        ProviderField(key="mirror_memory_writes", label="Mirror built-in memory writes", kind=KIND_BOOL,
                      default="true", inline=True,
                      description="Save what Hermes adds to MEMORY.md / USER.md as Katara facts."),
        ProviderField(key="read_only", label="Read only", kind=KIND_BOOL, default="false",
                      description="Hide katara_remember / katara_forget and never write to Katara.", group="Access"),
        ProviderField(key="recall_limit", label="Results per turn", kind=KIND_NUMBER, placeholder="8",
                      description="1 to 50 facts and knowledge results.", group="Recall"),
        ProviderField(key="recall_budget_tokens", label="Recall token budget", kind=KIND_NUMBER, placeholder="1200",
                      description="Server-side packing budget for automatic recall (100 to 8000).", group="Recall"),
        ProviderField(key="recall_timeout", label="Recall timeout (s)", kind=KIND_NUMBER, placeholder="6",
                      description="1 to 30 seconds; automatic recall is skipped when Katara is slower.", group="Recall"),
        ProviderField(
            key="evidence_kind",
            label="Knowledge lane",
            kind=KIND_SELECT,
            default="",
            options=(
                ProviderFieldOption("", "Brain (default)", "Facts, Wiki pages and original sources"),
                ProviderFieldOption("wiki", "Wiki", "Wiki pages only"),
                ProviderFieldOption("original_sources", "Original sources", "Uploaded documents only"),
            ),
            group="Recall",
        ),
        ProviderField(key="mcp_url", label="MCP endpoint", kind=KIND_TEXT, placeholder="https://mcp.katara.ai/mcp",
                      env_fallbacks=("KATARA_MCP_URL",), group="Advanced"),
        ProviderField(key="org_id", label="Organization id", kind=KIND_TEXT, placeholder="from token",
                      description="Only needed to pin an organization; API tokens already carry one.",
                      env_fallbacks=("KATARA_ORG_ID",), group="Advanced"),
    ),
)
