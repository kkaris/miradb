"""Data for the server's /docs page, built from its registered tools and
resources so the reference never goes out of date."""

import inspect

from mcp.server.fastmcp import FastMCP

# (key, label) in display order
GROUPS = [
    ("query", "Search & query"),
    ("inspect", "Inspect"),
    ("edit", "Edit"),
    ("transform", "Transform"),
    ("compare", "Compare"),
    ("simulate", "Simulate"),
]

_INSPECT = {
    "mira_get_model",
    "mira_get_equations",
    "mira_draw_model",
    "mira_export_model",
    "mira_list_workspace",
}
_TRANSFORM = {
    "mira_stratify",
    "mira_simplify_rate_laws",
    "mira_eliminate_unused_parameters",
    "mira_merge_duplicate_parameters",
    "mira_aggregate_parameters",
    "mira_counts_to_dimensionless",
    "mira_merge_models",
    "mira_apply_scenario",
}
_COMPARE = {
    "mira_compare_models",
    "mira_compare_concepts",
    "mira_compare_context",
    "mira_model_delta",
    "mira_draw_model_delta",
}
_SIMULATE = {"simulate_model", "get_simulation"}


def tool_group(name: str) -> str:
    """Return the GROUPS key a tool is listed under."""
    if name.startswith("db_"):
        return "query"
    for group, names in (
        ("inspect", _INSPECT),
        ("transform", _TRANSFORM),
        ("compare", _COMPARE),
        ("simulate", _SIMULATE),
    ):
        if name in names:
            return group
    # The remaining mira_add_* / mira_remove_* / mira_update_* / mira_set_*
    return "edit"


async def docs_payload(mcp: FastMCP) -> dict:
    """Describe every tool and resource the server exposes."""
    tools = await mcp.list_tools()
    resources = await mcp.list_resources()
    templates = await mcp.list_resource_templates()
    return {
        "server": mcp.name,
        "groups": [{"key": key, "label": label} for key, label in GROUPS],
        "tools": [
            {
                "name": tool.name,
                "group": tool_group(tool.name),
                "description": inspect.cleandoc(tool.description or ""),
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ],
        "resources": [
            {
                "uri": str(res.uri),
                "description": res.description or "",
                "mime_type": res.mimeType,
            }
            for res in resources
        ]
        + [
            {
                "uri": tmpl.uriTemplate,
                "description": tmpl.description or "",
                "mime_type": tmpl.mimeType,
            }
            for tmpl in templates
        ],
    }
