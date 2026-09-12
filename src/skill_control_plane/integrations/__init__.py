"""Agent-facing schemas, tool loops, context injection and history projection."""

from .reference_agent import (
    AGENT_INSTRUCTIONS,
    CAPABILITY_TOOLS,
    EVICTED_BODY_TOMBSTONE,
    SUPERSEDED_BODY_TOMBSTONE,
    AgentStepLimitError,
    ConversationClient,
    ExperimentalSkillAgent,
    LoadCapabilityArguments,
    LoadSkillBodyArguments,
    ReferenceSkillAgent,
    dto_payload,
    parse_json_object,
    parse_tool_arguments,
)

__all__ = [
    "AGENT_INSTRUCTIONS",
    "CAPABILITY_TOOLS",
    "EVICTED_BODY_TOMBSTONE",
    "SUPERSEDED_BODY_TOMBSTONE",
    "AgentStepLimitError",
    "ConversationClient",
    "ExperimentalSkillAgent",
    "LoadCapabilityArguments",
    "LoadSkillBodyArguments",
    "ReferenceSkillAgent",
    "dto_payload",
    "parse_json_object",
    "parse_tool_arguments",
]
