"""Strict NDJSON envelope, DTO JSON boundaries, and stable error codes."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from enum import StrEnum
from typing import Any

from skill_control_plane.runtime import (
    CapabilityDecision,
    CoverageClaim,
    StateSnapshotBundleV1,
    StateSnapshotMemberV1,
    StateSnapshotV1,
)

PROTOCOL_VERSION = "sidecar-protocol-v0.1"

INVALID_REQUEST = "INVALID_REQUEST"
INVALID_PARAMS = "INVALID_PARAMS"
UNKNOWN_METHOD = "UNKNOWN_METHOD"
CORE_VALIDATION_ERROR = "CORE_VALIDATION_ERROR"
NOT_READY = "NOT_READY"
INTERNAL_ERROR = "INTERNAL_ERROR"

ERROR_CODES = (
    INVALID_REQUEST,
    INVALID_PARAMS,
    UNKNOWN_METHOD,
    CORE_VALIDATION_ERROR,
    NOT_READY,
    INTERNAL_ERROR,
)


@dataclass(frozen=True, slots=True)
class Request:
    id: str
    method: str
    params: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ProtocolError(ValueError):
    """A stable protocol or method parameter error."""

    code: str
    message: str
    details: dict[str, Any] | None = None
    response_id: str | None = None

    def response(self) -> dict[str, Any]:
        error: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details is not None:
            error["details"] = encode_json(self.details)
        return {"id": self.response_id, "ok": False, "error": error}


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate object key: {key!r}")
        result[key] = value
    return result


def _reject_non_standard_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def parse_request(line: str) -> Request:
    try:
        payload = json.loads(
            line,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_standard_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as exc:
        raise ProtocolError(
            INVALID_REQUEST, "request must be one JSON object with unique keys"
        ) from exc

    if not isinstance(payload, dict):
        raise ProtocolError(INVALID_REQUEST, "request must be a JSON object")
    payload_id = payload.get("id")
    valid_id = payload_id if isinstance(payload_id, str) and payload_id else None
    if set(payload) - {"id", "method", "params"}:
        raise ProtocolError(
            INVALID_REQUEST, "request has unknown fields", response_id=valid_id
        )

    request_id = payload.get("id")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError(INVALID_REQUEST, "request id must be a non-empty string")
    method = payload.get("method")
    if not isinstance(method, str) or not method:
        raise ProtocolError(
            INVALID_REQUEST, "request method must be a non-empty string",
            response_id=request_id,
        )
    if "params" not in payload:
        raise ProtocolError(
            INVALID_PARAMS, "request params are required", response_id=request_id
        )
    params = payload["params"]
    if not isinstance(params, dict):
        raise ProtocolError(
            INVALID_PARAMS, "request params must be an object",
            response_id=request_id,
        )
    return Request(request_id, method, params)


def encode_json(value: Any) -> Any:
    """Encode Core DTOs without importing provider-specific serialization."""

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: encode_json(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, StrEnum):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): encode_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode_json(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"value is not JSON-compatible: {type(value).__name__}")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError(INVALID_PARAMS, f"{label} must be an object")
    return value


def _required_fields(value: dict[str, Any], names: set[str], label: str) -> None:
    missing = names - set(value)
    if missing:
        raise ProtocolError(
            INVALID_PARAMS,
            f"{label} is missing required fields",
            {"missing": sorted(missing)},
        )


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ProtocolError(INVALID_PARAMS, f"{label} must be a non-empty string")
    return value


def _string_list(value: Any, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProtocolError(INVALID_PARAMS, f"{label} must be an array")
    return tuple(_string(item, f"{label} item") for item in value)


def _optional_field(
    value: dict[str, Any], name: str, default: Any = None,
) -> Any:
    return value[name] if name in value else default


def decode_capability_decision(value: Any) -> CapabilityDecision:
    data = _object(value, "apply_capability params")
    allowed = {
        "action", "skill_ids", "reason", "target_bundle_id", "purpose",
        "coverage", "remaining_gaps",
    }
    unknown = set(data) - allowed
    if unknown:
        raise ProtocolError(
            INVALID_PARAMS, "apply_capability params have unknown fields",
            {"unknown": sorted(unknown)},
        )
    _required_fields(data, {"action", "skill_ids", "reason"}, "apply_capability params")

    action = _string(data["action"], "action")
    if action not in {"DIRECT", "EXTEND", "CREATE"}:
        raise ProtocolError(INVALID_PARAMS, "action must be DIRECT, EXTEND, or CREATE")
    skill_ids = _string_list(data["skill_ids"], "skill_ids")
    if not skill_ids:
        raise ProtocolError(INVALID_PARAMS, "skill_ids must not be empty")
    reason = _string(data["reason"], "reason")

    target_bundle_id: str | None
    purpose: str | None
    if action == "CREATE":
        target_bundle_id = _optional_field(data, "target_bundle_id")
        if target_bundle_id is not None:
            raise ProtocolError(INVALID_PARAMS, "CREATE cannot set target_bundle_id")
        purpose = _string(_optional_field(data, "purpose"), "CREATE purpose")
    else:
        target_bundle_id = _string(
            _optional_field(data, "target_bundle_id"), "target_bundle_id"
        )
        if "purpose" in data and data["purpose"] is not None:
            raise ProtocolError(INVALID_PARAMS, f"{action} cannot set purpose")
        purpose = None

    raw_coverage = _optional_field(data, "coverage", [])
    if not isinstance(raw_coverage, list):
        raise ProtocolError(INVALID_PARAMS, "coverage must be an array")
    coverage: list[CoverageClaim] = []
    for index, item in enumerate(raw_coverage):
        claim_data = _object(item, f"coverage item {index}")
        if set(claim_data) != {"need", "covered_by"}:
            raise ProtocolError(
                INVALID_PARAMS,
                f"coverage item {index} must contain only need and covered_by",
            )
        coverage.append(CoverageClaim(
            _string(claim_data["need"], f"coverage item {index} need"),
            _string(claim_data["covered_by"], f"coverage item {index} covered_by"),
        ))

    remaining_gaps = _string_list(
        _optional_field(data, "remaining_gaps", []), "remaining_gaps"
    )
    return CapabilityDecision(
        action=action,
        skill_ids=skill_ids,
        reason=reason,
        target_bundle_id=target_bundle_id,
        purpose=purpose,
        coverage=tuple(coverage),
        remaining_gaps=remaining_gaps,
    )


def decode_state_snapshot(value: Any) -> StateSnapshotV1:
    data = _object(value, "restore_state params")
    if set(data) != {"version", "store_fingerprint", "bundles"}:
        raise ProtocolError(
            INVALID_PARAMS,
            "restore_state params must contain only version, store_fingerprint, "
            "and bundles",
        )
    if data["version"] != "state-snapshot-v1":
        raise ProtocolError(
            INVALID_PARAMS,
            "unsupported state snapshot version",
            {"version": data["version"]},
        )
    store_fingerprint = _string(data["store_fingerprint"], "store_fingerprint")
    raw_bundles = data["bundles"]
    if not isinstance(raw_bundles, list):
        raise ProtocolError(INVALID_PARAMS, "bundles must be an array")

    bundles: list[StateSnapshotBundleV1] = []
    for bundle_index, raw_bundle in enumerate(raw_bundles):
        bundle_data = _object(raw_bundle, f"bundle {bundle_index}")
        if set(bundle_data) != {"bundle_id", "purpose", "members"}:
            raise ProtocolError(
                INVALID_PARAMS,
                f"bundle {bundle_index} must contain only bundle_id, purpose, "
                "and members",
            )
        raw_members = bundle_data["members"]
        if not isinstance(raw_members, list):
            raise ProtocolError(
                INVALID_PARAMS,
                f"bundle {bundle_index} members must be an array",
            )
        members: list[StateSnapshotMemberV1] = []
        for member_index, raw_member in enumerate(raw_members):
            member_data = _object(
                raw_member,
                f"bundle {bundle_index} member {member_index}",
            )
            if set(member_data) != {"skill_id", "member_role", "body_state"}:
                raise ProtocolError(
                    INVALID_PARAMS,
                    f"bundle {bundle_index} member {member_index} has invalid fields",
                )
            member_role = _string(member_data["member_role"], "member_role")
            body_state = _string(member_data["body_state"], "body_state")
            if member_role not in {"maintained", "direct"}:
                raise ProtocolError(INVALID_PARAMS, "invalid member_role")
            if body_state not in {"resident", "evicted"}:
                raise ProtocolError(INVALID_PARAMS, "invalid body_state")
            members.append(StateSnapshotMemberV1(
                skill_id=_string(member_data["skill_id"], "skill_id"),
                member_role=member_role,
                body_state=body_state,
            ))
        bundles.append(StateSnapshotBundleV1(
            bundle_id=_string(bundle_data["bundle_id"], f"bundle {bundle_index} id"),
            purpose=_string(bundle_data["purpose"], f"bundle {bundle_index} purpose"),
            members=tuple(members),
        ))
    return StateSnapshotV1(
        version="state-snapshot-v1",
        store_fingerprint=store_fingerprint,
        bundles=tuple(bundles),
    )
