"""Single-client sequential NDJSON sidecar dispatch and lifecycle."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TextIO

from skill_control_plane.discovery.bigmodel import (
    BigModelDenseRetriever,
    BigModelEmbeddingClient,
)
from skill_control_plane.runtime import (
    CapabilityDecision,
    ControlPlaneReadiness,
    SkillControlPlane,
)

from .protocol import (
    CORE_VALIDATION_ERROR,
    INVALID_REQUEST,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    NOT_READY,
    PROTOCOL_VERSION,
    ProtocolError,
    Request,
    decode_capability_decision,
    decode_state_snapshot,
    encode_json,
    parse_request,
)

SUPPORTED_METHODS = (
    "handshake",
    "begin_turn",
    "end_turn",
    "context_snapshot",
    "search_capability",
    "apply_capability",
    "load_skill_body",
    "mark_skill_body_evicted",
    "mark_all_skill_bodies_evicted",
    "turn_audit",
    "export_state",
    "restore_state",
    "shutdown",
)

class SidecarServer:
    """One request, one response; no concurrency and no notifications."""

    def __init__(
        self,
        control_plane: SkillControlPlane | None,
        *,
        startup_error: str | None = None,
        diagnostics: TextIO | None = None,
    ) -> None:
        if control_plane is None and startup_error is None:
            raise ValueError("unready sidecar requires a startup_error")
        if control_plane is not None and startup_error is not None:
            raise ValueError("provide control_plane or startup_error, not both")
        self._control_plane = control_plane
        self._startup_error = startup_error
        self._diagnostics = diagnostics if diagnostics is not None else sys.stderr
        self._readiness_cache: ControlPlaneReadiness | None = None
        self._handlers: dict[
            str,
            Callable[[Request], Any] | Callable[[Request], None],
        ] = {
            "handshake": self._handshake,
            "begin_turn": self._begin_turn,
            "end_turn": self._end_turn,
            "context_snapshot": self._context_snapshot,
            "search_capability": self._search_capability,
            "apply_capability": self._apply_capability,
            "load_skill_body": self._load_skill_body,
            "mark_skill_body_evicted": self._mark_skill_body_evicted,
            "mark_all_skill_bodies_evicted": self._mark_all_skill_bodies_evicted,
            "turn_audit": self._turn_audit,
            "export_state": self._export_state,
            "restore_state": self._restore_state,
            "shutdown": self._shutdown,
        }

    def handle_line(self, line: str) -> dict:
        """Return one response envelope."""

        try:
            request = parse_request(line)
        except ProtocolError as exc:
            return exc.response()

        if request.method not in self._handlers:
            return ProtocolError(
                "UNKNOWN_METHOD", f"unknown method: {request.method}",
                response_id=request.id,
            ).response()
        try:
            result = self._handlers[request.method](request)
            return {
                "id": request.id,
                "ok": True,
                "result": encode_json(result) if result is not None else {},
            }
        except ProtocolError as exc:
            return ProtocolError(
                exc.code, exc.message, exc.details, response_id=request.id,
            ).response()
        except _ShutdownRequested:
            return {
                "id": request.id,
                "ok": True,
                "result": {"shutdown": True},
            }
        except ValueError as exc:
            self._log_value_error(exc)
            return ProtocolError(
                CORE_VALIDATION_ERROR, str(exc), response_id=request.id,
            ).response()
        except KeyError as exc:
            key = exc.args[0] if exc.args else ""
            self._log_value_error(KeyError(key))
            return ProtocolError(
                CORE_VALIDATION_ERROR, f"unknown Core key: {key}",
                response_id=request.id,
            ).response()
        except Exception as exc:
            self._log_unexpected(exc)
            return ProtocolError(
                INTERNAL_ERROR, "internal sidecar error", response_id=request.id,
            ).response()

    def serve(
        self,
        lines: Iterable[str],
        writer: Callable[[str], None],
    ) -> int:
        for line in lines:
            response = self.handle_line(line)
            writer(self._dump(response))
            if response.get("result") == {"shutdown": True}:
                return 0
        return 0

    def serve_stdio(self, stdin: TextIO, stdout: TextIO) -> int:
        return self.serve(stdin, lambda line: (
            stdout.write(line),
            stdout.flush(),
        ))

    def serve_binary(self, stdin, stdout) -> int:
        for raw_line in stdin:
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                response = ProtocolError(
                    INVALID_REQUEST, "request line must be UTF-8"
                ).response()
            else:
                response = self.handle_line(line)
            stdout.write(self._dump(response).encode("utf-8"))
            stdout.flush()
            if response.get("result") == {"shutdown": True}:
                return 0
        return 0

    def _dump(self, value: dict) -> str:
        return json.dumps(
            value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
        ) + "\n"

    def _runtime(self, request: Request) -> SkillControlPlane:
        if self._control_plane is None:
            raise ProtocolError(NOT_READY, "Control Plane is not ready")
        if self._readiness_cache is None:
            self._readiness_cache = self._readiness()
        if not self._readiness_cache.ready:
            raise ProtocolError(
                NOT_READY, "Control Plane is not ready",
                {"errors": list(self._readiness_cache.errors)},
            )
        return self._control_plane

    def _handshake(self, request: Request) -> dict:
        self._require_empty(request)
        return {
            "protocol_version": PROTOCOL_VERSION,
            "readiness": self._readiness(),
            "supported_methods": SUPPORTED_METHODS,
        }

    def _readiness(self) -> ControlPlaneReadiness:
        if self._control_plane is None:
            return ControlPlaneReadiness(
                ready=False,
                skill_count=0,
                retrieval_cards_ready=False,
                dense_configured=False,
                dense_ready=False,
                fusion_backend="unknown",
                fusion_ready=False,
                probe_executed=False,
                probe_ready=False,
                checks=(),
                errors=(self._startup_error or "production initialization failed",),
            )
        readiness = self._control_plane.readiness()
        self._readiness_cache = readiness
        return readiness

    def _begin_turn(self, _request: Request) -> dict:
        self._runtime(_request).begin_turn()
        return {"turn_started": True}

    def _end_turn(self, _request: Request) -> dict:
        self._runtime(_request).end_turn()
        return {"turn_ended": True}

    def _context_snapshot(self, request: Request) -> object:
        if set(request.params) - {"compact"}:
            raise ProtocolError(INVALID_PARAMS, "context_snapshot has unknown fields")
        compact = request.params.get("compact", True)
        if not isinstance(compact, bool):
            raise ProtocolError(INVALID_PARAMS, "compact must be a boolean")
        return self._runtime(request).context_snapshot(compact=compact)

    def _search_capability(self, request: Request) -> object:
        self._runtime(request)
        if set(request.params) - {"need", "k"}:
            raise ProtocolError(INVALID_PARAMS, "search_capability has unknown fields")
        if "need" not in request.params:
            raise ProtocolError(INVALID_PARAMS, "need is required")
        need = request.params["need"]
        if not isinstance(need, str) or not need:
            raise ProtocolError(INVALID_PARAMS, "need must be a non-empty string")
        k = request.params.get("k", 10)
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ProtocolError(INVALID_PARAMS, "k must be a positive integer")
        return self._runtime(request).search_capability(need, k=k)

    def _apply_capability(self, request: Request) -> object:
        control_plane = self._runtime(request)
        decision = decode_capability_decision(request.params)
        return control_plane.apply_capability(decision)

    def _load_skill_body(self, request: Request) -> object:
        return self._runtime(request).load_skill_body(
            _required_string(request, "skill_id")
        )

    def _mark_skill_body_evicted(self, request: Request) -> dict:
        self._runtime(request).mark_skill_body_evicted(
            _required_string(request, "skill_id")
        )
        return {"skill_id": request.params["skill_id"], "body_state": "evicted"}

    def _mark_all_skill_bodies_evicted(self, request: Request) -> dict:
        self._require_empty(request)
        self._runtime(request).mark_all_skill_bodies_evicted()
        return {"body_state": "evicted"}

    def _turn_audit(self, request: Request) -> object:
        self._require_empty(request)
        return self._runtime(request).turn_audit()

    def _export_state(self, request: Request) -> object:
        self._require_empty(request)
        return self._runtime(request).export_state()

    def _restore_state(self, request: Request) -> dict:
        control_plane = self._runtime(request)
        snapshot = decode_state_snapshot(request.params)
        control_plane.restore_state(snapshot)
        return {"restored": True, "version": snapshot.version}

    def _shutdown(self, request: Request) -> None:
        self._require_empty(request)
        raise _ShutdownRequested()

    @staticmethod
    def _require_empty(request: Request) -> None:
        if request.params:
            raise ProtocolError(INVALID_PARAMS, "method params must be empty")

    def _log_value_error(self, exc: ValueError) -> None:
        print(f"sidecar: core validation failed: {exc}", file=self._diagnostics)

    def _log_unexpected(self, exc: Exception) -> None:
        print("sidecar: unexpected error", file=self._diagnostics)
        traceback.print_exception(
            type(exc), exc, exc.__traceback__, file=self._diagnostics
        )


class _ShutdownRequested(Exception):
    pass


def _required_string(request: Request, name: str) -> str:
    if set(request.params) != {name}:
        raise ProtocolError(INVALID_PARAMS, f"method params must contain only {name}")
    value = request.params[name]
    if not isinstance(value, str) or not value:
        raise ProtocolError(INVALID_PARAMS, f"{name} must be a non-empty string")
    return value


def build_sidecar_server(
    *,
    skill_root: str | Path,
    retrieval_cards: str | Path,
    max_searches_per_turn: int = 3,
    embedding_client_factory: Callable[
        [], BigModelEmbeddingClient] = BigModelEmbeddingClient,
) -> SidecarServer:
    try:
        embedding = embedding_client_factory()
        control_plane = SkillControlPlane.from_tree(
            skill_root,
            retrieval_cards=retrieval_cards,
            max_searches_per_turn=max_searches_per_turn,
            dense_factory=lambda records: BigModelDenseRetriever(
                records,
                model_name=embedding.model,
                dimensions=embedding.dimensions,
                embed_batch=embedding,
            ),
        )
    except Exception as exc:
        return SidecarServer(None, startup_error=str(exc))
    return SidecarServer(control_plane)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m skill_control_plane.sidecar",
        description="Run the Skill Control Plane NDJSON stdio sidecar.",
    )
    parser.add_argument("--skill-root", required=True, help="Local Skill tree")
    parser.add_argument(
        "--retrieval-cards", required=True, help="RetrievalCard v0.1 JSONL"
    )
    parser.add_argument(
        "--max-searches-per-turn", type=int, default=3,
        help="Positive per-turn search budget",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.max_searches_per_turn < 1:
        raise SystemExit("--max-searches-per-turn must be positive")
    server = build_sidecar_server(
        skill_root=args.skill_root,
        retrieval_cards=args.retrieval_cards,
        max_searches_per_turn=args.max_searches_per_turn,
    )
    return server.serve_binary(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
