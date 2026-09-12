#!/usr/bin/env python3
"""Production sidecar entrypoint with benchmark-only embedding telemetry."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from skill_control_plane.discovery.bigmodel import BigModelEmbeddingClient
from skill_control_plane.sidecar.server import build_sidecar_server


class TelemetryEmbedding:
    def __init__(self, path: Path) -> None:
        self.client = BigModelEmbeddingClient()
        self.model = self.client.model
        self.dimensions = self.client.dimensions
        self.path = path
        self.phase = "startup"

    def __call__(self, texts):
        vectors = self.client(texts)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"phase": self.phase, "batch_size": len(texts)}) + "\n")
        return vectors


def main(argv=None):
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw[:2] == ["-m", "skill_control_plane.sidecar"]:
        raw = raw[2:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill-root", required=True)
    parser.add_argument("--retrieval-cards", required=True)
    parser.add_argument("--dense-index", required=True)
    parser.add_argument("--max-searches-per-turn", type=int, default=3)
    args = parser.parse_args(raw)
    telemetry_path = Path(os.environ["BENCHMARK_EMBEDDING_TELEMETRY"])
    embedding = TelemetryEmbedding(telemetry_path)
    server = build_sidecar_server(
        skill_root=args.skill_root,
        retrieval_cards=args.retrieval_cards,
        dense_index=args.dense_index,
        max_searches_per_turn=args.max_searches_per_turn,
        embedding_client_factory=lambda: embedding,
    )
    for raw_line in sys.stdin.buffer:
        line = raw_line.decode("utf-8")
        request = json.loads(line)
        response = server.handle_line(line)
        sys.stdout.buffer.write((json.dumps(response, ensure_ascii=False,
                                            separators=(",", ":")) + "\n").encode())
        sys.stdout.buffer.flush()
        if request.get("method") == "handshake" and response.get("ok"):
            embedding.phase = "runtime"
        if response.get("result") == {"shutdown": True}:
            return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
