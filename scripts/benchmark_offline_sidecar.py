#!/usr/bin/env python3
"""Smoke-only sidecar launcher with a local zero query embedding.

This validates Pi/adapter/sidecar plumbing without sending benchmark prompts to
an external provider. It is intentionally unsuitable for benchmark results.
"""
from __future__ import annotations

import argparse
import sys

from skill_control_plane.sidecar.server import build_sidecar_server


class ZeroEmbedding:
    model = "GLM-Embedding-3"
    dimensions = 2048

    def __call__(self, texts):
        return [[0.0] * self.dimensions for _ in texts]


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
    server = build_sidecar_server(
        skill_root=args.skill_root, retrieval_cards=args.retrieval_cards,
        dense_index=args.dense_index, max_searches_per_turn=args.max_searches_per_turn,
        embedding_client_factory=ZeroEmbedding,
    )
    return server.serve_binary(sys.stdin.buffer, sys.stdout.buffer)


if __name__ == "__main__":
    raise SystemExit(main())
