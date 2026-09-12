from __future__ import annotations

from typing import Protocol

from skill_control_plane.models import RetrievalCandidate


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[RetrievalCandidate]:
        """Return candidates ordered best-first."""
        ...
