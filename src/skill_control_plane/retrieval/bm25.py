from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

from skill_control_plane.models import RetrievalCandidate, SkillRecord

_TOKEN_RE = re.compile(r"[A-Za-z0-9_+.-]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


class BM25Retriever:
    """Dependency-free BM25 baseline over complete searchable Skill text."""

    def __init__(self, skills: Iterable[SkillRecord], *, k1: float = 1.5, b: float = 0.75) -> None:
        self.skills = tuple(skills)
        self.k1 = k1
        self.b = b
        self._docs = [tokenize(skill.search_text) for skill in self.skills]
        self._freqs = [Counter(doc) for doc in self._docs]
        self._lengths = [len(doc) for doc in self._docs]
        self._avgdl = sum(self._lengths) / len(self._lengths) if self._lengths else 0.0

        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(set(doc))
        n = len(self._docs)
        self._idf = {
            term: math.log(1.0 + (n - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }

    def _score(self, query_terms: list[str], index: int) -> float:
        freq = self._freqs[index]
        dl = self._lengths[index]
        score = 0.0
        for term in query_terms:
            tf = freq.get(term, 0)
            if tf == 0:
                continue
            idf = self._idf.get(term, 0.0)
            norm = tf + self.k1 * (1.0 - self.b + self.b * dl / (self._avgdl or 1.0))
            score += idf * (tf * (self.k1 + 1.0)) / norm
        return score

    def search(self, query: str, k: int = 5) -> list[RetrievalCandidate]:
        if k <= 0 or not self.skills:
            return []
        terms = tokenize(query)
        unique_terms = tuple(dict.fromkeys(terms))
        scored: list[tuple[float, SkillRecord]] = []
        for i, skill in enumerate(self.skills):
            score = self._score(terms, i)
            if score > 0:
                scored.append((score, skill))

        scored.sort(key=lambda item: (-item[0], item[1].skill_id))
        results: list[RetrievalCandidate] = []
        for rank, (score, skill) in enumerate(scored[:k], start=1):
            skill_terms = set(tokenize(skill.search_text))
            matched = tuple(term for term in unique_terms if term in skill_terms)
            evidence = (f"matched_terms: {', '.join(matched[:12])}",) if matched else ()
            results.append(
                RetrievalCandidate(
                    skill_id=skill.skill_id,
                    score=score,
                    rank=rank,
                    source_scores={"bm25": score},
                    evidence=evidence,
                )
            )
        return results
