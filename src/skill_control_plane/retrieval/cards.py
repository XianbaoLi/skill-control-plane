from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Protocol

from skill_control_plane.models import SkillRecord
TextCompleter = Callable[[str], str]

RETRIEVAL_CARD_VERSION = "retrieval-card-v0.1"
RETRIEVAL_CARD_FIELDS = ("purpose", "use_when", "capabilities", "lexical_cues")


@dataclass(frozen=True, slots=True)
class RetrievalCard:
    skill_id: str
    source_content_hash: str
    purpose: str
    use_when: tuple[str, ...]
    capabilities: tuple[str, ...]
    lexical_cues: tuple[str, ...]
    version: str = RETRIEVAL_CARD_VERSION

    def augmentation_text(
        self,
        include_fields: Iterable[str] | None = None,
    ) -> str:
        """Render selected LLM-derived fields for retrieval ablations.

        include_fields=None preserves the complete RetrievalCard v0.1 text.
        Passing a subset enables leave-one-field-out experiments without changing
        the legacy metadata portion of the Skill representation.
        """
        selected = set(
            RETRIEVAL_CARD_FIELDS if include_fields is None else include_fields
        )
        unknown = selected - set(RETRIEVAL_CARD_FIELDS)
        if unknown:
            raise ValueError(f"unknown retrieval-card fields: {sorted(unknown)}")

        sections: list[str] = []
        if "purpose" in selected and self.purpose:
            sections.append(f"purpose: {self.purpose}")
        if "use_when" in selected and self.use_when:
            sections.append("use when: " + "; ".join(self.use_when))
        if "capabilities" in selected and self.capabilities:
            sections.append("capabilities: " + "; ".join(self.capabilities))
        if "lexical_cues" in selected and self.lexical_cues:
            sections.append("lexical cues: " + "; ".join(self.lexical_cues))
        return "\n".join(sections)

    def search_text(self, skill: SkillRecord) -> str:
        """Exact retrieval text: legacy metadata once, then card augmentation."""
        sections = [
            skill.name,
            skill.description,
            " ".join(skill.tags),
            self.augmentation_text(),
        ]
        return "\n".join(section for section in sections if section.strip())


class RetrievalCardExtractor(Protocol):
    def extract(self, skill: SkillRecord) -> RetrievalCard: ...


def _clean_list(
    value: object,
    field: str,
    *,
    min_items: int = 0,
    max_items: int | None = None,
) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be a JSON array")
    cleaned: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field} must contain only strings")
        text = " ".join(item.split()).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    if len(cleaned) < min_items:
        raise ValueError(f"{field} must contain at least {min_items} items")
    if max_items is not None and len(cleaned) > max_items:
        cleaned = cleaned[:max_items]
    return tuple(cleaned)


def build_retrieval_card_prompt(skill: SkillRecord) -> str:
    """Build a leakage-resistant offline extraction prompt from one Skill only."""
    payload = {
        "skill_id": skill.skill_id,
        "name": skill.name,
        "description": skill.description,
        "tags": list(skill.tags),
        "category": skill.category,
        "body": skill.body,
    }
    return """Create a compact retrieval card for ONE Agent Skill.

Goal:
Represent when this Skill should be retrieved from a natural-language capability
request. This is an offline indexing task, not task solving.

Rules:
- Use ONLY the Skill content provided below.
- Do not use benchmark cases, Gold labels, retrieval queries, other Skills, prior
  conversation context, tools, files, or web search.
- Do not invent capabilities the Skill does not support.
- Prefer trigger language and user-observable situations over implementation detail.
- purpose: one short sentence describing the Skill's core job.
- use_when: 2-8 concise situations/signals that should trigger retrieval.
- capabilities: 1-8 concrete actions the Skill enables or guides.
- lexical_cues: 3-15 short words/phrases users may naturally use for these situations.
  Synonyms are allowed only when directly grounded in the Skill's described behavior.
- Do not include the Skill ID merely to make lexical matching easy.
- Return ONLY one JSON object with exactly these keys:
  purpose, use_when, capabilities, lexical_cues.

Skill:
""" + json.dumps(payload, ensure_ascii=False)


@dataclass(slots=True)
class LLMRetrievalCardExtractor:
    complete: TextCompleter

    def extract(self, skill: SkillRecord) -> RetrievalCard:
        raw = self.complete(build_retrieval_card_prompt(skill))
        data = json.loads(raw)
        if not isinstance(data, dict):
            raise ValueError("retrieval card completion must be a JSON object")
        if set(data) != {"purpose", "use_when", "capabilities", "lexical_cues"}:
            raise ValueError("retrieval card completion has unexpected keys")
        purpose = data["purpose"]
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError("purpose must be a non-empty string")
        purpose = " ".join(purpose.split()).strip()
        return RetrievalCard(
            skill_id=skill.skill_id,
            source_content_hash=skill.content_hash,
            purpose=purpose,
            use_when=_clean_list(
                data["use_when"], "use_when", min_items=2, max_items=8
            ),
            capabilities=_clean_list(
                data["capabilities"], "capabilities", min_items=1, max_items=8
            ),
            lexical_cues=_clean_list(
                data["lexical_cues"], "lexical_cues", min_items=3, max_items=15
            ),
        )


def load_retrieval_cards(path: str | Path) -> dict[str, RetrievalCard]:
    cards: dict[str, RetrievalCard] = {}
    for line_number, raw in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw.strip():
            continue
        data = json.loads(raw)
        purpose = data.get("purpose")
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError(f"retrieval card line {line_number} has invalid purpose")
        card = RetrievalCard(
            skill_id=str(data["skill_id"]),
            source_content_hash=str(data["source_content_hash"]),
            purpose=" ".join(purpose.split()).strip(),
            use_when=_clean_list(
                data.get("use_when"), "use_when", min_items=2, max_items=8
            ),
            capabilities=_clean_list(
                data.get("capabilities"), "capabilities", min_items=1, max_items=8
            ),
            lexical_cues=_clean_list(
                data.get("lexical_cues"), "lexical_cues", min_items=3, max_items=15
            ),
            version=str(data.get("version", "")),
        )
        if card.version != RETRIEVAL_CARD_VERSION:
            raise ValueError(
                f"retrieval card line {line_number} has version={card.version!r}; "
                f"expected {RETRIEVAL_CARD_VERSION!r}"
            )
        if card.skill_id in cards:
            raise ValueError(f"duplicate retrieval card for {card.skill_id}")
        cards[card.skill_id] = card
    return cards


def apply_retrieval_cards(
    skills: Iterable[SkillRecord],
    cards: dict[str, RetrievalCard],
    *,
    include_fields: Iterable[str] | None = None,
) -> list[SkillRecord]:
    """Project validated cards into SkillRecord metadata for existing retrievers.

    include_fields is intentionally explicit so evaluation code can remove one
    card field while holding the underlying Skill metadata fixed.
    """
    skill_list = list(skills)
    selected_fields = None if include_fields is None else tuple(include_fields)
    validate_retrieval_cards(skill_list, cards)
    return [
        replace(
            skill,
            description="\n".join(
                part
                for part in (
                    skill.description,
                    cards[skill.skill_id].augmentation_text(selected_fields),
                )
                if part
            ),
            body="",
        )
        for skill in skill_list
    ]


def validate_retrieval_cards(
    skills: Iterable[SkillRecord],
    cards: dict[str, RetrievalCard],
) -> None:
    skill_map = {skill.skill_id: skill for skill in skills}
    missing = sorted(set(skill_map) - set(cards))
    extra = sorted(set(cards) - set(skill_map))
    stale = sorted(
        skill_id
        for skill_id, skill in skill_map.items()
        if skill_id in cards
        and cards[skill_id].source_content_hash != skill.content_hash
    )
    if missing or extra or stale:
        raise ValueError(
            "retrieval-card corpus mismatch: "
            f"missing={missing[:8]} extra={extra[:8]} stale={stale[:8]}"
        )


def _write_cards(path: Path, cards: dict[str, RetrievalCard]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    payload = "".join(
        json.dumps(asdict(cards[skill_id]), ensure_ascii=False, sort_keys=True) + "\n"
        for skill_id in sorted(cards)
    )
    temp.write_text(payload, encoding="utf-8")
    temp.replace(path)


def build_retrieval_card_cache(
    skills: Iterable[SkillRecord],
    *,
    extractor: RetrievalCardExtractor,
    output: str | Path,
    force: bool = False,
) -> dict[str, int]:
    """Extract cards incrementally and checkpoint after every successful LLM call."""
    skill_list = tuple(skills)
    output_path = Path(output)
    existing = (
        load_retrieval_cards(output_path)
        if output_path.exists() and not force
        else {}
    )
    skill_ids = {skill.skill_id for skill in skill_list}
    cards: dict[str, RetrievalCard] = {
        skill_id: card
        for skill_id, card in existing.items()
        if skill_id in skill_ids
    }
    extracted = 0
    reused = 0

    for skill in skill_list:
        cached = existing.get(skill.skill_id)
        if (
            cached is not None
            and cached.version == RETRIEVAL_CARD_VERSION
            and cached.source_content_hash == skill.content_hash
        ):
            cards[skill.skill_id] = cached
            reused += 1
            continue

        card = extractor.extract(skill)
        if card.skill_id != skill.skill_id:
            raise ValueError("retrieval card skill_id mismatch")
        cards[skill.skill_id] = card
        extracted += 1
        _write_cards(output_path, cards)

    _write_cards(output_path, cards)
    validate_retrieval_cards(skill_list, cards)
    return {"skill_count": len(skill_list), "extracted": extracted, "reused": reused}
