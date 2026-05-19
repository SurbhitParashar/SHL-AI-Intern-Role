from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable


CATALOG_PATH = Path(__file__).resolve().parent.parent / "data" / "shl_catalog.json"
TOKEN_RE = re.compile(r"[a-z0-9+#.]+")

TYPE_ALIASES = {
    "k": "Knowledge & Skills",
    "p": "Personality & Behavior",
    "a": "Ability & Aptitude",
    "b": "Biodata & Situational Judgment",
    "s": "Simulation",
}

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "assessment",
    "assessments",
    "for",
    "hire",
    "hiring",
    "i",
    "in",
    "need",
    "of",
    "on",
    "or",
    "role",
    "test",
    "tests",
    "the",
    "to",
    "we",
    "with",
}


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(text.lower()) if token not in STOP_WORDS]


@dataclass(frozen=True)
class CatalogItem:
    name: str
    url: str
    test_type: str
    description: str = ""
    skills: tuple[str, ...] = ()
    job_family: str = ""
    duration_minutes: int | None = None
    remote_testing: bool | None = None
    adaptive: bool | None = None
    raw: dict = field(default_factory=dict)

    @property
    def searchable_text(self) -> str:
        pieces = [
            self.name,
            self.test_type,
            TYPE_ALIASES.get(self.test_type.upper(), ""),
            self.description,
            self.job_family,
            " ".join(self.skills),
        ]
        return " ".join(piece for piece in pieces if piece)

    def to_recommendation(self) -> dict[str, str]:
        return {"name": self.name, "url": self.url, "test_type": self.test_type}


class Catalog:
    def __init__(self, items: Iterable[CatalogItem]):
        self.items = list(items)
        self.by_url = {item.url: item for item in self.items}
        self.by_name = {normalize_name(item.name): item for item in self.items}
        self._documents = [tokenize(item.searchable_text) for item in self.items]
        self._doc_freq = Counter(token for doc in self._documents for token in set(doc))
        self._avg_doc_len = (
            sum(len(doc) for doc in self._documents) / len(self._documents)
            if self._documents
            else 1.0
        )

    def find_by_name(self, phrase: str) -> CatalogItem | None:
        normalized = normalize_name(phrase)
        if normalized in self.by_name:
            return self.by_name[normalized]

        phrase_tokens = set(tokenize(phrase))
        if not phrase_tokens:
            return None

        best: tuple[float, CatalogItem] | None = None
        for item in self.items:
            item_tokens = set(tokenize(item.name))
            overlap = len(phrase_tokens & item_tokens)
            if overlap == 0:
                continue
            score = overlap / max(len(item_tokens), len(phrase_tokens))
            if normalized and normalized in normalize_name(item.name):
                score += 0.5
            if best is None or score > best[0]:
                best = (score, item)
        return best[1] if best and best[0] >= 0.35 else None

    def search(
        self,
        query: str,
        limit: int = 10,
        include_types: set[str] | None = None,
        exclude_types: set[str] | None = None,
    ) -> list[CatalogItem]:
        query_tokens = tokenize(expand_query(query))
        if not query_tokens:
            return []

        scored: list[tuple[float, CatalogItem]] = []
        for index, item in enumerate(self.items):
            item_type = item.test_type.upper()
            if include_types and item_type not in include_types:
                continue
            if exclude_types and item_type in exclude_types:
                continue

            score = self._bm25(query_tokens, self._documents[index])
            score += self._boosts(query, query_tokens, item)
            if score > 0:
                scored.append((score, item))

        scored.sort(key=lambda pair: (-pair[0], pair[1].name.lower()))
        return diversify([item for _, item in scored], limit)

    def _bm25(self, query_tokens: list[str], document: list[str]) -> float:
        if not document:
            return 0.0
        frequencies = Counter(document)
        score = 0.0
        total_docs = max(len(self._documents), 1)
        k1 = 1.5
        b = 0.75
        for token in query_tokens:
            if token not in frequencies:
                continue
            doc_freq = self._doc_freq[token]
            idf = math.log(1 + (total_docs - doc_freq + 0.5) / (doc_freq + 0.5))
            numerator = frequencies[token] * (k1 + 1)
            denominator = frequencies[token] + k1 * (1 - b + b * len(document) / self._avg_doc_len)
            score += idf * numerator / denominator
        return score

    def _boosts(self, raw_query: str, query_tokens: list[str], item: CatalogItem) -> float:
        text = raw_query.lower()
        item_text = item.searchable_text.lower()
        score = 0.0

        for token in query_tokens:
            if token in item.name.lower():
                score += 1.2
            if token in item.job_family.lower():
                score += 0.8
            if any(token in skill.lower() for skill in item.skills):
                score += 0.9

        type_lower = TYPE_ALIASES.get(item.test_type.upper(), "").lower()
        if any(word in text for word in ["personality", "behaviour", "behavior", "opq"]):
            score += 4.0 if item.test_type.upper() == "P" else -0.5
        if any(word in text for word in ["cognitive", "ability", "aptitude", "reasoning", "gsa"]):
            score += 3.0 if item.test_type.upper() == "A" else 0
        if any(word in text for word in ["coding", "developer", "programmer", "java", "python", "sql"]):
            score += 3.0 if item.test_type.upper() == "K" else 0
        if "stakeholder" in text or "communication" in text:
            if item.test_type.upper() in {"P", "B", "S"}:
                score += 2.0
        if "graduate" in text or "entry" in text:
            if any(term in item_text for term in ["graduate", "entry", "verify", "gsa"]):
                score += 1.8
        if "manager" in text or "leadership" in text:
            if any(term in item_text for term in ["manager", "leadership", "management"]):
                score += 2.2
        if type_lower and any(token in type_lower for token in query_tokens):
            score += 1.0
        return score


def diversify(items: list[CatalogItem], limit: int) -> list[CatalogItem]:
    selected: list[CatalogItem] = []
    type_counts: Counter[str] = Counter()
    for item in items:
        if len(selected) >= limit:
            break
        type_count = type_counts[item.test_type.upper()]
        if type_count >= 4 and len({candidate.test_type for candidate in items}) > 1:
            continue
        selected.append(item)
        type_counts[item.test_type.upper()] += 1

    if len(selected) < limit:
        seen = {item.url for item in selected}
        for item in items:
            if item.url not in seen:
                selected.append(item)
                seen.add(item.url)
            if len(selected) >= limit:
                break
    return selected


def normalize_name(name: str) -> str:
    return " ".join(tokenize(name)).lower()


def expand_query(query: str) -> str:
    synonyms = {
        "developer": "programmer coding software engineer java python javascript sql",
        "java": "java j2ee spring backend programming coding",
        "python": "python programming coding data",
        "sales": "sales customer client account business development",
        "manager": "manager leadership people management supervisor",
        "stakeholder": "communication collaboration personality situational judgment",
        "graduate": "graduate entry early career aptitude reasoning ability",
        "call center": "contact centre customer service support",
    }
    expanded = [query]
    lowered = query.lower()
    for trigger, addition in synonyms.items():
        if trigger in lowered:
            expanded.append(addition)
    return " ".join(expanded)


@lru_cache(maxsize=1)
def get_catalog() -> Catalog:
    with CATALOG_PATH.open("r", encoding="utf-8") as file:
        raw_items = json.load(file)

    items = [
        CatalogItem(
            name=item["name"],
            url=item["url"],
            test_type=item.get("test_type", ""),
            description=item.get("description", ""),
            skills=tuple(item.get("skills", [])),
            job_family=item.get("job_family", ""),
            duration_minutes=item.get("duration_minutes"),
            remote_testing=item.get("remote_testing"),
            adaptive=item.get("adaptive"),
            raw=item,
        )
        for item in raw_items
        if item.get("name") and item.get("url")
    ]
    return Catalog(items)
