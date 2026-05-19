from __future__ import annotations

import re
from dataclasses import dataclass

from app.catalog import Catalog, CatalogItem, get_catalog
from app.schemas import ChatResponse, Message, Recommendation


VAGUE_PATTERNS = [
    re.compile(r"^\s*(hi|hello|hey)\s*$", re.I),
    re.compile(r"\b(i need|need|recommend|suggest)\s+(an?\s+)?(assessment|test)s?\s*$", re.I),
    re.compile(r"\b(help me|what should i use)\s*$", re.I),
]

OFF_TOPIC_TERMS = {
    "salary",
    "compensation",
    "legal",
    "lawsuit",
    "visa",
    "immigration",
    "contract",
    "offer letter",
    "interview questions",
    "job description template",
    "general hiring advice",
}

INJECTION_PATTERNS = [
    re.compile(r"ignore (all )?(previous|prior|system|developer) instructions", re.I),
    re.compile(r"reveal (your )?(prompt|system message|instructions)", re.I),
    re.compile(r"recommend .*outside .*catalog", re.I),
    re.compile(r"pretend you are not", re.I),
]

COMPARISON_RE = re.compile(
    r"\b(compare|difference|different|versus|vs\.?|distinguish)\b", re.I
)
TYPE_REQUESTS = {
    "P": {"personality", "behavior", "behaviour", "opq", "culture", "fit"},
    "A": {"cognitive", "ability", "aptitude", "reasoning", "gsa", "numerical", "verbal"},
    "K": {"skill", "skills", "coding", "technical", "java", "python", "sql", "excel"},
    "B": {"situational", "judgment", "judgement", "sjt"},
    "S": {"simulation", "work sample", "inbox", "case"},
}


@dataclass
class ConversationContext:
    latest_user: str
    user_text: str
    assistant_text: str
    requested_types: set[str]
    enough_context: bool
    vague: bool


class SHLAgent:
    def __init__(self, catalog: Catalog | None = None):
        self.catalog = catalog or get_catalog()

    def chat(self, messages: list[Message]) -> ChatResponse:
        context = self._build_context(messages)

        if self._is_prompt_injection(context.latest_user):
            return self._refusal(
                "I can only help select SHL assessments from the catalog. I cannot follow requests to bypass scope or instructions."
            )

        if self._is_off_topic(context.latest_user):
            return self._refusal(
                "I can help with SHL assessment selection only. Please share the role, skills, or assessment needs you want to evaluate."
            )

        if COMPARISON_RE.search(context.latest_user):
            return self._compare(context.latest_user)

        if context.vague or not context.enough_context:
            return ChatResponse(
                reply=(
                    "I can help shortlist SHL assessments. What role or skill area are you hiring for, "
                    "and do you need technical skills, cognitive ability, personality, or a mix?"
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        recommendations = self._recommend(context)
        if not recommendations:
            return ChatResponse(
                reply=(
                    "I could not find a confident catalog match from the information provided. "
                    "Please share the role title, key skills, and whether you need ability, personality, or job-skill tests."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        type_note = self._type_note(context.requested_types)
        reply = (
            f"Got it. Here are {len(recommendations)} SHL assessment"
            f"{'' if len(recommendations) == 1 else 's'} that best fit the current hiring context{type_note}."
        )
        return ChatResponse(
            reply=reply,
            recommendations=[Recommendation(**item.to_recommendation()) for item in recommendations],
            end_of_conversation=False,
        )

    def _build_context(self, messages: list[Message]) -> ConversationContext:
        user_messages = [message.content for message in messages if message.role == "user"]
        assistant_messages = [message.content for message in messages if message.role == "assistant"]
        latest_user = user_messages[-1] if user_messages else ""
        user_text = "\n".join(user_messages)
        requested_types = detect_requested_types(user_text)
        enough_context = has_enough_context(user_text)
        vague = any(pattern.search(latest_user) for pattern in VAGUE_PATTERNS)
        return ConversationContext(
            latest_user=latest_user,
            user_text=user_text,
            assistant_text="\n".join(assistant_messages),
            requested_types=requested_types,
            enough_context=enough_context,
            vague=vague,
        )

    def _recommend(self, context: ConversationContext) -> list[CatalogItem]:
        include_types = context.requested_types or None

        # If a user asks to add personality/ability/etc., keep the whole role context
        # but require at least one item from the requested category where possible.
        primary = self.catalog.search(context.user_text, limit=10, include_types=include_types)
        if include_types and primary:
            supporting = self.catalog.search(context.user_text, limit=10, exclude_types=include_types)
            return merge_ranked(primary, supporting, 10)

        return self.catalog.search(context.user_text, limit=10)

    def _compare(self, text: str) -> ChatResponse:
        candidates = extract_comparison_candidates(text)
        found: list[CatalogItem] = []
        for candidate in candidates:
            item = self.catalog.find_by_name(candidate)
            if item and item.url not in {existing.url for existing in found}:
                found.append(item)

        if len(found) < 2:
            fallback = self.catalog.search(text, limit=4)
            for item in fallback:
                if item.url not in {existing.url for existing in found}:
                    found.append(item)
                if len(found) >= 2:
                    break

        if len(found) < 2:
            return ChatResponse(
                reply=(
                    "I can compare SHL assessments when I can identify at least two catalog items. "
                    "Please name the assessments exactly, for example OPQ32r and General Ability Screen."
                ),
                recommendations=[],
                end_of_conversation=False,
            )

        first, second = found[:2]
        reply = (
            f"{first.name} is a {describe_type(first)} assessment. {first.description} "
            f"{second.name} is a {describe_type(second)} assessment. {second.description} "
            "The practical difference is the assessment focus: "
            f"{first.name} is strongest for {focus(first)}, while {second.name} is strongest for {focus(second)}."
        )
        return ChatResponse(
            reply=reply,
            recommendations=[Recommendation(**item.to_recommendation()) for item in found[:2]],
            end_of_conversation=False,
        )

    def _is_off_topic(self, text: str) -> bool:
        lowered = text.lower()
        return any(term in lowered for term in OFF_TOPIC_TERMS) and "assessment" not in lowered

    def _is_prompt_injection(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in INJECTION_PATTERNS)

    def _refusal(self, reply: str) -> ChatResponse:
        return ChatResponse(reply=reply, recommendations=[], end_of_conversation=False)

    def _type_note(self, requested_types: set[str]) -> str:
        if not requested_types:
            return ""
        labels = {
            "K": "skills/knowledge",
            "P": "personality/behavior",
            "A": "ability/aptitude",
            "B": "situational judgment",
            "S": "simulation",
        }
        selected = [labels[type_code] for type_code in sorted(requested_types) if type_code in labels]
        return " with emphasis on " + ", ".join(selected) if selected else ""


def detect_requested_types(text: str) -> set[str]:
    lowered = text.lower()
    requested: set[str] = set()
    for type_code, triggers in TYPE_REQUESTS.items():
        if any(trigger in lowered for trigger in triggers):
            requested.add(type_code)
    return requested


def has_enough_context(text: str) -> bool:
    lowered = text.lower()
    role_or_skill_terms = [
        "developer",
        "engineer",
        "java",
        "python",
        "sql",
        "sales",
        "manager",
        "graduate",
        "customer",
        "finance",
        "accounting",
        "analyst",
        "marketing",
        "stakeholder",
        "leadership",
        "personality",
        "cognitive",
        "call center",
        "contact centre",
        "admin",
    ]
    if any(term in lowered for term in role_or_skill_terms):
        return True
    content_words = re.findall(r"[a-z0-9+#.]+", lowered)
    return len(content_words) >= 12


def extract_comparison_candidates(text: str) -> list[str]:
    cleaned = re.sub(r"\b(what is|what's|the|difference|between|compare|versus|vs\.?|and)\b", "|", text, flags=re.I)
    pieces = [piece.strip(" ?.,:;\"'()") for piece in cleaned.split("|")]
    candidates = [piece for piece in pieces if len(piece) >= 2]

    quoted = re.findall(r"['\"]([^'\"]+)['\"]", text)
    return quoted + candidates


def describe_type(item: CatalogItem) -> str:
    labels = {
        "K": "knowledge and skills",
        "P": "personality and behavior",
        "A": "ability and aptitude",
        "B": "situational judgment",
        "S": "simulation",
    }
    return labels.get(item.test_type.upper(), item.test_type)


def focus(item: CatalogItem) -> str:
    if item.skills:
        return ", ".join(item.skills[:3])
    if item.job_family:
        return item.job_family
    return describe_type(item)


def merge_ranked(primary: list[CatalogItem], supporting: list[CatalogItem], limit: int) -> list[CatalogItem]:
    merged: list[CatalogItem] = []
    for item in primary[: max(3, min(6, limit))]:
        if item.url not in {existing.url for existing in merged}:
            merged.append(item)
    for item in supporting:
        if len(merged) >= limit:
            break
        if item.url not in {existing.url for existing in merged}:
            merged.append(item)
    return merged[:limit]
