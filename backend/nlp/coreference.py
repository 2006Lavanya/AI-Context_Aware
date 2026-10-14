"""
coreference.py
----------------
Lightweight, rule-based coreference / context resolution.

Full neural coreference resolution (e.g. neuralcoref, or a transformer
based resolver) is heavy and often overkill for a task-oriented,
slot-filling conversational assistant like this one. Instead, this
module implements a practical, explainable resolution strategy that
covers the referring expressions called out in the project spec:

    it / this / that / this product / the previous one / which one /
    its price / its RAM

The strategy:
1. Detect referring expressions in the user's message using regex.
2. Resolve them against the ContextManager's "last_discussed_object"
   and other stored slots (brand, product, budget, ram, etc.).
3. Rewrite the message (an internal "resolved_message") by substituting
   the referring expression with the concretely resolved object, and/or
   annotate which context field is being asked about
   (e.g. "its RAM" -> asking about ctx['ram'] of ctx['last_discussed_object']).

This resolved information is then handed to the response generator so
it can produce a grounded, context-aware reply instead of a generic
"I don't know what you mean".
"""

import re

REFERRING_EXPRESSIONS = [
    r"\bthe previous one\b",
    r"\bthe last one\b",
    r"\bwhich one\b",
    r"\bthis product\b",
    r"\bthis one\b",
    r"\bthat one\b",
    r"\bits\b",
    r"\bit\b",
    r"\bthis\b",
    r"\bthat\b",
]

# Which context "slot" a follow-up question is probably asking about,
# based on keywords present alongside the referring expression.
SLOT_KEYWORDS = {
    "ram": ["ram", "memory"],
    "price": ["price", "cost", "rate", "expensive"],
    "budget": ["budget"],
    "brand": ["brand", "make"],
    "color": ["color", "colour"],
    "size": ["size"],
    "battery": ["battery"],
    "storage": ["storage", "space", "rom"],
    "camera": ["camera", "megapixel"],
    "processor": ["processor", "cpu", "chip"],
    "specs": ["spec", "specification", "feature", "detail"],
}


def contains_referring_expression(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(pat, text_lower) for pat in REFERRING_EXPRESSIONS)


def detect_referenced_slot(text: str):
    """Detects which attribute of the referenced object the user is
    asking about, e.g. 'its RAM' -> 'ram', 'which one has better price' -> 'price'."""
    text_lower = text.lower()
    for slot, keywords in SLOT_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return slot
    return None


def resolve_context(text: str, context: dict):
    """
    Attempts to resolve pronouns/referring expressions in `text` using
    the conversation `context` (as produced by ContextManager).

    Returns a dict:
    {
        "has_reference": True/False,
        "resolved_object": "HP laptop" | None,
        "referenced_slot": "ram" | "price" | None,
        "resolved_message": "which one has better ram" -> annotated string
    }
    """
    has_reference = contains_referring_expression(text)
    last_object = context.get("last_discussed_object")
    referenced_slot = detect_referenced_slot(text) if has_reference else None

    result = {
        "has_reference": has_reference,
        "resolved_object": last_object if has_reference else None,
        "referenced_slot": referenced_slot,
        "resolved_message": text,
    }

    if has_reference and last_object:
        # Build a human-readable resolved message for transparency /
        # debugging / the NLP analysis panel, e.g.:
        # "which one has 16GB RAM" -> "[HP laptop] which one has 16GB RAM"
        result["resolved_message"] = f"[{last_object}] {text}"

    return result
