"""
sentiment.py
-------------
Sentiment analysis using NLTK's VADER (Valence Aware Dictionary and
sEntiment Reasoner), tuned with domain-specific correction logic.

VADER is a lexicon/rule-based model well suited to short conversational
text (much better than a naive keyword count), but out of the box it
can mislabel plain informational requests such as "I want to know the
price" as positive purely because of the word "want". We correct for
this with an informational-intent guard: if the message is a neutral
information-seeking request (a question, or contains phrases like
"I want to know", "tell me", "what is", "how much", etc.) and VADER's
compound score is only mildly positive, we classify it as Neutral
rather than Positive.
"""

import re
import nltk

try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    try:
        nltk.download("vader_lexicon", quiet=True)
    except Exception:
        pass

from nltk.sentiment.vader import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()

# Phrases that indicate a neutral, informational request rather than a
# genuine emotional expression, even though they may contain words like
# "want", "like", or "need" that VADER weights as mildly positive.
INFORMATIONAL_PATTERNS = [
    r"\bi want to know\b",
    r"\bi need to know\b",
    r"\bi'd like to know\b",
    r"\bcan you tell me\b",
    r"\btell me\b",
    r"\bwhat is\b",
    r"\bwhat's\b",
    r"\bwhat are\b",
    r"\bhow much\b",
    r"\bhow many\b",
    r"\bwhere is\b",
    r"\bwhen is\b",
    r"\bwho is\b",
    r"\bwhich one\b",
    r"\bi want (?:a|an|to buy|to purchase|to see|to check)\b",
    r"\bi need (?:a|an)\b",
    r"\bshow me\b",
    r"\bcan you (?:show|find|help)\b",
]

STRONG_NEGATIVE_WORDS = [
    "terrible", "awful", "horrible", "worst", "disappointed", "disappointing",
    "broken", "defective", "damaged", "useless", "waste", "unacceptable",
    "poor", "bad", "hate", "angry", "frustrated", "complaint", "refund",
    "return", "cancel", "problem", "issue", "not working", "stopped working",
]

STRONG_POSITIVE_WORDS = [
    "amazing", "excellent", "awesome", "fantastic", "love", "great",
    "wonderful", "happy", "satisfied", "impressed", "perfect", "best",
    "brilliant", "superb", "delighted", "recommend",
]

NEUTRAL_UPPER_BOUND = 0.35   # compound scores below this (but above -0.05) => lean neutral
NEUTRAL_LOWER_BOUND = -0.35


def _is_informational(text_lower: str) -> bool:
    return any(re.search(pattern, text_lower) for pattern in INFORMATIONAL_PATTERNS)


def analyze_sentiment(text: str):
    """
    Returns a dict:
    {
        "label": "Positive" | "Negative" | "Neutral",
        "score": 0.62,        # VADER compound score, -1 to 1
        "confidence": 0.78
    }
    """
    if not text or not text.strip():
        return {"label": "Neutral", "score": 0.0, "confidence": 1.0}

    text_lower = text.lower()
    scores = _analyzer.polarity_scores(text)
    compound = scores["compound"]

    has_strong_negative = any(w in text_lower for w in STRONG_NEGATIVE_WORDS)
    has_strong_positive = any(w in text_lower for w in STRONG_POSITIVE_WORDS)
    informational = _is_informational(text_lower)

    # Rule 1: explicit strong sentiment words always win, regardless of
    # VADER's raw compound score direction (guards against short messages
    # where VADER's magnitude is muted).
    if has_strong_negative and not has_strong_positive:
        label = "Negative"
    elif has_strong_positive and not has_strong_negative:
        label = "Positive"
    # Rule 2: informational / request-style messages with only mild
    # polarity are treated as Neutral even if they contain words like
    # "want" or "like" (e.g. "I want to know the price").
    elif informational and NEUTRAL_LOWER_BOUND <= compound <= NEUTRAL_UPPER_BOUND:
        label = "Neutral"
    # Rule 3: fall back to VADER's compound score with reasonable
    # thresholds (tighter than the VADER defaults of +-0.05 to reduce
    # false positives on neutral statements).
    elif compound >= 0.4:
        label = "Positive"
    elif compound <= -0.4:
        label = "Negative"
    else:
        label = "Neutral"

    confidence = min(1.0, round(abs(compound) + (0.3 if (has_strong_negative or has_strong_positive) else 0.0), 2))
    if label == "Neutral" and confidence < 0.5:
        confidence = 0.6  # default neutral confidence baseline

    return {
        "label": label,
        "score": round(compound, 3),
        "confidence": confidence,
    }
