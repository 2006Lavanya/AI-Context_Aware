"""
preprocessing.py
-----------------
Handles the text preprocessing stage of the NLP pipeline:
Tokenization -> Stopword Removal -> Lemmatization

Uses NLTK for tokenization, stopwords, and lemmatization.
"""

import re
import string
import nltk

# Make sure the required NLTK resources are available. This is idempotent -
# if the resources are already downloaded, nothing happens.
_REQUIRED_NLTK_RESOURCES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]

for path, name in _REQUIRED_NLTK_RESOURCES:
    try:
        nltk.data.find(path)
    except (LookupError, OSError):
        try:
            nltk.download(name, quiet=True)
        except Exception:
            # Network might be unavailable at runtime; downstream code
            # falls back gracefully (see safe wrappers below).
            pass

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

_lemmatizer = WordNetLemmatizer()

try:
    _STOPWORDS = set(stopwords.words("english"))
except Exception:
    # Minimal fallback stopword list if NLTK corpus isn't available.
    _STOPWORDS = {
        "a", "an", "the", "is", "am", "are", "was", "were", "be", "been",
        "being", "do", "does", "did", "of", "in", "on", "at", "to", "for",
        "with", "as", "by", "this", "that", "it", "and", "or", "but", "if",
    }

# Words that should NEVER be stripped as stopwords because they carry
# important meaning for intent classification and coreference resolution
# (pronouns like "it", "this", "that" are critical for context resolution).
_KEEP_WORDS = {"it", "this", "that", "one", "which", "not", "no"}


def clean_text(text: str) -> str:
    """Lowercase and strip unwanted punctuation while preserving numbers,
    which are important for budgets/prices/quantities."""
    text = text.strip().lower()
    # Keep alphanumerics, spaces, and a few meaningful punctuation marks
    text = re.sub(r"[^\w\s₹%.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str):
    """Tokenize text into words using NLTK's word tokenizer."""
    try:
        return word_tokenize(text)
    except Exception:
        # Fallback simple split if punkt isn't available
        return text.split()


def remove_stopwords(tokens):
    """Remove common English stopwords, keeping context-critical words."""
    return [t for t in tokens if t in _KEEP_WORDS or t not in _STOPWORDS]


def lemmatize(tokens):
    """Lemmatize a list of tokens to their base dictionary form."""
    lemmatized = []
    for t in tokens:
        try:
            lemmatized.append(_lemmatizer.lemmatize(t))
        except Exception:
            lemmatized.append(t)
    return lemmatized


def preprocess(text: str, keep_original_case_for_ner: bool = False):
    """
    Full preprocessing pipeline used before feeding text into the
    intent classifier (TF-IDF vectorizer).

    Returns a tuple: (processed_string, tokens_list)
    """
    if not text or not text.strip():
        return "", []

    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    processed_string = " ".join(tokens)
    return processed_string, tokens


def strip_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))
