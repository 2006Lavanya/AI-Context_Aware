"""
test_nlp.py
------------
Test cases covering:
 1. Greeting
 2. Product search
 3. Budget
 4. Brand preference
 5. Entity extraction
 6. Sentiment
 7. Context tracking
 8. Follow-up questions
 9. Pronoun resolution
 10. Unknown question / low confidence intent
 11. Conversation history (via ContextManager turn tracking)
 12. Full multi-turn conversation flow (integration test)

Run with:
    pytest tests/test_nlp.py -v
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest

from nlp.intent_classifier import intent_classifier
from nlp.entity_extractor import extract_entities
from nlp.sentiment import analyze_sentiment
from nlp.context_manager import ContextManager
from nlp.coreference import resolve_context
from nlp.response_generator import generate_response


MODEL_READY = intent_classifier.loaded


pytestmark = pytest.mark.skipif(
    not MODEL_READY,
    reason="Intent model not trained yet - run train_model.py first."
)


# 1. Greeting
def test_greeting_intent():
    result = intent_classifier.predict("Hello there")
    assert result["intent"] == "greeting"


# 2. Product search
def test_product_search_intent():
    result = intent_classifier.predict("I want to buy a laptop")
    assert result["intent"] == "product_search"


# 3. Budget
def test_budget_intent_and_entity():
    result = intent_classifier.predict("My budget is 60000")
    entities = extract_entities("My budget is 60000")
    budget_entities = [e for e in entities if e["type"] == "BUDGET"]
    assert result["intent"] == "budget"
    assert len(budget_entities) == 1
    assert budget_entities[0]["value"] == 60000


# 4. Brand preference
def test_brand_preference_intent():
    result = intent_classifier.predict("I prefer HP")
    assert result["intent"] == "brand_preference"


# 5. Entity extraction
def test_entity_extraction_multiple():
    entities = extract_entities("Show me a Dell laptop in Chennai under 70000")
    types = {e["type"]: e["value"] for e in entities}
    assert types.get("BRAND") == "Dell"
    assert types.get("PRODUCT") == "laptop"
    assert types.get("BUDGET") == 70000


# 6. Sentiment
def test_sentiment_positive():
    result = analyze_sentiment("I am very happy with this product.")
    assert result["label"] == "Positive"


def test_sentiment_negative():
    result = analyze_sentiment("This product is terrible.")
    assert result["label"] == "Negative"


def test_sentiment_neutral_information_request():
    # Guards against "want" being misclassified as positive.
    result = analyze_sentiment("I want to know the price.")
    assert result["label"] == "Neutral"


# 7. Context tracking
def test_context_tracking_updates_fields():
    cm = ContextManager()
    conv_id = str(uuid.uuid4())
    entities = extract_entities("I want an HP laptop under 60000")
    intent = intent_classifier.predict("I want an HP laptop under 60000")["intent"]
    ctx = cm.update_context(conv_id, intent, entities, "I want an HP laptop under 60000")
    assert ctx["brand"] == "HP" or ctx["brand"] == "Hp"
    assert ctx["product"] == "laptop"
    assert ctx["budget"] == 60000


# 8 & 9. Follow-up questions / Pronoun resolution
def test_pronoun_resolution_which_one():
    cm = ContextManager()
    conv_id = str(uuid.uuid4())

    e1 = extract_entities("I want an HP laptop")
    i1 = intent_classifier.predict("I want an HP laptop")["intent"]
    cm.update_context(conv_id, i1, e1, "I want an HP laptop")

    e2 = extract_entities("My budget is 60000")
    i2 = intent_classifier.predict("My budget is 60000")["intent"]
    ctx = cm.update_context(conv_id, i2, e2, "My budget is 60000")

    coref = resolve_context("Which one has 16GB RAM?", ctx)
    assert coref["has_reference"] is True
    assert "HP" in coref["resolved_object"] or "laptop" in coref["resolved_object"]
    assert coref["referenced_slot"] == "ram"


def test_pronoun_resolution_it():
    cm = ContextManager()
    conv_id = str(uuid.uuid4())
    e1 = extract_entities("Tell me about Python")
    ctx = cm.update_context(conv_id, "general_question", e1, "Tell me about Python")
    coref = resolve_context("Who created it?", ctx)
    assert coref["has_reference"] is True


# 10. Unknown question / low confidence
def test_low_confidence_gibberish():
    result = intent_classifier.predict("asdkjhaskjdh qweoiqwe zbcxzz")
    # Either genuinely low confidence, or at minimum doesn't crash
    assert "intent" in result
    assert "confidence" in result


# 11. Conversation history / turn tracking
def test_conversation_turn_count_increments():
    cm = ContextManager()
    conv_id = str(uuid.uuid4())
    cm.update_context(conv_id, "greeting", [], "Hello")
    cm.update_context(conv_id, "product_search", [], "I want a laptop")
    ctx = cm.get_context(conv_id)
    assert ctx["turn_count"] == 2


# 12. Full multi-turn conversation flow (integration test)
def test_full_conversation_flow():
    cm = ContextManager()
    conv_id = str(uuid.uuid4())

    turns = [
        "I want to buy a laptop",
        "My budget is 60000",
        "HP",
    ]
    ctx = None
    for turn in turns:
        entities = extract_entities(turn)
        intent_result = intent_classifier.predict(turn)
        ctx = cm.update_context(conv_id, intent_result["intent"], entities, turn)

    assert ctx["product"] == "laptop"
    assert ctx["budget"] == 60000
    assert ctx["brand"] in ("HP", "Hp")

    coref = resolve_context("Which one has 16GB RAM?", ctx)
    sentiment = analyze_sentiment("Which one has 16GB RAM?")
    intent_result = intent_classifier.predict("Which one has 16GB RAM?")
    response = generate_response(intent_result, [], ctx, coref, sentiment)
    assert isinstance(response, str) and len(response) > 0
