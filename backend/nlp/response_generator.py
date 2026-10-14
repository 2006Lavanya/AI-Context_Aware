"""
response_generator.py
-----------------------
Generates the AI's reply using:
  - the detected intent
  - extracted entities
  - the current conversation context
  - the coreference/context-resolution result
  - the detected sentiment

This is template-based (not a single giant if/else chain): each intent
maps to a small set of response templates with placeholders that are
filled in dynamically from context. Which template is chosen, and how
placeholders are filled, depends on what information is actually known
at this point in the conversation - so the same intent can produce very
different responses depending on context, which is what makes this
"real" context-aware generation rather than a static script.
"""

import random

LOW_CONFIDENCE_RESPONSES = [
    "I'm not completely sure what you mean. Could you please clarify?",
    "Sorry, I didn't quite catch that. Could you rephrase it?",
    "I want to make sure I understand - could you say that a bit differently?",
]

NEGATIVE_SENTIMENT_PREFIXES = [
    "I'm sorry to hear that. ",
    "I understand your frustration. ",
    "I apologize for the trouble. ",
]


def _fmt_currency(value):
    try:
        return f"₹{int(value):,}"
    except (TypeError, ValueError):
        return str(value)


def _format_known_slots(context: dict) -> str:
    """Builds a natural-language summary of what's known so far, e.g.
    'an HP laptop within your ₹60,000 budget'."""
    parts = []
    brand = context.get("brand")
    product = context.get("product") or "product"
    budget = context.get("budget")
    color = context.get("color")
    size = context.get("size")

    descriptor = " ".join(filter(None, [color, size, brand, product]))
    descriptor = descriptor.strip() or product

    sentence = f"a {descriptor}" if descriptor else "that"
    if budget:
        sentence += f" within your {_fmt_currency(budget)} budget"
    return sentence


def _generate_for_intent(intent: str, entities: list, context: dict, coref: dict, sentiment: dict) -> str:
    entity_types = {e["type"] for e in entities}

    # ---------------- product_search ----------------
    if intent == "product_search":
        if context.get("budget") is None:
            return "Sure! What is your budget?"
        elif context.get("brand") is None:
            return "Great, noted. Which brand do you prefer?"
        else:
            return f"Got it. You are looking for {_format_known_slots(context)}."

    # ---------------- budget ----------------
    if intent == "budget" or ("BUDGET" in entity_types and context.get("brand") is None and not coref["has_reference"]):
        budget = context.get("budget")
        budget_str = _fmt_currency(budget) if budget else "that budget"
        if context.get("brand") is None:
            return f"Got it, noted your budget of {budget_str}. Which brand do you prefer?"
        return f"Got it. You are looking for {_format_known_slots(context)}."

    # ---------------- brand_preference ----------------
    if intent == "brand_preference" or ("BRAND" in entity_types and not coref["has_reference"]):
        return f"Got it. You are looking for {_format_known_slots(context)}."

    # ---------------- product_details (with coreference) ----------------
    if intent in ("product_details", "price_inquiry") or coref["has_reference"]:
        if coref["has_reference"] and coref["resolved_object"]:
            slot = coref["referenced_slot"]
            obj = coref["resolved_object"]
            if slot == "ram":
                ram = context.get("ram")
                if ram:
                    return f"The {obj} you're discussing has {ram} of RAM."
                return f"You are asking about the {obj} discussed earlier. I can help you compare the available RAM options."
            elif slot in ("price", "budget"):
                budget = context.get("budget") or context.get("price")
                if budget:
                    return f"The {obj} is priced around {_fmt_currency(budget)}, within the range you mentioned."
                return f"Let me check the current price of the {obj} for you."
            elif slot:
                return f"Here's what I know about the {slot} of the {obj} you were discussing."
            else:
                return f"The {obj} you were discussing is {('within your ' + _fmt_currency(context['budget']) + ' budget') if context.get('budget') else 'a great option'}. What would you like to know about it?"
        elif intent == "price_inquiry":
            return "Let me check the price for you. Could you tell me which product you mean?"
        else:
            return "Sure, could you tell me which product you'd like more details about?"

    # ---------------- product_recommendation ----------------
    if intent == "product_recommendation":
        if context.get("budget") and context.get("brand"):
            return f"Based on your {_fmt_currency(context['budget'])} budget and preference for {context['brand']}, I can suggest a few great options."
        return "I'd be happy to recommend something. What's your budget, and do you have a brand preference?"

    # ---------------- product_comparison ----------------
    if intent == "product_comparison":
        return "Sure, let me compare those options for you based on price, specifications, and features."

    # ---------------- order_status / tracking ----------------
    if intent in ("order_status", "order_tracking"):
        order_id = context.get("order_id")
        if order_id:
            return f"Checking the status of order {order_id} now. It is currently being processed."
        return "Sure, could you please share your order ID so I can check its status?"

    # ---------------- cancel_order ----------------
    if intent == "cancel_order":
        order_id = context.get("order_id")
        if order_id:
            return f"Your cancellation request for order {order_id} has been received and is being processed."
        return "I can help you cancel your order. Could you share the order ID?"

    # ---------------- return_request ----------------
    if intent == "return_request":
        order_id = context.get("order_id")
        if order_id:
            return f"I've started a return request for order {order_id}. You'll receive pickup details shortly."
        return "Sure, I can help you start a return. Could you provide the order ID?"

    # ---------------- complaint ----------------
    if intent == "complaint":
        return "I'm really sorry to hear that. Could you share a few more details so I can help resolve this right away?"

    # ---------------- feedback ----------------
    if intent == "feedback":
        return "Thank you so much for your feedback! We're glad to hear about your experience."

    # ---------------- confirmation / rejection / clarification ----------------
    if intent == "confirmation":
        return "Great, confirmed! Let's proceed."
    if intent == "rejection":
        return "No problem, let's try something else. What would you like instead?"
    if intent == "clarification":
        return "Sure, let me clarify. Could you tell me exactly which part is unclear?"

    # ---------------- help ----------------
    if intent == "help":
        return "I can answer questions on a wide range of topics, as well as help with products and orders. What would you like to know?"

    # ---------------- greeting / goodbye / thanks ----------------
    if intent == "greeting":
        return random.choice([
            "Hello! How can I help you today?",
            "Hi there! What can I do for you?",
            "Hey! I'm here to help. What are you looking for?",
        ])
    if intent == "goodbye":
        return random.choice([
            "Goodbye! Have a great day!",
            "See you soon! Feel free to come back anytime.",
        ])
    if intent == "thanks":
        return random.choice([
            "You're welcome! Happy to help.",
            "Anytime! Let me know if you need anything else.",
        ])

    # ---------------- general_question ----------------
    if intent == "general_question":
        return "I can help with questions on any topic. What would you like to know?"

    # ---------------- fallback ----------------
    return "I see. Could you tell me a bit more so I can help you better?"


def generate_response(intent_result: dict, entities: list, context: dict, coref: dict, sentiment: dict) -> str:
    """
    Main entry point used by app.py. Handles the low-confidence guard
    first, then dispatches to the intent/entity/context-aware generator.
    """
    if intent_result.get("low_confidence"):
        return random.choice(LOW_CONFIDENCE_RESPONSES)

    intent = intent_result.get("intent", "unknown")
    reply = _generate_for_intent(intent, entities, context, coref, sentiment)

    if sentiment.get("label") == "Negative" and intent not in ("complaint", "return_request", "cancel_order"):
        reply = random.choice(NEGATIVE_SENTIMENT_PREFIXES) + reply

    return reply
