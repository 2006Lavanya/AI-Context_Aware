"""
context_manager.py
--------------------
The Context Manager is the heart of the "context-aware" behavior of
this system. It keeps a running, per-conversation memory of everything
relevant that has been discussed so far, and updates it dynamically as
new entities and intents are detected in each turn.

An in-memory dict acts as the short-term/active-session context cache
(fast access during a live conversation), while the full turn-by-turn
history is persisted to MongoDB (or the JSON fallback) by database/mongo.py
for long-term/analytics purposes.
"""

import time

# Maps an entity type detected by entity_extractor.py to the context
# field it should update. This is what lets "BRAND -> HP" update
# context["brand"], "BUDGET -> 60000" update context["budget"], etc.
ENTITY_TO_CONTEXT_FIELD = {
    "PRODUCT": "product",
    "BRAND": "brand",
    "BUDGET": "budget",
    "PRICE": "price",
    "LOCATION": "location",
    "DATE": "date",
    "COLOR": "color",
    "SIZE": "size",
    "CATEGORY": "category",
    "ORDER_ID": "order_id",
    "QUANTITY": "quantity",
    "RAM": "ram",
    "PERSON": "person",
    "ORGANIZATION": "organization",
}

DEFAULT_CONTEXT_FIELDS = [
    "product", "brand", "budget", "price", "location", "date", "color",
    "size", "category", "order_id", "quantity", "ram", "person",
    "organization",
]


class ContextManager:
    """
    In-memory store of active conversation contexts, keyed by
    conversation_id. In a production system this would live in Redis;
    for this project an in-process dict is sufficient and is backed up
    to MongoDB after every turn so nothing is lost on restart.
    """

    def __init__(self):
        self._contexts = {}

    def get_context(self, conversation_id: str) -> dict:
        if conversation_id not in self._contexts:
            self._contexts[conversation_id] = self._new_context()
        return self._contexts[conversation_id]

    def has_context(self, conversation_id: str) -> bool:
        return conversation_id in self._contexts

    def _new_context(self) -> dict:
        ctx = {field: None for field in DEFAULT_CONTEXT_FIELDS}
        ctx.update({
            "last_intent": None,
            "last_entities": [],
            "last_discussed_object": None,   # e.g. "HP laptop"
            "last_user_message": None,
            "preferences": {},
            "turn_count": 0,
            "updated_at": time.time(),
        })
        return ctx

    def update_context(self, conversation_id: str, intent: str, entities: list, user_message: str):
        """Dynamically update the context with the newly detected intent
        and entities for this turn."""
        ctx = self.get_context(conversation_id)

        for entity in entities:
            field = ENTITY_TO_CONTEXT_FIELD.get(entity["type"])
            if field:
                ctx[field] = entity["value"]
                # track light-weight preference history
                ctx["preferences"][field] = entity["value"]

        ctx["last_intent"] = intent
        ctx["last_entities"] = entities
        ctx["last_user_message"] = user_message
        ctx["turn_count"] += 1
        ctx["updated_at"] = time.time()

        # Update the "last discussed object" - the noun phrase the user
        # is currently focused on. This is what pronouns like "it" /
        # "this" / "that" / "which one" will resolve to. We only refresh
        # this from PRODUCT/BRAND (the "main noun" of the conversation),
        # never from incidental detail entities like RAM/COLOR/QUANTITY,
        # so a follow-up detail question doesn't accidentally hijack what
        # "it"/"this"/"which one" refers to. If neither is known yet, we
        # fall back to a general_question topic (e.g. "Python").
        if ctx.get("brand") and ctx.get("product"):
            ctx["last_discussed_object"] = f"{ctx['brand']} {ctx['product']}"
        elif ctx.get("product"):
            ctx["last_discussed_object"] = ctx["product"]
        elif ctx.get("brand"):
            ctx["last_discussed_object"] = ctx["brand"]
        elif ctx["last_discussed_object"] is None and intent == "general_question" and entities:
            non_numeric = [e for e in entities if e["type"] not in ("BUDGET", "PRICE", "QUANTITY", "RAM")]
            if non_numeric:
                ctx["last_discussed_object"] = str(non_numeric[0]["value"])

        self._contexts[conversation_id] = ctx
        return ctx

    def reset_context(self, conversation_id: str):
        self._contexts[conversation_id] = self._new_context()
        return self._contexts[conversation_id]

    def load_context(self, conversation_id: str, context_dict: dict):
        """Used to restore context from MongoDB when resuming a
        previous conversation."""
        self._contexts[conversation_id] = context_dict

    def get_summary(self, conversation_id: str) -> dict:
        """A trimmed-down, display-friendly version of the context for
        the frontend's Context/Memory panel."""
        ctx = self.get_context(conversation_id)
        return {
            "product": ctx.get("product"),
            "brand": ctx.get("brand"),
            "budget": ctx.get("budget"),
            "price": ctx.get("price"),
            "location": ctx.get("location"),
            "color": ctx.get("color"),
            "size": ctx.get("size"),
            "category": ctx.get("category"),
            "ram": ctx.get("ram"),
            "order_id": ctx.get("order_id"),
            "last_intent": ctx.get("last_intent"),
            "last_discussed_object": ctx.get("last_discussed_object"),
            "turn_count": ctx.get("turn_count"),
        }


# Singleton instance shared across the Flask app
context_manager = ContextManager()
