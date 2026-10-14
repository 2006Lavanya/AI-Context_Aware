"""
mongo.py
---------
Persistence layer for conversations, messages, and contexts.

Uses MongoDB when available (collections: conversations, messages,
contexts). If MongoDB cannot be reached (not installed / not running),
the app automatically falls back to a local JSON-file-based store so
the project can still be run and demonstrated without requiring a
MongoDB installation, per the project requirements.
"""

import os
import json
import time
import uuid
from datetime import datetime

try:
    from pymongo import MongoClient
    from pymongo.errors import ServerSelectionTimeoutError
    PYMONGO_INSTALLED = True
except ImportError:
    PYMONGO_INSTALLED = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FALLBACK_DIR = os.path.join(BASE_DIR, "data", "fallback_db")
FALLBACK_FILE = os.path.join(FALLBACK_DIR, "store.json")

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "nlp_conversation_system")


def _now_iso():
    return datetime.utcnow().isoformat() + "Z"


class JSONFallbackStore:
    """A minimal drop-in replacement for the MongoDB-backed store,
    used automatically when MongoDB is unavailable."""

    def __init__(self):
        os.makedirs(FALLBACK_DIR, exist_ok=True)
        if not os.path.exists(FALLBACK_FILE):
            self._write({"conversations": [], "messages": [], "contexts": []})

    def _read(self):
        try:
            with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {"conversations": [], "messages": [], "contexts": []}

    def _write(self, data):
        with open(FALLBACK_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def save_message(self, record: dict):
        data = self._read()
        data["messages"].append(record)
        self._write(data)

    def upsert_context(self, conversation_id: str, context: dict):
        data = self._read()
        found = False
        for c in data["contexts"]:
            if c["conversation_id"] == conversation_id:
                c["context"] = context
                c["updated_at"] = _now_iso()
                found = True
                break
        if not found:
            data["contexts"].append({
                "conversation_id": conversation_id,
                "context": context,
                "updated_at": _now_iso(),
            })
        self._write(data)

    def get_context(self, conversation_id: str):
        data = self._read()
        for c in data["contexts"]:
            if c["conversation_id"] == conversation_id:
                return c["context"]
        return None

    def get_conversation_messages(self, conversation_id: str):
        data = self._read()
        return [m for m in data["messages"] if m["conversation_id"] == conversation_id]

    def get_all_conversations(self):
        data = self._read()
        conv_ids = {}
        for m in data["messages"]:
            cid = m["conversation_id"]
            if cid not in conv_ids:
                conv_ids[cid] = {"conversation_id": cid, "message_count": 0, "started_at": m["timestamp"]}
            conv_ids[cid]["message_count"] += 1
            conv_ids[cid]["last_message_at"] = m["timestamp"]
        return list(conv_ids.values())

    def delete_conversation(self, conversation_id: str):
        data = self._read()
        data["messages"] = [m for m in data["messages"] if m["conversation_id"] != conversation_id]
        data["contexts"] = [c for c in data["contexts"] if c["conversation_id"] != conversation_id]
        self._write(data)

    def get_all_messages(self):
        return self._read()["messages"]


class MongoStore:
    """MongoDB-backed persistence."""

    def __init__(self):
        self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        self.client.server_info()  # forces connection check
        self.db = self.client[MONGO_DB_NAME]
        self.messages = self.db["messages"]
        self.contexts = self.db["contexts"]
        self.conversations = self.db["conversations"]

    def save_message(self, record: dict):
        self.messages.insert_one(dict(record))
        self.conversations.update_one(
            {"conversation_id": record["conversation_id"]},
            {"$set": {"last_message_at": record["timestamp"]},
             "$setOnInsert": {"started_at": record["timestamp"]},
             "$inc": {"message_count": 1}},
            upsert=True,
        )

    def upsert_context(self, conversation_id: str, context: dict):
        self.contexts.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"context": context, "updated_at": _now_iso()}},
            upsert=True,
        )

    def get_context(self, conversation_id: str):
        doc = self.contexts.find_one({"conversation_id": conversation_id})
        return doc["context"] if doc else None

    def get_conversation_messages(self, conversation_id: str):
        cursor = self.messages.find({"conversation_id": conversation_id}, {"_id": 0}).sort("timestamp", 1)
        return list(cursor)

    def get_all_conversations(self):
        cursor = self.conversations.find({}, {"_id": 0})
        return list(cursor)

    def delete_conversation(self, conversation_id: str):
        self.messages.delete_many({"conversation_id": conversation_id})
        self.contexts.delete_many({"conversation_id": conversation_id})
        self.conversations.delete_many({"conversation_id": conversation_id})

    def get_all_messages(self):
        cursor = self.messages.find({}, {"_id": 0})
        return list(cursor)


def get_store():
    """Factory that returns a MongoStore if MongoDB is reachable,
    otherwise transparently falls back to the JSON store."""
    if PYMONGO_INSTALLED:
        try:
            return MongoStore(), "mongodb"
        except Exception as e:
            print(f"[database] MongoDB unavailable ({e}). Falling back to local JSON store.")
    else:
        print("[database] pymongo not installed. Falling back to local JSON store.")
    return JSONFallbackStore(), "json_fallback"


def new_conversation_id():
    return str(uuid.uuid4())


def build_message_record(conversation_id, user_message, intent_result, entities, sentiment, ai_response, context_snapshot):
    return {
        "conversation_id": conversation_id,
        "timestamp": _now_iso(),
        "user_message": user_message,
        "detected_intent": intent_result.get("intent"),
        "confidence": intent_result.get("confidence"),
        "entities": entities,
        "sentiment": sentiment.get("label"),
        "sentiment_score": sentiment.get("score"),
        "ai_response": ai_response,
        "context_snapshot": context_snapshot,
    }
