"""
app.py
-------
Main Flask application exposing the REST API for the AI Context-Aware
Human Conversation Understanding System.

Pipeline executed on every /api/chat call:

    User Input -> Preprocessing -> Intent Classification -> NER ->
    Sentiment Analysis -> Context Resolution (coreference) ->
    Context Manager Update -> Response Generation -> Persist to DB
"""

import os
import sys
import subprocess
import traceback

from flask import Flask, request, jsonify
from flask_cors import CORS

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config
from nlp.intent_classifier import intent_classifier, IntentClassifier
from nlp.entity_extractor import extract_entities
from nlp.sentiment import analyze_sentiment
from nlp.context_manager import context_manager
from nlp.coreference import resolve_context
from nlp.response_generator import generate_response
from nlp.general_assistant import generate_llm_response
from database.mongo import get_store, new_conversation_id, build_message_record

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

store, store_type = get_store()
print(f"[app] Using data store: {store_type}")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_model.pkl")


def ensure_model_trained():
    """If no trained model exists yet, train it automatically so the
    project can run out of the box with a single `python app.py`."""
    if not os.path.exists(MODEL_PATH):
        print("[app] No trained model found. Training model automatically...")
        try:
            subprocess.run([sys.executable, os.path.join(BASE_DIR, "train_model.py")], check=True)
            intent_classifier.load()
        except Exception as e:
            print(f"[app] Automatic training failed: {e}")


ensure_model_trained()


# -------------------------------------------------------------------
# Error handling helpers - never leak Python stack traces to clients.
# -------------------------------------------------------------------
def error_response(message, status_code=400):
    return jsonify({"error": message}), status_code


@app.errorhandler(404)
def not_found(_e):
    return error_response("The requested resource was not found.", 404)


@app.errorhandler(500)
def server_error(_e):
    return error_response("An internal server error occurred. Please try again.", 500)


# -------------------------------------------------------------------
# POST /api/chat
# -------------------------------------------------------------------
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(silent=True)
        if not data:
            return error_response("Invalid or missing JSON body.", 400)

        user_message = (data.get("message") or "").strip()
        conversation_id = data.get("conversation_id") or new_conversation_id()

        if not user_message:
            return error_response("Message cannot be empty.", 400)

        if not intent_classifier.loaded:
            return error_response(
                "NLP model is not available yet. Please run train_model.py and restart the server.",
                503,
            )

        # If this conversation isn't in the live in-memory context cache
        # (e.g. the server was restarted), try restoring it from
        # persistent storage before proceeding, so long-running
        # conversations survive restarts.
        if not context_manager.has_context(conversation_id):
            try:
                stored_context = store.get_context(conversation_id)
                if stored_context:
                    context_manager.load_context(conversation_id, stored_context)
            except Exception as restore_err:
                print(f"[app] Warning: failed to restore context: {restore_err}")

        # ---- NLP PIPELINE ----
        intent_result = intent_classifier.predict(user_message)
        entities = extract_entities(user_message)
        sentiment = analyze_sentiment(user_message)

        context_before = context_manager.get_context(conversation_id)
        coref = resolve_context(user_message, context_before)

        context_after = context_manager.update_context(
            conversation_id, intent_result["intent"], entities, user_message
        )

        try:
            history = store.get_conversation_messages(conversation_id)
        except Exception as history_err:
            print(f"[app] Warning: failed to load conversation history: {history_err}")
            history = []

        ai_response = generate_llm_response(user_message, context_after, history)
        if ai_response is None:
            ai_response = generate_response(intent_result, entities, context_after, coref, sentiment)

        context_summary = context_manager.get_summary(conversation_id)

        # ---- PERSIST ----
        try:
            record = build_message_record(
                conversation_id, user_message, intent_result, entities,
                sentiment, ai_response, context_summary,
            )
            store.save_message(record)
            store.upsert_context(conversation_id, context_after)
        except Exception as db_err:
            print(f"[app] Warning: failed to persist message: {db_err}")

        return jsonify({
            "response": ai_response,
            "intent": intent_result["intent"],
            "confidence": intent_result["confidence"],
            "low_confidence": intent_result.get("low_confidence", False),
            "sentiment": sentiment["label"],
            "sentiment_score": sentiment["score"],
            "entities": entities,
            "context": context_summary,
            "coreference": coref,
            "conversation_id": conversation_id,
        })

    except Exception:
        print(traceback.format_exc())
        return error_response("Something went wrong while processing your message.", 500)


# -------------------------------------------------------------------
# GET /api/conversations
# -------------------------------------------------------------------
@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    try:
        conversations = store.get_all_conversations()
        return jsonify({"conversations": conversations})
    except Exception:
        print(traceback.format_exc())
        return error_response("Could not fetch conversations.", 500)


# -------------------------------------------------------------------
# GET /api/conversations/<id>
# -------------------------------------------------------------------
@app.route("/api/conversations/<conversation_id>", methods=["GET"])
def get_conversation(conversation_id):
    try:
        messages = store.get_conversation_messages(conversation_id)
        if not messages:
            return error_response("Conversation not found.", 404)
        return jsonify({"conversation_id": conversation_id, "messages": messages})
    except Exception:
        print(traceback.format_exc())
        return error_response("Could not fetch conversation.", 500)


# -------------------------------------------------------------------
# DELETE /api/conversations/<id>
# -------------------------------------------------------------------
@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation(conversation_id):
    try:
        store.delete_conversation(conversation_id)
        context_manager.reset_context(conversation_id)
        return jsonify({"message": "Conversation deleted successfully."})
    except Exception:
        print(traceback.format_exc())
        return error_response("Could not delete conversation.", 500)


# -------------------------------------------------------------------
# POST /api/train
# -------------------------------------------------------------------
@app.route("/api/train", methods=["POST"])
def train():
    try:
        result = subprocess.run(
            [sys.executable, os.path.join(BASE_DIR, "train_model.py")],
            check=True, capture_output=True, text=True,
        )
        intent_classifier.load()
        return jsonify({
            "message": "Model retrained successfully.",
            "log": result.stdout[-3000:],
        })
    except subprocess.CalledProcessError as e:
        return error_response(f"Training failed: {e.stderr[-1000:] if e.stderr else 'unknown error'}", 500)
    except Exception:
        print(traceback.format_exc())
        return error_response("Could not retrain model.", 500)


# -------------------------------------------------------------------
# GET /api/context/<conversation_id>
# -------------------------------------------------------------------
@app.route("/api/context/<conversation_id>", methods=["GET"])
def get_context(conversation_id):
    try:
        summary = context_manager.get_summary(conversation_id)
        return jsonify({"conversation_id": conversation_id, "context": summary})
    except Exception:
        print(traceback.format_exc())
        return error_response("Could not fetch context.", 500)


# -------------------------------------------------------------------
# GET /api/analytics
# -------------------------------------------------------------------
@app.route("/api/analytics", methods=["GET"])
def analytics():
    try:
        all_messages = store.get_all_messages()
        total_messages = len(all_messages)

        conversation_ids = set(m["conversation_id"] for m in all_messages)
        total_conversations = len(conversation_ids)

        intent_counts = {}
        sentiment_counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
        entity_counts = {}

        for m in all_messages:
            intent = m.get("detected_intent", "unknown")
            intent_counts[intent] = intent_counts.get(intent, 0) + 1

            sentiment = m.get("sentiment", "Neutral")
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1

            for entity in m.get("entities", []):
                etype = entity.get("type", "UNKNOWN")
                entity_counts[etype] = entity_counts.get(etype, 0) + 1

        avg_messages_per_conversation = (
            round(total_messages / total_conversations, 2) if total_conversations else 0
        )

        top_intents = sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        return jsonify({
            "total_conversations": total_conversations,
            "total_messages": total_messages,
            "avg_messages_per_conversation": avg_messages_per_conversation,
            "intent_distribution": dict(top_intents),
            "sentiment_distribution": sentiment_counts,
            "entity_distribution": dict(top_entities),
            "data_store": store_type,
        })
    except Exception:
        print(traceback.format_exc())
        return error_response("Could not compute analytics.", 500)


# -------------------------------------------------------------------
# GET /api/health (simple health check, useful for the frontend and
# for verifying the server is up during setup)
# -------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": intent_classifier.loaded,
        "data_store": store_type,
    })


if __name__ == "__main__":
    app.run(host=Config.HOST, port=Config.PORT, debug=Config.DEBUG)
