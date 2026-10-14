"""
config.py
----------
Central configuration for the Flask backend.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class Config:
    DEBUG = os.environ.get("FLASK_DEBUG", "True") == "True"
    HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
    PORT = int(os.environ.get("FLASK_PORT", 5000))

    MODEL_DIR = os.path.join(BASE_DIR, "models")
    DATA_DIR = os.path.join(BASE_DIR, "data")
    INTENTS_FILE = os.path.join(DATA_DIR, "intents.json")

    MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017/")
    MONGO_DB_NAME = os.environ.get("MONGO_DB_NAME", "nlp_conversation_system")

    LOW_CONFIDENCE_THRESHOLD = 0.50

    # Allowed frontend origins for CORS (Vite dev server default port)
    CORS_ORIGINS = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ]
