"""
intent_classifier.py
---------------------
Loads the trained TF-IDF + Logistic Regression model and exposes a
predict() function that returns the predicted intent tag along with a
confidence score.

The actual training happens in train_model.py (run separately, or
automatically the first time the Flask app starts if no model file
is found).
"""

import os
import joblib
import numpy as np

from nlp.preprocessing import preprocess

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "intent_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "models", "label_encoder.pkl")

LOW_CONFIDENCE_THRESHOLD = 0.50


class IntentClassifier:
    """Wraps the trained model + vectorizer + label encoder and exposes
    a simple predict(text) -> (intent, confidence, all_scores) API."""

    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        self.loaded = False
        self.load()

    def load(self):
        try:
            if (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)
                    and os.path.exists(LABEL_ENCODER_PATH)):
                self.model = joblib.load(MODEL_PATH)
                self.vectorizer = joblib.load(VECTORIZER_PATH)
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
                self.loaded = True
            else:
                self.loaded = False
        except Exception as e:
            print(f"[IntentClassifier] Failed to load model: {e}")
            self.loaded = False

    def predict(self, text: str):
        """
        Returns a dict:
        {
            "intent": "product_search",
            "confidence": 0.94,
            "all_scores": {"product_search": 0.94, "greeting": 0.02, ...},
            "low_confidence": False
        }
        """
        if not self.loaded:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "all_scores": {},
                "low_confidence": True,
                "error": "Model not trained yet. Please run train_model.py",
            }

        processed, _tokens = preprocess(text)
        if not processed:
            return {
                "intent": "unknown",
                "confidence": 0.0,
                "all_scores": {},
                "low_confidence": True,
            }

        X = self.vectorizer.transform([processed])
        probabilities = self.model.predict_proba(X)[0]
        classes = self.label_encoder.inverse_transform(
            np.arange(len(probabilities))
        )

        all_scores = {
            str(cls): float(round(prob, 4))
            for cls, prob in zip(classes, probabilities)
        }

        best_idx = int(np.argmax(probabilities))
        best_intent = str(self.label_encoder.inverse_transform([best_idx])[0])
        best_confidence = float(round(probabilities[best_idx], 4))

        return {
            "intent": best_intent,
            "confidence": best_confidence,
            "all_scores": all_scores,
            "low_confidence": best_confidence < LOW_CONFIDENCE_THRESHOLD,
        }


# Singleton instance used across the Flask app
intent_classifier = IntentClassifier()
