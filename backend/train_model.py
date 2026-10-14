"""
train_model.py
----------------
Trains the intent classification model:

    1. Load intents.json
    2. Read patterns
    3. Preprocess text (tokenize, remove stopwords, lemmatize)
    4. Create TF-IDF vectors
    5. Train a Logistic Regression classifier
    6. Calculate accuracy on a held-out split
    7. Save the model, vectorizer, and label encoder
    8. Print training statistics

Run this from the backend/ directory:

    python train_model.py
"""

import os
import sys
import json
import time
import joblib
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nlp.preprocessing import preprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INTENTS_PATH = os.path.join(BASE_DIR, "data", "intents.json")
MODEL_DIR = os.path.join(BASE_DIR, "models")

MODEL_PATH = os.path.join(MODEL_DIR, "intent_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")


def load_intents():
    with open(INTENTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["intents"]


def build_dataset(intents):
    texts, labels = [], []
    for intent in intents:
        tag = intent["tag"]
        for pattern in intent["patterns"]:
            processed, _tokens = preprocess(pattern)
            if processed:
                texts.append(processed)
                labels.append(tag)
    return texts, labels


def main():
    print("=" * 60)
    print("Training NLP model...")
    print("=" * 60)
    start_time = time.time()

    os.makedirs(MODEL_DIR, exist_ok=True)

    intents = load_intents()
    texts, labels = build_dataset(intents)

    print(f"\nNumber of intents: {len(intents)}")
    print(f"Training examples: {len(texts)}")
    print("Model: TF-IDF + Logistic Regression\n")

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95)
    X = vectorizer.fit_transform(texts)

    # Stratified split when possible; fall back to a plain split for
    # very small classes where stratification would fail.
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    model = LogisticRegression(max_iter=1000, C=5.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy * 100:.1f}%\n")
    print("Classification report (held-out test split):")
    try:
        print(classification_report(
            y_test, y_pred,
            labels=np.unique(y_test),
            target_names=label_encoder.inverse_transform(np.unique(y_test)),
            zero_division=0,
        ))
    except Exception as e:
        print(f"(Could not print full classification report: {e})")

    # Refit on the FULL dataset before saving, so the deployed model
    # benefits from every example, not just the 80% training split.
    model.fit(X, y)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)

    elapsed = time.time() - start_time
    print(f"\nModel saved successfully to: {MODEL_DIR}")
    print(f"Training completed in {elapsed:.2f} seconds.")
    print("=" * 60)


if __name__ == "__main__":
    main()
