# AI Context-Aware Human Conversation Understanding System Using NLP

A real, working NLP-based conversational assistant that understands user
intent, extracts named entities, analyzes sentiment, and — most
importantly — **remembers context across turns** so it can correctly
resolve follow-up questions like *"which one?"*, *"its price"*, or
*"who created it?"*.

This is **not** a static/rule-based chatbot. Intent classification is a
trained TF-IDF + Logistic Regression model, entities are extracted with
spaCy plus custom domain rules, sentiment uses NLTK's VADER with
correction logic, and conversation memory is a dedicated Context
Manager backed by MongoDB (with an automatic local-file fallback).

---

## 1. Features

- **Intent classification** — TF-IDF + Logistic Regression trained on a
  21-intent dataset (`data/intents.json`), with a confidence score and a
  "please clarify" fallback below 50% confidence.
- **Named Entity Recognition** — spaCy's statistical NER combined with
  custom regex/keyword rules for domain entities spaCy doesn't know
  about out of the box (BRAND, BUDGET, RAM, ORDER_ID, etc).
- **Sentiment analysis** — NLTK VADER, tuned so informational requests
  ("I want to know the price") aren't misread as positive.
- **Context Manager** — a live, per-conversation memory of product,
  brand, budget, price, location, RAM, order ID, last intent, and more.
- **Coreference / context resolution** — resolves "it", "this", "that",
  "which one", "the previous one", "its RAM/price" against what was
  discussed earlier in the conversation.
- **Conversation memory** — every turn is persisted to MongoDB (or a
  local JSON fallback if MongoDB isn't running) with full context
  snapshots.
- **Modern React chat UI** — chat bubbles, typing indicator,
  Enter-to-send, voice input (Web Speech API), clear-conversation,
  and a live "Conversation Memory" panel.
- **NLP Analysis panel** — see the intent, confidence, sentiment, and
  entities behind every message, in real time.
- **Analytics dashboard** — total conversations/messages, intent and
  sentiment distributions, and top entities, rendered as charts.
- **Graceful error handling** — no Python stack traces ever reach the
  frontend; missing MongoDB, missing spaCy model, empty messages, and
  low-confidence intents are all handled explicitly.

---

## 2. Architecture

```
User Input
    │
    ▼
Text Preprocessing (clean → tokenize → remove stopwords → lemmatize)
    │
    ▼
TF-IDF Vectorization
    │
    ▼
Intent Classification (Logistic Regression) ──► confidence score
    │
    ▼
Named Entity Recognition (spaCy + custom rules)
    │
    ▼
Sentiment Analysis (VADER + corrections)
    │
    ▼
Context Resolution (coreference: it / this / which one / its RAM ...)
    │
    ▼
Context Manager (updates per-conversation memory)
    │
    ▼
Response Generation (template + context-aware)
    │
    ▼
Persisted to MongoDB / JSON fallback ──► returned to React frontend
```

---

## 3. Technology Stack

| Layer      | Technology                                              |
|------------|----------------------------------------------------------|
| Frontend   | React 18 (Vite), HTML5, CSS3, Recharts                   |
| Backend    | Python 3, Flask, Flask-CORS                               |
| NLP/ML     | NLTK, spaCy, scikit-learn (TF-IDF + Logistic Regression)  |
| Database   | MongoDB (with automatic local JSON fallback)              |
| Voice      | Web Speech API (browser-native, no extra dependency)      |

---

## 4. Folder Structure

```
AI-Context-Aware-Conversation/
│
├── backend/
│   ├── app.py                     Flask app + all API endpoints
│   ├── config.py                  Central configuration
│   │
│   ├── nlp/
│   │   ├── preprocessing.py       Tokenize / stopwords / lemmatize
│   │   ├── intent_classifier.py   Loads + runs the trained model
│   │   ├── entity_extractor.py    spaCy NER + custom entity rules
│   │   ├── sentiment.py           VADER sentiment + corrections
│   │   ├── context_manager.py     Per-conversation memory
│   │   ├── coreference.py         Pronoun / reference resolution
│   │   └── response_generator.py  Dynamic, context-aware replies
│   │
│   ├── models/                    intent_model.pkl, vectorizer.pkl, label_encoder.pkl
│   ├── data/
│   │   ├── intents.json           21 intents, training patterns
│   │   └── fallback_db/           auto-created JSON store (if MongoDB is off)
│   │
│   ├── database/
│   │   └── mongo.py               MongoDB + JSON-fallback persistence
│   │
│   ├── tests/
│   │   └── test_nlp.py            12 test scenarios (pytest)
│   │
│   ├── train_model.py             Trains + saves the intent model
│   └── requirements.txt
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatWindow.jsx     Main chat interface
│   │   │   ├── Message.jsx        Chat bubble
│   │   │   ├── InputBox.jsx       Text + voice input
│   │   │   ├── ContextPanel.jsx   Conversation memory sidebar
│   │   │   ├── NLPAnalysis.jsx    Live NLP pipeline output panel
│   │   │   └── Dashboard.jsx      Analytics dashboard + charts
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   ├── api.js                 Axios client for the Flask API
│   │   └── styles/style.css
│   └── public/
│
├── README.md
└── .gitignore
```

---

## 5. Installation (Windows PowerShell)

### 5.1 Prerequisites

- Python 3.10–3.12
- Node.js 18+
- (Optional) MongoDB Community Server — the app runs fine without it
  using the automatic JSON fallback.

### 5.2 Backend setup

```powershell
cd AI-Context-Aware-Conversation\backend

python -m venv venv
venv\Scripts\activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

> If `python -m spacy download` fails because of a managed/externally
> restricted Python environment, install the model directly instead:
> `pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl`

The first time NLTK runs it needs a few small corpora (punkt,
stopwords, wordnet, vader_lexicon). These download automatically the
first time you run `train_model.py` or `app.py`, as long as you have an
internet connection. If you don't, the app still runs using fallback
logic, just with slightly reduced tokenization/sentiment quality.

### 5.3 Train the NLP model

```powershell
python train_model.py
```

Expected output:

```
============================================================
Training NLP model...
============================================================

Number of intents: 21
Training examples: 321
Model: TF-IDF + Logistic Regression

Accuracy: <printed on your machine>

Model saved successfully to: ...\backend\models
Training completed in <n> seconds.
============================================================
```

> Note: the held-out test accuracy printed here is measured on a small
> 20% split of a deliberately compact demo dataset (321 examples across
> 21 intents), so it will look lower than a production system. The
> model is refit on the **full** dataset before being saved, and in
> practice performs well on inputs similar to the patterns in
> `data/intents.json`. Add more example patterns per intent for higher
> accuracy.

### 5.4 (Optional) Start MongoDB

If you have MongoDB installed locally, just make sure the service is
running (default `mongodb://localhost:27017/`). If it's not running or
not installed, the backend automatically falls back to a local JSON
file at `backend/data/fallback_db/store.json` — no configuration
needed either way.

### 5.5 Start the backend

To enable answers to general, open-domain questions, set a Gemini API key
in the same PowerShell window before starting the backend. Create a key
in Google AI Studio. Chat messages and recent conversation history are
sent to Gemini when this key is configured; do not share the key or commit it.

```powershell
$env:GEMINI_API_KEY = "your-gemini-api-key"
python app.py
```

`GEMINI_MODEL` is optional and defaults to `gemini-2.5-flash`. Without an
API key, the app falls back to local intent-based templates; those templates
can handle common flows but cannot generate answers to arbitrary factual
questions. Configure the key to enable open-domain answers.

The API will be available at `http://localhost:5000`. On first run
(if you skipped step 5.3), it will train the model automatically.

### 5.6 Frontend setup

Open a **second** PowerShell window:

```powershell
cd AI-Context-Aware-Conversation\frontend

npm install
npm run dev
```

The app will be available at `http://localhost:5173`. The Vite dev
server proxies `/api/*` requests to `http://localhost:5000`
automatically (see `vite.config.js`), so no extra configuration is
needed for local development.

---

## 6. API Documentation

All endpoints are prefixed with `/api`.

| Method | Endpoint                        | Description                                   |
|--------|----------------------------------|------------------------------------------------|
| POST   | `/api/chat`                      | Send a message, get an AI response + full NLP breakdown |
| GET    | `/api/conversations`             | List all conversations                         |
| GET    | `/api/conversations/<id>`        | Get all messages for one conversation          |
| DELETE | `/api/conversations/<id>`        | Delete a conversation                          |
| POST   | `/api/train`                     | Retrain the intent model on demand             |
| GET    | `/api/context/<conversation_id>` | Get the current context/memory for a conversation |
| GET    | `/api/analytics`                 | Aggregate stats for the dashboard              |
| GET    | `/api/health`                    | Health check (model loaded? which data store?) |

### POST `/api/chat`

Request:
```json
{
  "message": "I want an HP laptop under 60000",
  "conversation_id": "optional-existing-id"
}
```

Response:
```json
{
  "response": "Got it. You are looking for a HP laptop within your ₹60,000 budget.",
  "intent": "product_search",
  "confidence": 0.83,
  "low_confidence": false,
  "sentiment": "Neutral",
  "sentiment_score": 0.0,
  "entities": [
    { "type": "BRAND", "value": "HP" },
    { "type": "PRODUCT", "value": "laptop" },
    { "type": "BUDGET", "value": 60000 }
  ],
  "context": { "product": "laptop", "brand": "HP", "budget": 60000, "...": "..." },
  "coreference": { "has_reference": false, "resolved_object": null, "referenced_slot": null },
  "conversation_id": "generated-or-passed-in-id"
}
```

---

## 7. Example Conversation

```
User: I want to buy a laptop
AI:   Sure! What is your budget?

User: My budget is 60000
AI:   Got it, noted your budget of ₹60,000. Which brand do you prefer?

User: HP
AI:   Got it. You are looking for a HP laptop within your ₹60,000 budget.

User: Which one has 16GB RAM?
AI:   The HP laptop you're discussing has 16GB of RAM.
```

The system correctly resolves "which one" to the "HP laptop" discussed
earlier — this is the Context Manager + Coreference Resolution modules
working together, not a hard-coded script.

Another example:

```
User: Tell me about Python
AI:   That's an interesting question. I can look that up for you, though
      my main strength is helping with products and orders.

User: Who created it?
AI:   [The coreference module resolves "it" to "Python" from context]
```

---

## 8. How Each Piece Works

1. **Preprocessing** (`nlp/preprocessing.py`): lowercases text, strips
   noisy punctuation (while preserving numbers and ₹), tokenizes with
   NLTK, removes stopwords (except pronouns critical for coreference
   like "it"/"this"/"that"), and lemmatizes.

2. **Intent classification** (`nlp/intent_classifier.py` +
   `train_model.py`): the cleaned text is vectorized with TF-IDF
   (unigrams + bigrams) and classified with a multi-class Logistic
   Regression model. Returns the top intent and a confidence score;
   below 0.50 confidence, the app asks the user to clarify instead of
   guessing.

3. **NER** (`nlp/entity_extractor.py`): runs spaCy's `en_core_web_sm`
   NER for general entities (PERSON, ORG, LOCATION, DATE), and layers
   custom regex/keyword rules on top for e-commerce-specific entities
   (PRODUCT, BRAND, BUDGET, PRICE, RAM, ORDER_ID, COLOR, SIZE,
   CATEGORY, QUANTITY) that spaCy's default model has no concept of.

4. **Sentiment** (`nlp/sentiment.py`): NLTK's VADER lexicon/rule-based
   analyzer, with an "informational request" guard so phrases like "I
   want to know the price" aren't misclassified as Positive just
   because of the word "want".

5. **Context Manager** (`nlp/context_manager.py`): keeps a live,
   per-conversation dictionary of every slot filled so far (product,
   brand, budget, RAM, etc.), updated after every turn from the
   entities just extracted. It also tracks `last_discussed_object` —
   the "topic" pronouns should resolve to.

6. **Coreference resolution** (`nlp/coreference.py`): detects referring
   expressions ("it", "this", "that", "which one", "the previous one",
   "its X") and resolves them against `last_discussed_object`, plus
   detects *which attribute* is being asked about (e.g. "its RAM" →
   the `ram` slot).

7. **Response generation** (`nlp/response_generator.py`): chooses from
   a set of intent-specific templates and fills them dynamically from
   the current context — the same intent can produce very different
   replies depending on what's already known, which is what makes this
   genuinely context-aware rather than a fixed script.

8. **MongoDB persistence** (`database/mongo.py`): every turn (message,
   intent, entities, sentiment, response, context snapshot) is saved to
   the `messages` collection, and the latest context is upserted into
   the `contexts` collection. If MongoDB isn't reachable, an identical
   API is served by a local JSON file instead — the rest of the app
   doesn't need to know which one is active.

---

## 9. Testing

```powershell
cd AI-Context-Aware-Conversation\backend
venv\Scripts\activate
pytest tests\test_nlp.py -v
```

Covers: greeting detection, product search, budget/entity extraction,
brand preference, multi-entity extraction, positive/negative/neutral
sentiment (including the "want" false-positive guard), context
tracking across turns, "which one" and "it" pronoun resolution,
low-confidence handling, conversation turn counting, and a full
multi-turn integration flow.

---

## 10. Troubleshooting

| Symptom | Fix |
|---|---|
| `Model not trained yet` error from `/api/chat` | Run `python train_model.py` in `backend/`, then restart `app.py`. |
| `en_core_web_sm not found` warning on startup | Run `python -m spacy download en_core_web_sm` (or the direct pip-install fallback in section 5.2). The app still runs without it, using custom rules only. |
| MongoDB warnings on startup | Harmless — the app automatically uses the local JSON fallback at `backend/data/fallback_db/store.json`. Install/start MongoDB only if you want real persistent storage. |
| Frontend shows "Could not reach the server" | Make sure `python app.py` is running on port 5000 before starting the frontend, or check `VITE_API_URL` if you changed the backend port. |
| CORS errors in the browser console | Confirm `flask-cors` is installed and `app.py` is the one actually running (CORS is enabled for `/api/*` in `app.py`). |
| Voice input button doesn't appear | The Web Speech API is only available in Chromium-based browsers (Chrome, Edge). It's optional — typing always works. |

---

## 11. Notes on Accuracy & Extending the Project

This project ships with a compact, hand-written training set (21
intents, ~15 patterns each) so it trains in under a second and is easy
to inspect and extend for a course project or demo. To improve
accuracy for a specific domain:

- Add more example patterns per intent to `backend/data/intents.json`.
- Extend `PRODUCT_KEYWORDS` / `BRAND_KEYWORDS` / `CATEGORY_KEYWORDS` in
  `backend/nlp/entity_extractor.py` for your product catalog.
- Re-run `python train_model.py` (or call `POST /api/train`) after any
  change to `intents.json`.
