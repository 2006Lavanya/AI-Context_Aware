import io
import json

from nlp import general_assistant
from nlp.response_generator import generate_response


def test_returns_none_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    assert general_assistant.generate_llm_response("Hello", {}, []) is None


def test_uses_gemini_response_with_context_and_history(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        response = {
            "candidates": [{"content": {"parts": [{"text": "A useful answer."}]}}]
        }
        return io.BytesIO(json.dumps(response).encode("utf-8"))

    monkeypatch.setattr(general_assistant, "urlopen", fake_urlopen)
    answer = general_assistant.generate_llm_response(
        "Who created it?",
        {"last_discussed_object": "Python"},
        [{"user_message": "Tell me about Python", "ai_response": "Python is a language."}],
    )

    payload = json.loads(captured["request"].data.decode("utf-8"))
    assert answer == "A useful answer."
    assert captured["timeout"] == 20
    assert payload["contents"][-1]["parts"][0]["text"] == "Who created it?"
    assert payload["contents"][0]["parts"][0]["text"] == "Tell me about Python"
    assert "Python" in payload["system_instruction"]["parts"][0]["text"]
    assert "on any topic" in payload["system_instruction"]["parts"][0]["text"]
    assert "Do not restrict answers to products or orders" in payload["system_instruction"]["parts"][0]["text"]


def test_local_general_question_response_is_not_product_limited():
    answer = generate_response(
        {"intent": "general_question", "low_confidence": False},
        [],
        {},
        {"has_reference": False},
        {"label": "Neutral"},
    )

    assert "any topic" in answer
    assert "products and orders" not in answer