"""Optional Gemini-backed open-domain responses for chat messages."""

import json
import os
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def generate_llm_response(user_message: str, context: dict, history: list) -> str | None:
    """Generate a response with Gemini, or return None when unavailable."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
    endpoint = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?"
        f"{urlencode({'key': api_key})}"
    )
    contents = []
    for message in history[-8:]:
        previous_user_message = message.get("user_message")
        previous_response = message.get("ai_response")
        if previous_user_message:
            contents.append({"role": "user", "parts": [{"text": previous_user_message}]})
        if previous_response:
            contents.append({"role": "model", "parts": [{"text": previous_response}]})
    contents.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "system_instruction": {
            "parts": [{
                "text": (
                    "You are a helpful, context-aware assistant. Answer the user's actual question "
                    "directly and in the language they used, on any topic. Do not restrict answers "
                    "to products or orders. Be clear when you are uncertain. "
                    "Do not claim to access live websites, current prices, private order systems, "
                    "or perform real-world actions. Use this conversation context when relevant: "
                    f"{json.dumps(context, ensure_ascii=False, default=str)}"
                )
            }]
        },
        "contents": contents,
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 600},
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
        parts = result["candidates"][0]["content"]["parts"]
        answer = "".join(part.get("text", "") for part in parts).strip()
        return answer or None
    except Exception as error:
        print(f"[assistant] Gemini request failed ({type(error).__name__}); using local response.")
        return None