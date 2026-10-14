import React, { useEffect, useRef, useState } from "react";

export default function InputBox({ onSend, disabled }) {
  const [text, setText] = useState("");
  const [listening, setListening] = useState(false);
  const recognitionRef = useRef(null);
  const [voiceSupported, setVoiceSupported] = useState(false);

  useEffect(() => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      setVoiceSupported(true);
      const recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = "en-US";

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        setText((prev) => (prev ? prev + " " + transcript : transcript));
      };
      recognition.onend = () => setListening(false);
      recognition.onerror = () => setListening(false);

      recognitionRef.current = recognition;
    }
  }, []);

  const handleSend = () => {
    const trimmed = text.trim();
    if (!trimmed || disabled) return;
    onSend(trimmed);
    setText("");
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const toggleVoice = () => {
    if (!recognitionRef.current) return;
    if (listening) {
      recognitionRef.current.stop();
      setListening(false);
    } else {
      try {
        recognitionRef.current.start();
        setListening(true);
      } catch {
        setListening(false);
      }
    }
  };

  return (
    <div className="input-bar">
      <textarea
        className="input-box"
        placeholder="Type a message... (e.g. I want to buy a laptop)"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        rows={1}
        disabled={disabled}
      />
      {voiceSupported && (
        <button
          type="button"
          className={`mic-btn ${listening ? "mic-btn--active" : ""}`}
          onClick={toggleVoice}
          title="Voice input"
          aria-label="Voice input"
        >
          {listening ? "●" : "🎤"}
        </button>
      )}
      <button
        type="button"
        className="send-btn"
        onClick={handleSend}
        disabled={disabled || !text.trim()}
      >
        Send
      </button>
    </div>
  );
}
