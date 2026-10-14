import React, { useEffect, useRef } from "react";
import Message from "./Message.jsx";
import InputBox from "./InputBox.jsx";

export default function ChatWindow({
  messages,
  onSend,
  loading,
  error,
  onClear,
  analysisOpen,
  onToggleAnalysis,
  memoryOpen,
  onToggleMemory,
}) {
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, loading]);

  return (
    <div className="chat-window">
      <div className="chat-header">
        <div className="chat-header__identity">
          <div className="chat-header__badge">NLP</div>
          <div>
            <h1 className="chat-header__title">Context-Aware Conversation Assistant</h1>
            <p className="chat-header__subtitle">
              Intent detection · Entity extraction · Sentiment · Memory
            </p>
          </div>
        </div>
        <div className="chat-header__actions">
          <button className="ghost-btn" onClick={onToggleMemory}>
            {memoryOpen ? "Hide Memory" : "Memory"}
          </button>
          <button className="ghost-btn" onClick={onToggleAnalysis}>
            {analysisOpen ? "Hide Analysis" : "NLP Analysis"}
          </button>
          <button className="ghost-btn ghost-btn--danger" onClick={onClear}>
            Clear Chat
          </button>
        </div>
      </div>

      <div className="chat-body" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="chat-empty">
            <p>Start a conversation - try:</p>
            <ul>
              <li>"I want to buy a laptop"</li>
              <li>"My budget is 60000"</li>
              <li>"HP"</li>
              <li>"Which one has 16GB RAM?"</li>
            </ul>
          </div>
        )}
        {messages.map((m) => (
          <Message
            key={m.id}
            role={m.role}
            text={m.text}
            timestamp={m.timestamp}
            sentiment={m.sentiment}
          />
        ))}
        {loading && (
          <div className="msg-row msg-row--ai">
            <div className="msg-avatar msg-avatar--ai">AI</div>
            <div className="msg-bubble msg-bubble--ai typing-bubble">
              <span className="typing-dot" />
              <span className="typing-dot" />
              <span className="typing-dot" />
            </div>
          </div>
        )}
        {error && <div className="chat-error">{error}</div>}
      </div>

      <InputBox onSend={onSend} disabled={loading} />
    </div>
  );
}
