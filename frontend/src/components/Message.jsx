import React from "react";

function formatTime(ts) {
  try {
    return new Date(ts).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return "";
  }
}

export default function Message({ role, text, timestamp, sentiment }) {
  const isUser = role === "user";
  return (
    <div className={`msg-row ${isUser ? "msg-row--user" : "msg-row--ai"}`}>
      {!isUser && <div className="msg-avatar msg-avatar--ai">AI</div>}
      <div className={`msg-bubble ${isUser ? "msg-bubble--user" : "msg-bubble--ai"}`}>
        <p className="msg-text">{text}</p>
        <div className="msg-meta">
          <span>{formatTime(timestamp)}</span>
          {sentiment && isUser && (
            <span className={`sentiment-chip sentiment-chip--${sentiment.toLowerCase()}`}>
              {sentiment}
            </span>
          )}
        </div>
      </div>
      {isUser && <div className="msg-avatar msg-avatar--user">You</div>}
    </div>
  );
}
