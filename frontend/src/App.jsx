import React, { useCallback, useState } from "react";
import ChatWindow from "./components/ChatWindow.jsx";
import ContextPanel from "./components/ContextPanel.jsx";
import NLPAnalysis from "./components/NLPAnalysis.jsx";
import Dashboard from "./components/Dashboard.jsx";
import { sendMessage } from "./api.js";

function newId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function newConversationId() {
  if (window.crypto && window.crypto.randomUUID) return window.crypto.randomUUID();
  return "conv-" + newId();
}

export default function App() {
  const [view, setView] = useState("chat"); // "chat" | "dashboard"
  const [conversationId, setConversationId] = useState(newConversationId());
  const [messages, setMessages] = useState([]);
  const [context, setContext] = useState(null);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [analysisOpen, setAnalysisOpen] = useState(true);
  const [memoryOpen, setMemoryOpen] = useState(true);
  const [dashboardRefreshKey, setDashboardRefreshKey] = useState(0);

  const handleSend = useCallback(
    async (text) => {
      setError(null);
      const userMsg = { id: newId(), role: "user", text, timestamp: new Date().toISOString() };
      setMessages((prev) => [...prev, userMsg]);
      setLoading(true);

      try {
        const data = await sendMessage(text, conversationId);

        setMessages((prev) => [
          ...prev,
          {
            id: newId(),
            role: "ai",
            text: data.response,
            timestamp: new Date().toISOString(),
          },
        ]);
        // retroactively tag the user's message with detected sentiment
        setMessages((prev) =>
          prev.map((m) => (m.id === userMsg.id ? { ...m, sentiment: data.sentiment } : m))
        );

        setContext(data.context);
        setAnalysis({
          intent: data.intent,
          confidence: data.confidence,
          low_confidence: data.low_confidence,
          sentiment: data.sentiment,
          sentiment_score: data.sentiment_score,
          entities: data.entities,
          coreference: data.coreference,
        });
        setDashboardRefreshKey((k) => k + 1);
      } catch (err) {
        const message =
          err?.response?.data?.error ||
          "Could not reach the server. Please make sure the Flask backend is running on port 5000.";
        setError(message);
      } finally {
        setLoading(false);
      }
    },
    [conversationId]
  );

  const handleClear = useCallback(() => {
    setMessages([]);
    setContext(null);
    setAnalysis(null);
    setError(null);
    setConversationId(newConversationId());
  }, []);

  return (
    <div className="app-shell">
      <nav className="topnav">
        <div className="topnav__brand">
          <span className="topnav__brand-mark">◆</span>
          AI Context-Aware Conversation Understanding System
        </div>
        <div className="topnav__tabs">
          <button
            className={`topnav__tab ${view === "chat" ? "topnav__tab--active" : ""}`}
            onClick={() => setView("chat")}
          >
            Chat
          </button>
          <button
            className={`topnav__tab ${view === "dashboard" ? "topnav__tab--active" : ""}`}
            onClick={() => setView("dashboard")}
          >
            Dashboard
          </button>
        </div>
      </nav>

      {view === "chat" ? (
        <div className="main-layout">
          {memoryOpen && (
            <div className="side-col side-col--left">
              <ContextPanel context={context} />
            </div>
          )}
          <div className="center-col">
            <ChatWindow
              messages={messages}
              onSend={handleSend}
              loading={loading}
              error={error}
              onClear={handleClear}
              analysisOpen={analysisOpen}
              onToggleAnalysis={() => setAnalysisOpen((v) => !v)}
              memoryOpen={memoryOpen}
              onToggleMemory={() => setMemoryOpen((v) => !v)}
            />
          </div>
          {analysisOpen && (
            <div className="side-col side-col--right">
              <NLPAnalysis analysis={analysis} />
            </div>
          )}
        </div>
      ) : (
        <div className="dashboard-wrapper">
          <Dashboard refreshKey={dashboardRefreshKey} />
        </div>
      )}
    </div>
  );
}
