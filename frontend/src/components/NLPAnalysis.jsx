import React, { useState } from "react";

function ConfidenceBar({ value }) {
  const pct = Math.round((value || 0) * 100);
  const tone = pct >= 70 ? "high" : pct >= 50 ? "mid" : "low";
  return (
    <div className="confidence-bar">
      <div className={`confidence-bar__fill confidence-bar__fill--${tone}`} style={{ width: `${pct}%` }} />
      <span className="confidence-bar__label">{pct}%</span>
    </div>
  );
}

export default function NLPAnalysis({ analysis }) {
  const [expanded, setExpanded] = useState(true);

  if (!analysis) {
    return (
      <aside className="panel">
        <h2 className="panel__title">NLP Analysis</h2>
        <p className="panel__empty">Send a message to see the pipeline output.</p>
      </aside>
    );
  }

  const { intent, confidence, sentiment, sentiment_score, entities, coreference, low_confidence } = analysis;

  return (
    <aside className="panel">
      <div className="panel__header-row" onClick={() => setExpanded((e) => !e)} role="button" tabIndex={0}>
        <h2 className="panel__title">NLP Analysis</h2>
        <span className="panel__toggle">{expanded ? "−" : "+"}</span>
      </div>

      {expanded && (
        <div className="analysis-body">
          <div className="analysis-section">
            <h3 className="analysis-section__title">Intent</h3>
            <div className="analysis-intent-row">
              <code className="analysis-code">{intent}</code>
              {low_confidence && <span className="low-confidence-tag">low confidence</span>}
            </div>
            <ConfidenceBar value={confidence} />
          </div>

          <div className="analysis-section">
            <h3 className="analysis-section__title">Sentiment</h3>
            <span className={`sentiment-chip sentiment-chip--${(sentiment || "neutral").toLowerCase()}`}>
              {sentiment}
            </span>
            <span className="analysis-subtle"> score: {sentiment_score}</span>
          </div>

          <div className="analysis-section">
            <h3 className="analysis-section__title">Entities</h3>
            {entities && entities.length > 0 ? (
              <div className="entity-list">
                {entities.map((e, i) => (
                  <div className="entity-chip" key={i}>
                    <span className="entity-chip__type">{e.type}</span>
                    <span className="entity-chip__value">{String(e.value)}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="analysis-subtle">No entities detected</p>
            )}
          </div>

          {coreference && coreference.has_reference && (
            <div className="analysis-section">
              <h3 className="analysis-section__title">Context Resolution</h3>
              <p className="analysis-subtle">
                Reference resolved to: <strong>{coreference.resolved_object || "unknown"}</strong>
                {coreference.referenced_slot && (
                  <> (asking about <strong>{coreference.referenced_slot}</strong>)</>
                )}
              </p>
              <code className="analysis-code analysis-code--block">{coreference.resolved_message}</code>
            </div>
          )}
        </div>
      )}
    </aside>
  );
}
