import React from "react";

const FIELD_LABELS = [
  { key: "product", label: "Product" },
  { key: "brand", label: "Brand" },
  { key: "budget", label: "Budget", currency: true },
  { key: "price", label: "Price", currency: true },
  { key: "category", label: "Category" },
  { key: "color", label: "Color" },
  { key: "size", label: "Size" },
  { key: "ram", label: "RAM" },
  { key: "location", label: "Location" },
  { key: "order_id", label: "Order ID" },
  { key: "last_intent", label: "Last Intent" },
  { key: "last_discussed_object", label: "Discussing" },
];

function formatValue(field, value) {
  if (value === null || value === undefined || value === "") return "—";
  if (field.currency) {
    const num = Number(value);
    return Number.isNaN(num) ? String(value) : `₹${num.toLocaleString("en-IN")}`;
  }
  return String(value);
}

export default function ContextPanel({ context }) {
  if (!context) {
    return (
      <aside className="panel">
        <h2 className="panel__title">Conversation Memory</h2>
        <p className="panel__empty">No context yet. Start chatting to build memory.</p>
      </aside>
    );
  }

  const knownCount = FIELD_LABELS.filter(
    (f) => context[f.key] !== null && context[f.key] !== undefined
  ).length;

  return (
    <aside className="panel">
      <div className="panel__header-row">
        <h2 className="panel__title">Conversation Memory</h2>
        <span className="panel__badge">{context.turn_count || 0} turns</span>
      </div>
      <div className="memory-grid">
        {FIELD_LABELS.map((field) => (
          <div className="memory-row" key={field.key}>
            <span className="memory-row__label">{field.label}</span>
            <span
              className={`memory-row__value ${
                context[field.key] !== null && context[field.key] !== undefined
                  ? "memory-row__value--set"
                  : ""
              }`}
            >
              {formatValue(field, context[field.key])}
            </span>
          </div>
        ))}
      </div>
      <p className="panel__footnote">{knownCount} of {FIELD_LABELS.length} slots filled</p>
    </aside>
  );
}
