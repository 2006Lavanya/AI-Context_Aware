import React, { useEffect, useState, useCallback } from "react";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell, Legend,
} from "recharts";
import { getAnalytics } from "../api.js";

const SENTIMENT_COLORS = {
  Positive: "#3DDC97",
  Negative: "#FF6B6B",
  Neutral: "#8891A5",
};

const BAR_COLOR = "#00C2A8";
const ENTITY_COLOR = "#8B7FD1";

export default function Dashboard({ refreshKey }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await getAnalytics();
      setData(result);
    } catch (e) {
      setError("Could not load analytics. Is the backend running?");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load, refreshKey]);

  if (loading && !data) {
    return <div className="dashboard-status">Loading analytics...</div>;
  }
  if (error) {
    return <div className="dashboard-status dashboard-status--error">{error}</div>;
  }
  if (!data) return null;

  const intentData = Object.entries(data.intent_distribution || {}).map(([name, value]) => ({ name, value }));
  const entityData = Object.entries(data.entity_distribution || {}).map(([name, value]) => ({ name, value }));
  const sentimentData = Object.entries(data.sentiment_distribution || {}).map(([name, value]) => ({ name, value }));

  return (
    <div className="dashboard">
      <div className="dashboard-header">
        <h1 className="dashboard-header__title">NLP Analytics Dashboard</h1>
        <button className="ghost-btn" onClick={load}>Refresh</button>
      </div>

      <div className="stat-grid">
        <StatCard label="Total Conversations" value={data.total_conversations} />
        <StatCard label="Total Messages" value={data.total_messages} />
        <StatCard label="Avg Msgs / Conversation" value={data.avg_messages_per_conversation} />
        <StatCard label="Data Store" value={data.data_store} small />
      </div>

      <div className="chart-grid">
        <div className="chart-card">
          <h2 className="chart-card__title">Intent Distribution</h2>
          {intentData.length === 0 ? (
            <p className="panel__empty">No data yet - start a conversation.</p>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={intentData} margin={{ top: 8, right: 8, left: -20, bottom: 40 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232B3D" />
                <XAxis dataKey="name" angle={-35} textAnchor="end" interval={0} tick={{ fontSize: 11, fill: "#8891A5" }} />
                <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: "#8891A5" }} />
                <Tooltip contentStyle={{ background: "#161D2E", border: "1px solid #232B3D", borderRadius: 8 }} />
                <Bar dataKey="value" fill={BAR_COLOR} radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="chart-card">
          <h2 className="chart-card__title">Sentiment Distribution</h2>
          {sentimentData.every((d) => d.value === 0) ? (
            <p className="panel__empty">No data yet - start a conversation.</p>
          ) : (
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie data={sentimentData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label>
                  {sentimentData.map((entry) => (
                    <Cell key={entry.name} fill={SENTIMENT_COLORS[entry.name] || "#8891A5"} />
                  ))}
                </Pie>
                <Legend />
                <Tooltip contentStyle={{ background: "#161D2E", border: "1px solid #232B3D", borderRadius: 8 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="chart-card chart-card--wide">
          <h2 className="chart-card__title">Most Common Entities</h2>
          {entityData.length === 0 ? (
            <p className="panel__empty">No data yet - start a conversation.</p>
          ) : (
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={entityData} layout="vertical" margin={{ top: 8, right: 24, left: 16, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#232B3D" />
                <XAxis type="number" allowDecimals={false} tick={{ fontSize: 11, fill: "#8891A5" }} />
                <YAxis type="category" dataKey="name" width={100} tick={{ fontSize: 11, fill: "#8891A5" }} />
                <Tooltip contentStyle={{ background: "#161D2E", border: "1px solid #232B3D", borderRadius: 8 }} />
                <Bar dataKey="value" fill={ENTITY_COLOR} radius={[0, 4, 4, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}

function StatCard({ label, value, small }) {
  return (
    <div className="stat-card">
      <span className="stat-card__label">{label}</span>
      <span className={`stat-card__value ${small ? "stat-card__value--small" : ""}`}>{value}</span>
    </div>
  );
}
