import axios from "axios";

// In dev, Vite proxies /api -> http://localhost:5000 (see vite.config.js),
// so relative paths work whether the frontend is opened at
// http://localhost:5173 or the API URL is overridden via env var.
const API_BASE = import.meta.env.VITE_API_URL || "/api";

const client = axios.create({
  baseURL: API_BASE,
  timeout: 15000,
});

export async function sendMessage(message, conversationId) {
  const { data } = await client.post("/chat", {
    message,
    conversation_id: conversationId,
  });
  return data;
}

export async function getConversations() {
  const { data } = await client.get("/conversations");
  return data.conversations || [];
}

export async function getConversation(conversationId) {
  const { data } = await client.get(`/conversations/${conversationId}`);
  return data;
}

export async function deleteConversation(conversationId) {
  const { data } = await client.delete(`/conversations/${conversationId}`);
  return data;
}

export async function getContext(conversationId) {
  const { data } = await client.get(`/context/${conversationId}`);
  return data.context;
}

export async function getAnalytics() {
  const { data } = await client.get("/analytics");
  return data;
}

export async function retrainModel() {
  const { data } = await client.post("/train");
  return data;
}

export async function checkHealth() {
  const { data } = await client.get("/health");
  return data;
}

export default client;
