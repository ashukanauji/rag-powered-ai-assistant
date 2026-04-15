'use client';

import { FormEvent, useMemo, useState } from 'react';

type Message = {
  role: 'user' | 'assistant';
  text: string;
  sources?: string[];
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [question, setQuestion] = useState('');
  const [loading, setLoading] = useState(false);
  const [apiKey, setApiKey] = useState('');
  const [uploading, setUploading] = useState(false);

  const headers = useMemo(() => ({ 'Content-Type': 'application/json', 'x-api-key': apiKey }), [apiKey]);

  async function handleAsk(e: FormEvent) {
    e.preventDefault();
    if (!question.trim()) return;

    const q = question.trim();
    setMessages((prev) => [...prev, { role: 'user', text: q }]);
    setQuestion('');
    setLoading(true);

    const response = await fetch(`${API_URL}/query`, {
      method: 'POST',
      headers,
      body: JSON.stringify({ question: q, top_k: 5, stream: false }),
    });
    const data = await response.json();

    setMessages((prev) => [...prev, { role: 'assistant', text: data.answer, sources: data.sources }]);
    setLoading(false);
  }

  async function handleUpload(file: File) {
    setUploading(true);
    const formData = new FormData();
    formData.append('file', file);
    await fetch(`${API_URL}/upload`, {
      method: 'POST',
      headers: { 'x-api-key': apiKey },
      body: formData,
    });
    setUploading(false);
  }

  return (
    <div className="container">
      <h1>Production AI Knowledge Assistant</h1>
      <p>Upload your docs and ask grounded questions.</p>

      <div className="card">
        <label>API Key (optional in dev): </label>
        <input type="text" value={apiKey} onChange={(e) => setApiKey(e.target.value)} placeholder="x-api-key" />
      </div>

      <div className="card">
        <label>Upload PDF/TXT: </label>
        <input
          type="file"
          accept=".pdf,.txt"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleUpload(file);
          }}
        />
        {uploading && <p className="small">Uploading and indexing...</p>}
      </div>

      <div className="card chat-box">
        {messages.map((m, idx) => (
          <div key={idx} className={m.role === 'user' ? 'msg-user' : 'msg-assistant'}>
            {m.text}
            {m.sources && m.sources.length > 0 && <div className="small">Sources: {m.sources.join(', ')}</div>}
          </div>
        ))}
        {loading && <div className="msg-assistant">Thinking...</div>}
      </div>

      <form className="controls" onSubmit={handleAsk}>
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Ask a question about your documents..."
        />
        <button type="submit" disabled={loading}>Send</button>
      </form>
    </div>
  );
}
