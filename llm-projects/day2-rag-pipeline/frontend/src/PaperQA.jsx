import React, { useState, useRef, useEffect } from "react";

const API_BASE = "https://llm-projects-production.up.railway.app";

export default function PaperQA() {
  const [stage, setStage] = useState("upload"); // upload | ingesting | ready
  const [fileName, setFileName] = useState("");
  const [collectionName, setCollectionName] = useState("");
  const [pagesLoaded, setPagesLoaded] = useState(null);
  const [chunksCreated, setChunksCreated] = useState(null);
  const [error, setError] = useState("");
  const [dragActive, setDragActive] = useState(false);

  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);

  const fileInputRef = useRef(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages, asking]);

  async function handleFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("This reads PDFs only. Choose a .pdf file.");
      return;
    }
    setError("");
    setFileName(file.name);
    setStage("ingesting");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/ingest-upload`, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Upload failed (${res.status})`);
      }
      const data = await res.json();
      setCollectionName(data.collection_name);
      setPagesLoaded(data.pages_loaded);
      setChunksCreated(data.chunks_created);
      setStage("ready");
      setMessages([
        {
          role: "system",
          text: `Indexed. ${data.pages_loaded} pages, ${data.chunks_created} passages. Ask it something.`,
        },
      ]);
    } catch (err) {
      setError(err.message || "Could not read that file.");
      setStage("upload");
    }
  }

  function onDrop(e) {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    handleFile(file);
  }

  async function askQuestion() {
    const q = question.trim();
    if (!q || asking) return;
    setQuestion("");
    setMessages((m) => [...m, { role: "user", text: q }]);
    setAsking(true);

    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          collection_name: collectionName,
          question: q,
          top_k: 3,
        }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail || `Query failed (${res.status})`);
      }
      const data = await res.json();
      setMessages((m) => [
        ...m,
        { role: "answer", text: data.answer, sources: data.sources || [] },
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "error", text: err.message || "Could not reach the document." },
      ]);
    } finally {
      setAsking(false);
    }
  }

  function reset() {
    setStage("upload");
    setFileName("");
    setCollectionName("");
    setMessages([]);
    setError("");
  }

  const tokens = {
    paper: "#FAF8F3",
    ink: "#1A1A1A",
    inkSoft: "#5C5650",
    accent: "#8B2500",
    parchment: "#E8E3D8",
    parchmentLine: "#D8D0C0",
    stamp: "#2D5F4F",
  };

  return (
    <div
      style={{
        minHeight: "100%",
        background: tokens.paper,
        color: tokens.ink,
        fontFamily:
          "'Iowan Old Style', 'Palatino Linotype', Georgia, 'Times New Roman', serif",
        display: "flex",
        justifyContent: "center",
        padding: "32px 16px",
        boxSizing: "border-box",
      }}
    >
      <div style={{ width: "100%", maxWidth: 640 }}>
        <header style={{ marginBottom: 28, textAlign: "center" }}>
          <div
            style={{
              fontFamily: "'Courier New', monospace",
              fontSize: 11,
              letterSpacing: "0.12em",
              color: tokens.inkSoft,
              textTransform: "uppercase",
              marginBottom: 6,
            }}
          >
            Document Q&amp;A
          </div>
          <h1
            style={{
              fontSize: 28,
              fontWeight: 600,
              margin: 0,
              letterSpacing: "-0.01em",
            }}
          >
            Read it to me, then answer for it.
          </h1>
          <p
            style={{
              fontFamily: "system-ui, sans-serif",
              fontSize: 14,
              color: tokens.inkSoft,
              marginTop: 8,
              lineHeight: 1.5,
            }}
          >
            Drop in a PDF. Every answer below comes with the page it was
            found on — nothing invented.
          </p>
        </header>

        {stage === "upload" && (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
            onClick={() => fileInputRef.current?.click()}
            style={{
              border: `1.5px dashed ${dragActive ? tokens.accent : tokens.parchmentLine}`,
              background: dragActive ? "#F2ECE0" : tokens.parchment,
              borderRadius: 6,
              padding: "48px 24px",
              textAlign: "center",
              cursor: "pointer",
              transition: "border-color 120ms ease, background 120ms ease",
            }}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="application/pdf"
              style={{ display: "none" }}
              onChange={(e) => handleFile(e.target.files?.[0])}
            />
            <div
              style={{
                fontFamily: "'Courier New', monospace",
                fontSize: 32,
                color: tokens.inkSoft,
                marginBottom: 12,
              }}
            >
              ⊙
            </div>
            <div style={{ fontSize: 16, fontWeight: 600, marginBottom: 4 }}>
              Drop a PDF here, or click to choose one
            </div>
            <div
              style={{
                fontFamily: "system-ui, sans-serif",
                fontSize: 13,
                color: tokens.inkSoft,
              }}
            >
              Research papers work best — tables and figures are read too.
            </div>
            {error && (
              <div
                style={{
                  marginTop: 16,
                  fontFamily: "system-ui, sans-serif",
                  fontSize: 13,
                  color: tokens.accent,
                  fontWeight: 600,
                }}
              >
                {error}
              </div>
            )}
          </div>
        )}

        {stage === "ingesting" && (
          <div
            style={{
              border: `1px solid ${tokens.parchmentLine}`,
              background: tokens.parchment,
              borderRadius: 6,
              padding: "40px 24px",
              textAlign: "center",
            }}
          >
            <div
              style={{
                fontFamily: "'Courier New', monospace",
                fontSize: 13,
                color: tokens.inkSoft,
                marginBottom: 10,
              }}
            >
              reading {fileName}
            </div>
            <div
              style={{
                height: 2,
                background: tokens.parchmentLine,
                borderRadius: 2,
                overflow: "hidden",
                maxWidth: 220,
                margin: "0 auto",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: "40%",
                  background: tokens.accent,
                  animation: "paperqa-scan 1.1s ease-in-out infinite",
                }}
              />
            </div>
            <style>{`
              @keyframes paperqa-scan {
                0% { transform: translateX(-100%); }
                100% { transform: translateX(350%); }
              }
            `}</style>
          </div>
        )}

        {stage === "ready" && (
          <div
            style={{
              border: `1px solid ${tokens.parchmentLine}`,
              borderRadius: 6,
              background: "#FFFEFB",
              overflow: "hidden",
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "10px 16px",
                background: tokens.parchment,
                borderBottom: `1px solid ${tokens.parchmentLine}`,
                fontFamily: "system-ui, sans-serif",
                fontSize: 12.5,
                color: tokens.inkSoft,
              }}
            >
              <span>
                <strong style={{ color: tokens.ink }}>{fileName}</strong>
                {pagesLoaded != null && (
                  <span> &middot; {pagesLoaded} pages &middot; {chunksCreated} passages</span>
                )}
              </span>
              <button
                onClick={reset}
                style={{
                  border: "none",
                  background: "none",
                  color: tokens.accent,
                  fontFamily: "system-ui, sans-serif",
                  fontSize: 12.5,
                  cursor: "pointer",
                  textDecoration: "underline",
                  padding: 0,
                }}
              >
                read a different paper
              </button>
            </div>

            <div
              ref={scrollRef}
              style={{
                maxHeight: 380,
                overflowY: "auto",
                padding: "18px 16px",
                display: "flex",
                flexDirection: "column",
                gap: 14,
              }}
            >
              {messages.map((m, i) => (
                <MessageBlock key={i} m={m} tokens={tokens} />
              ))}
              {asking && (
                <div
                  style={{
                    fontFamily: "system-ui, sans-serif",
                    fontSize: 13,
                    color: tokens.inkSoft,
                    fontStyle: "italic",
                  }}
                >
                  searching the document…
                </div>
              )}
            </div>

            <div
              style={{
                display: "flex",
                gap: 8,
                padding: "12px 16px",
                borderTop: `1px solid ${tokens.parchmentLine}`,
                background: tokens.parchment,
              }}
            >
              <input
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") askQuestion();
                }}
                placeholder="Ask something about this document…"
                style={{
                  flex: 1,
                  border: `1px solid ${tokens.parchmentLine}`,
                  borderRadius: 4,
                  padding: "9px 12px",
                  fontFamily: "system-ui, sans-serif",
                  fontSize: 14,
                  background: "#FFFEFB",
                  color: tokens.ink,
                  outline: "none",
                }}
              />
              <button
                onClick={askQuestion}
                disabled={asking || !question.trim()}
                style={{
                  border: "none",
                  borderRadius: 4,
                  padding: "9px 18px",
                  fontFamily: "system-ui, sans-serif",
                  fontSize: 14,
                  fontWeight: 600,
                  background: asking || !question.trim() ? "#C9C2B4" : tokens.accent,
                  color: "#FFFEFB",
                  cursor: asking || !question.trim() ? "default" : "pointer",
                }}
              >
                Ask
              </button>
            </div>
          </div>
        )}

        <footer
          style={{
            marginTop: 20,
            textAlign: "center",
            fontFamily: "system-ui, sans-serif",
            fontSize: 11.5,
            color: tokens.inkSoft,
          }}
        >
          RAG pipeline · LangChain · Qdrant · Groq/Llama 3.1 · FastAPI on Railway
        </footer>
      </div>
    </div>
  );
}

function MessageBlock({ m, tokens }) {
  if (m.role === "system") {
    return (
      <div
        style={{
          fontFamily: "system-ui, sans-serif",
          fontSize: 12.5,
          color: tokens.inkSoft,
          fontStyle: "italic",
        }}
      >
        {m.text}
      </div>
    );
  }
  if (m.role === "user") {
    return (
      <div
        style={{
          fontFamily: "system-ui, sans-serif",
          fontSize: 14.5,
          fontWeight: 600,
          color: tokens.ink,
        }}
      >
        {m.text}
      </div>
    );
  }
  if (m.role === "error") {
    return (
      <div
        style={{
          fontFamily: "system-ui, sans-serif",
          fontSize: 13,
          color: tokens.accent,
        }}
      >
        {m.text}
      </div>
    );
  }
  // answer
  return (
    <div style={{ display: "flex", gap: 10 }}>
      <div
        style={{
          borderLeft: `2px solid ${tokens.accent}`,
          paddingLeft: 12,
          flex: 1,
          fontFamily: "system-ui, sans-serif",
          fontSize: 14.5,
          lineHeight: 1.55,
          color: tokens.ink,
        }}
      >
        {m.text}
      </div>
      {m.sources && m.sources.length > 0 && (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 4,
            flexShrink: 0,
          }}
        >
          {[...new Set(m.sources)].map((s, i) => (
            <span
              key={i}
              style={{
                fontFamily: "'Courier New', monospace",
                fontSize: 10.5,
                color: tokens.stamp,
                border: `1px solid ${tokens.stamp}`,
                borderRadius: 3,
                padding: "1px 6px",
                whiteSpace: "nowrap",
                textTransform: "uppercase",
                letterSpacing: "0.03em",
              }}
            >
              {s}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
