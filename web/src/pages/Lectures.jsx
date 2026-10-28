import { useEffect, useState } from "react";
import { ListChecks, MessageCircle, NotebookText } from "lucide-react";
import { getJSON, postJSON } from "../api.js";
import Markdown from "../components/Markdown.jsx";
import Sheet from "../components/Sheet.jsx";

function SummarySheet({ lecture, onClose }) {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    postJSON(`/lectures/${encodeURIComponent(lecture.source)}/summary`)
      .then((data) => setSummary(data.summary))
      .catch((e) => setError(e.message));
  }, [lecture.source]);

  return (
    <Sheet title={`${lecture.label} ${lecture.title}`.trim()} subtitle="Revision summary" onClose={onClose}>
      {error && <div className="system-note" style={{ margin: 0 }}>{error}</div>}
      {!summary && !error && <div className="loading"><div className="spinner" /> Summarizing this lecture…</div>}
      {summary && (
        <>
          <div className="summary-text"><Markdown>{summary}</Markdown></div>
          <div className="footnote">Written by Gemini from {lecture.source} only, and saved so it opens instantly next time.</div>
        </>
      )}
    </Sheet>
  );
}

export default function Lectures({ onAsk, onQuiz }) {
  const [lectures, setLectures] = useState(null);
  const [error, setError] = useState(null);
  const [summaryFor, setSummaryFor] = useState(null);

  useEffect(() => {
    getJSON("/lectures").then(setLectures).catch((e) => setError(e.message));
  }, []);

  const pages = lectures?.reduce((total, lecture) => total + lecture.pages, 0);

  return (
    <div className="page">
      <h1 className="large-title">Lectures</h1>
      <p className="page-sub">
        {lectures ? `${lectures.length} lectures, ${pages} pages. ` : ""}
        The key terms on each card are picked with TF-IDF: words that are frequent in that lecture but rare in the others.
      </p>

      {error && <div className="empty-state">{error}</div>}
      {!lectures && !error && <div className="loading"><div className="spinner" /> Loading your lectures…</div>}

      <div className="cards">
        {lectures?.map((lecture) => (
          <article className="card" key={lecture.source}>
            <span className="label">{lecture.label || "Notes"}</span>
            <h3>{lecture.title}</h3>
            <div className="info">{lecture.pages} page{lecture.pages === 1 ? "" : "s"} · {lecture.source}</div>
            <div className="terms">
              {lecture.key_terms.map((term) => <span key={term}>{term}</span>)}
            </div>
            <div className="actions">
              <button className="btn small" onClick={() => setSummaryFor(lecture)}><NotebookText size={15} /> Summary</button>
              <button className="btn small" onClick={() => onQuiz(lecture.source)}><ListChecks size={15} /> Quiz</button>
              <button className="btn small" onClick={() => onAsk(`Give me an overview of ${lecture.title}.`)}>
                <MessageCircle size={15} /> Ask
              </button>
            </div>
          </article>
        ))}
      </div>

      {summaryFor && <SummarySheet lecture={summaryFor} onClose={() => setSummaryFor(null)} />}
    </div>
  );
}
