import { useEffect, useState } from "react";
import { Check, Sparkles, X } from "lucide-react";
import { getJSON, postJSON } from "../api.js";
import { PageSheet } from "../components/Pages.jsx";

export default function Quiz({ source, onSourceChange }) {
  const [lectures, setLectures] = useState([]);
  const [quiz, setQuiz] = useState(null);          // { source, questions }
  const [answers, setAnswers] = useState({});      // question number -> picked option
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [openPage, setOpenPage] = useState(null);

  useEffect(() => {
    getJSON("/lectures").then((list) => {
      setLectures(list);
      if (!source && list.length) onSourceChange(list[0].source);
    }).catch((e) => setError(e.message));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const lecture = lectures.find((l) => l.source === source);

  async function newQuiz() {
    setLoading(true);
    setError(null);
    try {
      const data = await postJSON("/quiz", { source });
      setQuiz({ source, questions: data.questions });
      setAnswers({});
      setSubmitted(false);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const current = quiz && quiz.source === source ? quiz : null;
  const answeredCount = Object.keys(answers).length;
  const score = current?.questions.filter((q, i) => answers[i] === q.answer).length ?? 0;

  return (
    <div className="page">
      <h1 className="large-title">Quiz</h1>
      <p className="page-sub">Five multiple-choice questions from one lecture, written by Gemini from your notes. Each answer tells you which page to revise.</p>

      <div className="toolbar">
        <select className="select" value={source || ""} aria-label="Lecture"
                onChange={(event) => onSourceChange(event.target.value)}>
          {lectures.map((l) => <option key={l.source} value={l.source}>{`${l.label} ${l.title}`.trim()}</option>)}
        </select>
        <button className="btn primary" onClick={newQuiz} disabled={!source || loading}>
          <Sparkles size={16} /> {current ? "New quiz" : "Start quiz"}
        </button>
      </div>

      {error && <div className="system-note" style={{ marginBottom: 20 }}>{error}</div>}
      {loading && <div className="loading"><div className="spinner" /> Writing 5 questions on {lecture?.title}…</div>}

      {!current && !loading && (
        <div className="empty-state">Pick a lecture and press <b>Start quiz</b>.</div>
      )}

      {current && !loading && (
        <>
          {submitted && (
            <div className="score">
              <span className="value">{score}/{current.questions.length}</span>
              <span className="label">correct{score < current.questions.length ? ". Revise the pages under the ones you missed." : ". Well done!"}</span>
            </div>
          )}

          {current.questions.map((q, i) => (
            <section className="question" key={i}>
              <div className="q-num">Question {i + 1}</div>
              <div className="q-text">{q.question}</div>
              <div className="group">
                {q.options.map((option, j) => {
                  const picked = answers[i] === j;
                  let state = picked ? "picked" : "";
                  if (submitted && j === q.answer) state = "right";
                  else if (submitted && picked) state = "wrong";
                  return (
                    <div className="row" key={j} style={{ padding: 0 }}>
                      <button className={`option ${state}`} disabled={submitted}
                              onClick={() => setAnswers((previous) => ({ ...previous, [i]: j }))} aria-pressed={picked}>
                        <span className="mark">
                          {state === "right" && <Check size={14} strokeWidth={3} />}
                          {state === "wrong" && <X size={14} strokeWidth={3} />}
                          {state === "picked" && <Check size={14} strokeWidth={3} />}
                        </span>
                        {option}
                      </button>
                    </div>
                  );
                })}
              </div>
              {submitted && (
                <div className="explain">
                  <b>{answers[i] === q.answer ? "Correct." : answers[i] === undefined ? "Skipped." : "Not quite."}</b>{" "}
                  {q.explanation}{" "}
                  {q.page && (
                    <button className="btn plain small" onClick={() => setOpenPage({ source: current.source, page: q.page })}>
                      Page {q.page} of {current.source}
                    </button>
                  )}
                </div>
              )}
            </section>
          ))}

          {!submitted ? (
            <button className="btn primary" onClick={() => setSubmitted(true)} disabled={!answeredCount}>
              Check answers ({answeredCount} of {current.questions.length})
            </button>
          ) : (
            <button className="btn" onClick={newQuiz}><Sparkles size={16} /> Try another quiz</button>
          )}
        </>
      )}

      {openPage && <PageSheet {...openPage} onClose={() => setOpenPage(null)} />}
    </div>
  );
}
