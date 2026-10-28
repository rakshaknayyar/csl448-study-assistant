import { useEffect, useState } from "react";
import { Check, Sparkles, X } from "lucide-react";
import { getJSON, postJSON } from "../api.js";
import { Segmented } from "../components/Controls.jsx";
import { PageSheet } from "../components/Pages.jsx";

const COUNTS = [5, 10, 15].map((n) => ({ value: n, label: `${n}` }));

// "Grassland, Desert & Aquatic Ecosystem" is long for a chip; keep chips short.
const shortTitle = (title) => (title.length > 26 ? `${title.slice(0, 24).trimEnd()}…` : title);

// Pick one or more lectures, get a mixed quiz, see your score per lecture.
export default function Quiz({ source }) {
  const [lectures, setLectures] = useState([]);
  const [selected, setSelected] = useState(source ? [source] : []);
  const [count, setCount] = useState(5);
  const [quiz, setQuiz] = useState(null);          // { questions, sources }
  const [answers, setAnswers] = useState({});      // question number -> picked option
  const [submitted, setSubmitted] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [openPage, setOpenPage] = useState(null);

  useEffect(() => {
    getJSON("/lectures").then((list) => {
      setLectures(list);
      if (!source && list.length) setSelected([list[0].source]);
    }).catch((e) => setError(e.message));
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // "Quiz" on a lecture card opens this page with that lecture picked.
  useEffect(() => { if (source) setSelected([source]); }, [source]);

  const byFile = Object.fromEntries(lectures.map((l) => [l.source, l]));
  const allPicked = lectures.length > 0 && selected.length === lectures.length;
  const toggle = (file) => setSelected((now) => (now.includes(file) ? now.filter((f) => f !== file) : [...now, file]));
  const toggleAll = () => setSelected(allPicked ? [] : lectures.map((l) => l.source));
  const name = (file) => (byFile[file] ? `${byFile[file].label} ${byFile[file].title}`.trim() : file);

  async function newQuiz() {
    setLoading(true);
    setError(null);
    try {
      // Keep the lectures in course order, whatever order they were tapped in.
      const sources = lectures.map((l) => l.source).filter((f) => selected.includes(f));
      const data = await postJSON("/quiz", { sources, questions: count });
      setQuiz({ questions: data.questions, sources });
      setAnswers({});
      setSubmitted(false);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const answeredCount = Object.keys(answers).length;
  const score = quiz?.questions.filter((q, i) => answers[i] === q.answer).length ?? 0;
  const perLecture = quiz && quiz.sources.map((file) => {
    const indexes = quiz.questions.map((q, i) => (q.source === file ? i : -1)).filter((i) => i >= 0);
    return { file, total: indexes.length, right: indexes.filter((i) => answers[i] === quiz.questions[i].answer).length };
  }).filter((row) => row.total > 0);
  const lectureWord = (n) => `${n} lecture${n === 1 ? "" : "s"}`;

  return (
    <div className="page">
      <h1 className="large-title">Quiz</h1>
      <p className="page-sub">Pick one or more lectures for a mixed multiple-choice quiz, written by Gemini from your notes. Every answer shows the page to revise.</p>

      <section className="quiz-setup">
        <div className="setup-head">
          <span className="group-title">Lectures</span>
          <button className="btn plain small" onClick={toggleAll}>{allPicked ? "Clear all" : "Select all"}</button>
        </div>
        <div className="pills">
          {lectures.map((l) => {
            const on = selected.includes(l.source);
            return (
              <button key={l.source} className={`pill${on ? " on" : ""}`} aria-pressed={on}
                      onClick={() => toggle(l.source)} title={`${l.label} ${l.title}`}>
                <b>{l.label}</b> {shortTitle(l.title)}
              </button>
            );
          })}
        </div>

        <div className="setup-foot">
          <div className="count">
            <span className="group-title" style={{ margin: 0 }}>Questions</span>
            <Segmented label="Number of questions" options={COUNTS} value={count} onChange={setCount} />
          </div>
          <button className="btn primary" onClick={newQuiz} disabled={!selected.length || loading}>
            <Sparkles size={16} /> {quiz ? "New quiz" : "Start quiz"}
          </button>
        </div>
        <div className="setup-note">
          {selected.length ? `${count} questions from ${lectureWord(selected.length)}` : "Pick at least one lecture"}
        </div>
      </section>

      {error && <div className="system-note" style={{ margin: "0 0 20px" }}>{error}</div>}
      {loading && (
        <div className="loading"><div className="spinner" /> Writing {count} questions from {lectureWord(selected.length)}…</div>
      )}

      {quiz && !loading && (
        <>
          <div className="results-anchor" />
          {submitted && (
            <section className="results-card">
              <div className="score">
                <span className="value">{score}/{quiz.questions.length}</span>
                <span className="label">{score === quiz.questions.length ? "All correct. Well done!" : "correct. Revise the pages under the ones you missed."}</span>
              </div>
              {perLecture.length > 1 && (
                <div className="group breakdown">
                  {perLecture.map((row) => {
                    const share = row.right / row.total;
                    const level = share === 1 ? "good" : share >= 0.5 ? "ok" : "low";
                    return (
                      <div className="row" key={row.file}>
                        <span className="label">{name(row.file)}</span>
                        <span className={`tally ${level}`}>
                          <span className="bar"><i style={{ width: `${share * 100}%` }} /></span>
                          {row.right}/{row.total}
                        </span>
                      </div>
                    );
                  })}
                </div>
              )}
            </section>
          )}

          {quiz.questions.map((q, i) => (
            <section className="question" key={i}>
              <div className="q-num">
                Question {i + 1}
                {quiz.sources.length > 1 && <span className="q-tag">{name(q.source)}</span>}
              </div>
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
                          {(state === "right" || state === "picked") && <Check size={14} strokeWidth={3} />}
                          {state === "wrong" && <X size={14} strokeWidth={3} />}
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
                    <button className="btn plain small" onClick={() => setOpenPage({ source: q.source, page: q.page })}>
                      Page {q.page} of {q.source}
                    </button>
                  )}
                </div>
              )}
            </section>
          ))}

          {!submitted ? (
            <button className="btn primary" disabled={!answeredCount}
                    onClick={() => {
                      setSubmitted(true);
                      // Bring the score into view.
                      document.querySelector(".results-anchor")?.scrollIntoView({ behavior: "smooth", block: "start" });
                    }}>
              Check answers ({answeredCount} of {quiz.questions.length})
            </button>
          ) : (
            <button className="btn" onClick={newQuiz}><Sparkles size={16} /> Another quiz on the same lectures</button>
          )}
        </>
      )}

      {openPage && <PageSheet {...openPage} onClose={() => setOpenPage(null)} />}
    </div>
  );
}
