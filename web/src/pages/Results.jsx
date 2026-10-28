import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, LabelList, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { getJSON } from "../api.js";

// One fixed colour per retriever (validated categorical palette, dark-mode steps).
const RETRIEVERS = [
  { key: "BM25", name: "Keywords (BM25)", color: "#3987e5" },
  { key: "Dense", name: "Meaning (Dense)", color: "#d95926" },
  { key: "Hybrid", name: "Hybrid", color: "#199e70" },
];
const percent = (value) => `${Math.round(value * 100)}%`;

function ChartTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="chart-tooltip">
      <div className="t">{label}</div>
      {payload.map((p) => (
        <div key={p.dataKey}><span style={{ color: p.fill }}>■</span> {RETRIEVERS.find((r) => r.key === p.dataKey).name}: <b>{percent(p.value)}</b></div>
      ))}
    </div>
  );
}

function GroupedBars({ data, xKey, yLabel }) {
  return (
    <div className="chart-card">
      <div className="legend">
        {RETRIEVERS.map((r) => <span key={r.key}><i style={{ background: r.color }} />{r.name}</span>)}
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} margin={{ top: 22, right: 16, left: 4, bottom: 4 }} barGap={3} barCategoryGap="24%">
          <CartesianGrid vertical={false} stroke="#2c2c2e" />
          <XAxis dataKey={xKey} tick={{ fill: "#8e8e93", fontSize: 12 }} axisLine={{ stroke: "#3a3a3c" }} tickLine={false} />
          <YAxis domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} tickFormatter={percent}
                 tick={{ fill: "#8e8e93", fontSize: 12 }} axisLine={false} tickLine={false} width={48}
                 label={{ value: yLabel, angle: -90, position: "insideLeft", fill: "#8e8e93", fontSize: 12, dy: 30 }} />
          <Tooltip content={<ChartTooltip />} cursor={{ fill: "rgba(255,255,255,0.04)" }} />
          {RETRIEVERS.map((r) => (
            <Bar key={r.key} dataKey={r.key} fill={r.color} radius={[4, 4, 0, 0]} maxBarSize={44}
                 isAnimationActive={false}>
              <LabelList dataKey={r.key} position="top" formatter={percent} fill="#8e8e93" fontSize={11} />
            </Bar>
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function verdictClass(verdict) {
  if (verdict.startsWith("correct")) return "ok";
  if (verdict.startsWith("missed")) return "miss";
  return "bad";
}

export default function Results() {
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    getJSON("/results").then(setResults).catch((e) => setError(e.message));
  }, []);

  if (error) return <div className="page"><h1 className="large-title">Results</h1><div className="empty-state">{error}</div></div>;
  if (!results) return <div className="page"><div className="loading"><div className="spinner" /> Loading results…</div></div>;
  if (!results.recall) {
    return (
      <div className="page">
        <h1 className="large-title">Results</h1>
        <div className="empty-state">No results yet. Run <span className="mono">venv\Scripts\python.exe evaluate.py --chunk-sizes 500 800 1000 1500 --answers</span> and reload.</div>
      </div>
    );
  }

  const { recall, chunk_sizes: chunkSizes, answers } = results;
  const measures = Object.keys(recall[0]).filter((key) => key.startsWith("Recall@"));
  const kColumn = measures.includes("Recall@3") ? "Recall@3" : measures[measures.length - 1];
  const recallData = measures.map((measure) =>
    Object.fromEntries([["measure", measure], ...recall.map((row) => [row.retriever, row[measure]])]));
  const sizeData = chunkSizes && [...new Set(chunkSizes.map((row) => row.chunk_size))].map((size) =>
    Object.fromEntries([["size", `${size} characters`],
      ...chunkSizes.filter((row) => row.chunk_size === size).map((row) => [row.retriever, row[kColumn]])]));

  const hybrid = recall.find((row) => row.retriever === "Hybrid");
  const inNotes = answers?.filter((a) => a.in_notes) || [];
  const notInNotes = answers?.filter((a) => !a.in_notes) || [];
  const share = (rows) => rows.filter((a) => a.correct).length / rows.length;

  return (
    <div className="page">
      <h1 className="large-title">Results</h1>
      <p className="page-sub">
        {answers ? `Tested on ${answers.length} questions written from the notes: ${inNotes.length} the notes can answer and ${notInNotes.length} they can't.` : "Tested on questions written from the notes."}
      </p>

      <div className="stats">
        <div className="stat">
          <div className="value">{percent(hybrid[kColumn])}</div>
          <div className="label">of questions find the right page in the top {kColumn.split("@")[1]} (hybrid search)</div>
        </div>
        {inNotes.length > 0 && (
          <div className="stat">
            <div className="value">{percent(share(inNotes))}</div>
            <div className="label">of answers judged correct by Gemini</div>
          </div>
        )}
        {notInNotes.length > 0 && (
          <div className="stat">
            <div className="value green">{percent(share(notInNotes))}</div>
            <div className="label">of off-topic questions answered with "Not found in the notes"</div>
          </div>
        )}
      </div>

      <h2 className="section-title">Finding the right page</h2>
      <p className="section-sub">Keywords (BM25) matches words, Meaning (dense embeddings) matches ideas, Hybrid combines both. Recall@k is the share of questions where the right page is among the first k results. Hover a bar for details.</p>
      <GroupedBars data={recallData} xKey="measure" yLabel="Questions" />

      <div className="table-wrap" style={{ marginTop: 14 }}>
        <table className="data">
          <thead><tr><th>Retriever</th>{measures.map((m) => <th key={m} className="num">{m}</th>)}<th className="num">MRR</th></tr></thead>
          <tbody>
            {RETRIEVERS.map((r) => {
              const row = recall.find((x) => x.retriever === r.key);
              return row && (
                <tr key={r.key}>
                  <td>{r.name}</td>
                  {measures.map((m) => <td key={m} className="num">{percent(row[m])}</td>)}
                  <td className="num">{row.MRR.toFixed(2)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {sizeData && (
        <>
          <h2 className="section-title">Does chunk size matter?</h2>
          <p className="section-sub">The same questions, searched in indexes that cut the notes into chunks of different lengths (overlap is 20% of the chunk size). Showing {kColumn}.</p>
          <GroupedBars data={sizeData} xKey="size" yLabel={kColumn} />
        </>
      )}

      {answers && (
        <>
          <h2 className="section-title">Are the answers right?</h2>
          <p className="section-sub">Each question went through the full app. Gemini compared every answer with the expected answer; off-topic questions count as correct only when the app says "Not found in the notes".</p>
          <div className="table-wrap">
            <table className="data">
              <thead><tr><th>Question</th><th>Result</th><th className="num">Best match</th></tr></thead>
              <tbody>
                {answers.map((a) => (
                  <tr key={a.question}>
                    <td>{a.question}</td>
                    <td><span className={`verdict ${verdictClass(a.verdict)}`}>{a.verdict}</span></td>
                    <td className="num">{percent(a.best_similarity)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
