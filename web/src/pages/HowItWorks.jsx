import { BookMarked, Search, ShieldCheck, Sparkles } from "lucide-react";

const STEPS = [
  { icon: Search, name: "Find", text: "Your question is matched against every chunk of your notes by keywords (BM25), by meaning (sentence embeddings in a FAISS index), or both." },
  { icon: ShieldCheck, name: "Check", text: "If even the closest passage is a weak match, the answer is \"Not found in the notes\" and Gemini is never asked." },
  { icon: Sparkles, name: "Answer", text: "The closest passages go into a prompt that tells Gemini to answer only from them. Earlier questions are included, so follow-ups work." },
  { icon: BookMarked, name: "Cite", text: "The answer shows the file and page of each passage it used, a confidence level, and a picture of the real page to check it." },
];

const SYLLABUS = [
  ["Unit 2", "Text preprocessing (lowercase, tokenize, remove stop-words) and BM25 keyword retrieval", ["retriever.py"]],
  ["Unit 3", "Bag of words and term counts (BM25) compared with word embeddings; TF-IDF key terms on the Lectures page", ["retriever.py", "study.py"]],
  ["Unit 4", "Transformer sentence embeddings, plus chunk size, overlap and top-k experiments", ["ingest.py", "evaluate.py"]],
  ["Unit 5", "RAG, a FAISS vector database, prompt engineering and Gemini; plain Gemini vs RAG; generated summaries and quizzes", ["rag.py", "study.py"]],
  ["CO5", "Apply and evaluate: Recall@k, MRR, chunk sizes and Gemini-judged answers", ["evaluate.py"]],
];

export default function HowItWorks({ info }) {
  const index = info?.index;
  const details = info && [
    ["Notes", info.notes.join(", ")],
    ["Pages with text", info.pages],
    ["Chunks", `${index.num_chunks} (${index.chunk_size} characters, ${index.chunk_overlap} overlap)`],
    ["Embeddings", `${index.embedding_model} (${index.vector_dimension} dimensions)`],
    ["Vector store", "FAISS IndexFlatIP on normalized vectors (cosine similarity)"],
    ["Language model", info.model],
    ["Not-found cut-off", `best match below ${Math.round(info.threshold * 100)}%`],
  ];

  return (
    <div className="page">
      <h1 className="large-title">How it works</h1>
      <p className="page-sub">Four steps turn your question into an answer you can check against your notes.</p>

      <div className="steps">
        {STEPS.map(({ icon: Icon, name, text }, i) => (
          <div className="step" key={name}>
            <div className="icon"><Icon size={20} /></div>
            <div className="n">Step {i + 1}</div>
            <h3>{name}</h3>
            <p>{text}</p>
          </div>
        ))}
      </div>

      <h2 className="section-title">Where the syllabus shows up</h2>
      <div className="table-wrap">
        <table className="data">
          <thead><tr><th>Unit</th><th>Topic</th><th>Code</th></tr></thead>
          <tbody>
            {SYLLABUS.map(([unit, topic, files]) => (
              <tr key={unit}>
                <td style={{ fontWeight: 600, whiteSpace: "nowrap" }}>{unit}</td>
                <td style={{ color: "var(--label-2)" }}>{topic}</td>
                <td>{files.map((f) => <span className="mono" key={f}>{f}</span>)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {details && (
        <>
          <h2 className="section-title">About this index</h2>
          <div className="group" style={{ background: "var(--bg-2)" }}>
            {details.map(([name, value]) => (
              <div className="row" key={name}>
                <span className="label" style={{ fontSize: 15 }}>{name}</span>
                <span style={{ color: "var(--label-2)", fontSize: 15, textAlign: "right" }}>{value}</span>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
