import { pageImageUrl } from "../api.js";
import Sheet from "./Sheet.jsx";

const where = (page) => (page ? `page ${page}` : "overview");

// One page of a PDF, full size.
export function PageSheet({ source, page, onClose }) {
  return (
    <Sheet title={source} subtitle={`Page ${page}`} onClose={onClose} wide>
      <img className="page-image" src={pageImageUrl(source, page)} alt={`${source}, page ${page}`} />
    </Sheet>
  );
}

// Every passage that was retrieved for an answer, with a picture of its page.
export function PassagesSheet({ message, showPrompt, onOpenPage, onClose }) {
  const cited = new Set(message.cited_ids || []);
  return (
    <Sheet title="Sources" subtitle={`${message.chunks.length} passages from your notes, closest first`}
           onClose={onClose} wide>
      {message.chunks.map((chunk) => (
        <div className="passage" key={chunk.id}>
          {chunk.page ? (
            <button className="thumb" onClick={() => onOpenPage(chunk.source, chunk.page)}
                    aria-label={`Open ${chunk.source} page ${chunk.page}`}>
              <img src={pageImageUrl(chunk.source, chunk.page)} alt="" loading="lazy" />
            </button>
          ) : <div />}
          <div>
            <div className="where">
              {chunk.source}, {where(chunk.page)}
              <span className="match">{Math.round(chunk.similarity * 100)}% match</span>
              {cited.has(chunk.id) && <span className="tag">Used in the answer</span>}
            </div>
            <div className="text">{chunk.text}</div>
          </div>
        </div>
      ))}
      {showPrompt && message.prompt && (
        <>
          <h4 className="group-title" style={{ marginTop: 20 }}>Prompt sent to Gemini</h4>
          <div className="prompt-box">{message.prompt}</div>
        </>
      )}
    </Sheet>
  );
}
