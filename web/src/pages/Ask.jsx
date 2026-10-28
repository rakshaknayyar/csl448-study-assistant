import { useEffect, useRef, useState } from "react";
import { ArrowUp, BookOpen, ChevronRight, Menu, SlidersHorizontal, SquarePen } from "lucide-react";
import { askStream } from "../api.js";
import { Segmented, Switch } from "../components/Controls.jsx";
import Markdown from "../components/Markdown.jsx";
import { PageSheet, PassagesSheet } from "../components/Pages.jsx";
import Sheet from "../components/Sheet.jsx";

const SETTINGS_KEY = "study-assistant.settings.v1";
const MODES = [
  { value: "hybrid", label: "Hybrid" },
  { value: "dense", label: "Meaning" },
  { value: "bm25", label: "Keywords" },
];
const MODE_NAMES = { hybrid: "Hybrid", dense: "Meaning", bm25: "Keywords" };

function loadSettings(defaultK) {
  const defaults = { mode: "hybrid", k: defaultK, compare: false, showPrompt: false };
  try {
    return { ...defaults, ...JSON.parse(localStorage.getItem(SETTINGS_KEY)) };
  } catch {
    return defaults;
  }
}

const newId = () => crypto.randomUUID();
const shortName = (source) => source.replace(/\.pdf$/i, "");

// While Gemini is still typing: hide the "Sources: [1]" line and a "Not found" reply.
function visibleWhileStreaming(text) {
  if (text.trim().toLowerCase().startsWith("not found")) return "";
  return text.split(/\n\s*\**\s*Sources?\s*\**\s*:/i)[0];
}

// Finished question/answer pairs, oldest first, for follow-up questions.
function finishedTurns(messages) {
  const turns = [];
  for (let i = 0; i + 1 < messages.length; i++) {
    const [asked, answered] = [messages[i], messages[i + 1]];
    if (asked.role === "user" && answered.role === "assistant" && answered.status === "done") {
      turns.push({ question: asked.text, answer: answered.text });
    }
  }
  return turns;
}

function timeStamp(time) {
  const date = new Date(time);
  const today = new Date().toDateString() === date.toDateString();
  const day = today ? "Today" : date.toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" });
  return [day, date.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" })];
}

function chatAsMarkdown(chat) {
  const lines = [`# ${chat.title}`, "", "_Study Assistant revision notes_", ""];
  for (const message of chat.messages) {
    if (message.role === "user") lines.push(`## ${message.text}`, "");
    else if (message.status === "done") {
      lines.push(message.text, "");
      if (message.sources?.length) {
        lines.push(`*Sources: ${message.sources.map((s) => `${s.source} ${s.page ? `page ${s.page}` : "overview"}`).join(", ")}*`, "");
      }
    }
  }
  return lines.join("\n");
}

function download(filename, text) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/markdown" }));
  const link = Object.assign(document.createElement("a"), { href: url, download: filename });
  link.click();
  URL.revokeObjectURL(url);
}

// ------------------------------------------------------------------- pieces

function Typing({ tail = true }) {
  return (
    <div className={`bubble in${tail ? " tail" : ""}`} aria-label="Typing">
      <div className="typing" style={{ padding: 0 }}><span /><span /><span /></div>
    </div>
  );
}

function Reply({ message, onOpenPage, onOpenPassages }) {
  const compare = message.compare;
  const hasPlain = compare && (message.plainPending || message.plain);

  let main;
  if (message.status === "typing") {
    main = <Typing tail={!hasPlain} />;
  } else if (message.status === "streaming") {
    const text = visibleWhileStreaming(message.text);
    main = text
      ? <div className={`bubble in${hasPlain ? "" : " tail"} cursor`}><Markdown>{text}</Markdown></div>
      : <Typing tail={!hasPlain} />;
  } else if (message.status === "error") {
    main = <div className="system-note">{message.error}</div>;
  } else if (message.found) {
    main = <div className={`bubble in${hasPlain ? "" : " tail"}`}><Markdown>{message.text}</Markdown></div>;
  } else {
    const why = message.reason === "low_similarity"
      ? `The closest passage is only a ${Math.round(message.best_similarity * 100)}% match, so Gemini wasn't asked.`
      : `Gemini read the ${message.chunks?.length || 0} closest passages and the answer isn't in them.`;
    main = (
      <div className={`bubble in${hasPlain ? "" : " tail"}`}>
        <div className="nf-title">Not found in the notes</div>
        <div className="muted">Your notes don't cover this, so I won't guess. {why}</div>
      </div>
    );
  }

  const done = message.status === "done";
  return (
    <>
      {compare && <div className="sender">Study Assistant · from your notes</div>}
      {main}

      {done && (
        <div className="msg-meta">
          {message.confidence && <span className={`confidence ${message.confidence}`}>{message.confidence} confidence</span>}
          {message.sources?.map((s) =>
            s.page ? (
              <button key={`${s.source}-${s.page}`} className="chip" onClick={() => onOpenPage(s.source, s.page)}
                      title={`Open ${s.source}, page ${s.page}`}>
                {shortName(s.source)} · p.{s.page}
              </button>
            ) : (
              <span key={`${s.source}-overview`} className="chip">Notes overview</span>
            ))}
          {message.chunks?.length > 0 && (
            <button className="chip link" onClick={() => onOpenPassages(message)}>
              {message.found ? `${message.chunks.length} passages` : "Closest passages"}
            </button>
          )}
        </div>
      )}

      {hasPlain && (
        <>
          <div className="sender plain">Plain Gemini · without your notes</div>
          {message.plainPending
            ? <Typing />
            : <div className="bubble in tail plain-gemini"><Markdown>{message.plain}</Markdown></div>}
          {!message.plainPending && (
            <div className="plain-note">Not checked against your notes: there's no page to verify it, and it may include things your course never taught.</div>
          )}
        </>
      )}
      {message.plainError && <div className="system-note">{message.plainError}</div>}
    </>
  );
}

// --------------------------------------------------------------------- page

export default function Ask({ info, chat, createChat, updateChat, onNewChat, onMenu, pendingQuestion, clearPending }) {
  const [draft, setDraft] = useState("");
  const [busy, setBusy] = useState(false);
  const [settings, setSettings] = useState(() => loadSettings(info?.default_top_k ?? 5));
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [passagesFor, setPassagesFor] = useState(null);
  const [openPage, setOpenPage] = useState(null);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);
  const messages = chat?.messages || [];

  useEffect(() => {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(settings)); } catch { /* ignore */ }
  }, [settings]);

  // Keep the newest message in view, unless you have scrolled up to read.
  // stickToBottom is decided when YOU scroll, so a big new bubble can't break it.
  const stickToBottom = useRef(true);
  const onScroll = () => {
    const el = scrollRef.current;
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
  };
  useEffect(() => {
    const el = scrollRef.current;
    if (el && stickToBottom.current) el.scrollTop = el.scrollHeight;
  }, [messages]);
  useEffect(() => { stickToBottom.current = true; }, [chat?.id]);

  // Grow the text box with what you type.
  useEffect(() => {
    const el = inputRef.current;
    if (el) { el.style.height = "auto"; el.style.height = `${Math.min(el.scrollHeight, 160)}px`; }
  }, [draft]);

  async function send(text) {
    const question = text.trim();
    if (!question || busy) return;
    setDraft("");
    setBusy(true);

    const history = finishedTurns(messages);
    const chatId = chat?.id ?? createChat(question.length > 46 ? `${question.slice(0, 44)}…` : question);
    const reply = { id: newId(), role: "assistant", status: "typing", text: "", mode: settings.mode,
                    k: settings.k, compare: settings.compare };
    updateChat(chatId, (c) => ({
      ...c, updated: Date.now(),
      messages: [...c.messages, { id: newId(), role: "user", text: question, time: Date.now() }, reply],
    }));
    const patch = (change) => updateChat(chatId, (c) => ({
      ...c,
      messages: c.messages.map((m) => (m.id === reply.id ? { ...m, ...(typeof change === "function" ? change(m) : change) } : m)),
    }));
    stickToBottom.current = true;           // sending a question always jumps to the newest message

    try {
      await askStream({ question, mode: settings.mode, k: settings.k, history, compare: settings.compare }, (event) => {
        const { type, ...data } = event;
        if (type === "retrieved") patch(data);
        else if (type === "token") patch((m) => ({ status: "streaming", text: m.text + data.text }));
        else if (type === "answer") patch({ ...data, text: data.answer, status: "done", plainPending: settings.compare });
        else if (type === "plain") patch({ plain: data.text, plainPending: false });
        else if (type === "error") {
          patch((m) => (m.status === "done"
            ? { plainPending: false, plainError: data.message }
            : { status: "error", error: data.message }));
        }
      });
    } catch (error) {
      patch({ status: "error", error: error.message || "Couldn't reach the server." });
    } finally {
      patch((m) => (["done", "error"].includes(m.status)
        ? { plainPending: false }
        : { status: "error", error: "The answer stopped early. Try again." }));
      setBusy(false);
    }
  }

  useEffect(() => {
    if (pendingQuestion && !busy) {
      clearPending();
      send(pendingQuestion);
    }
  }, [pendingQuestion]); // eslint-disable-line react-hooks/exhaustive-deps

  const lastUserIndex = messages.map((m) => m.role).lastIndexOf("user");
  const [day, time] = messages.length ? timeStamp(messages[0].time) : [];
  const lectureCount = info?.lectures ?? "";

  return (
    <div className="chat">
      <header className="chat-header">
        <div className="chat-header-inner">
          <button className="menu-button" onClick={onMenu} aria-label="Open menu"><Menu size={22} /></button>
          <div className="contact">
            <div className="avatar small"><BookOpen size={20} /></div>
            <div className="name">Study Assistant <ChevronRight size={14} /></div>
          </div>
          <button className="icon-button" onClick={onNewChat} aria-label="New chat" title="New chat"
                  style={{ justifySelf: "end" }}>
            <SquarePen size={19} />
          </button>
        </div>
      </header>

      <div className={`messages${messages.length === 0 ? " empty" : ""}`} ref={scrollRef} onScroll={onScroll}>
        {messages.length === 0 ? (
          <div className="welcome">
            <div className="avatar large"><BookOpen size={30} /></div>
            <h2>What do you want to revise?</h2>
            <p>Ask anything from your {lectureCount} lectures. Every answer shows the page it came from.</p>
            <div className="suggestions">
              {(info?.suggestions || []).map((s, i) => (
                <button key={s} onClick={() => send(s)}>
                  <span className="s-text">{s}</span>
                  <span className="s-hint">{i === (info.suggestions.length - 1) ? "Not in your notes: watch it refuse" : "From your lectures"}</span>
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="thread">
            <div className="stamp"><b>{day}</b> {time}</div>
            {messages.map((m, i) =>
              m.role === "user" ? (
                <div key={m.id} style={{ display: "contents" }}>
                  <div className="bubble out tail">{m.text}</div>
                  {i === lastUserIndex && <div className="delivered">Delivered</div>}
                </div>
              ) : (
                <div key={m.id} style={{ display: "contents" }}>
                  <div style={{ height: 4 }} />
                  <Reply message={m} onOpenPage={(source, page) => setOpenPage({ source, page })}
                         onOpenPassages={setPassagesFor} />
                </div>
              ))}
          </div>
        )}
      </div>

      <div className="composer">
        <div className="composer-row">
          <div className="field">
            <button className="field-button" onClick={() => setSettingsOpen(true)} aria-label="Search settings"
                    title="Search settings">
              <SlidersHorizontal size={18} />
            </button>
            <textarea
              ref={inputRef} rows={1} value={draft} placeholder="Ask your notes"
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); send(draft); }
              }}
              aria-label="Your question"
            />
            <button className="send" onClick={() => send(draft)} disabled={!draft.trim() || busy} aria-label="Send">
              <ArrowUp size={18} strokeWidth={2.6} />
            </button>
          </div>
        </div>
        <div className="composer-hint">
          {MODE_NAMES[settings.mode]} search · {settings.k} passages{settings.compare ? " · comparing with plain Gemini" : ""}
        </div>
      </div>

      {settingsOpen && (
        <Sheet title="Search settings" onClose={() => setSettingsOpen(false)}>
          <div className="group-title">How to search your notes</div>
          <div className="group">
            <div className="row">
              <div>
                <div className="label">Search by</div>
                <div className="hint">Keywords = BM25, Meaning = embeddings + FAISS</div>
              </div>
              <Segmented label="Search by" options={MODES} value={settings.mode}
                         onChange={(mode) => setSettings({ ...settings, mode })} />
            </div>
            <div className="row">
              <div>
                <div className="label">Passages to read</div>
                <div className="hint">How many passages Gemini reads (top-k)</div>
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                <input type="range" min={1} max={10} value={settings.k} aria-label="Passages to read"
                       onChange={(event) => setSettings({ ...settings, k: Number(event.target.value) })} />
                <span className="stepper-value">{settings.k}</span>
              </div>
            </div>
          </div>

          <div className="group-title">Demo</div>
          <div className="group">
            <div className="row">
              <div>
                <div className="label">Compare with plain Gemini</div>
                <div className="hint">Also ask Gemini without your notes, to see what RAG prevents. Uses one more request.</div>
              </div>
              <Switch label="Compare with plain Gemini" checked={settings.compare}
                      onChange={(compare) => setSettings({ ...settings, compare })} />
            </div>
            <div className="row">
              <div>
                <div className="label">Show the prompt</div>
                <div className="hint">See exactly what was sent to Gemini, under each answer's passages</div>
              </div>
              <Switch label="Show the prompt" checked={settings.showPrompt}
                      onChange={(showPrompt) => setSettings({ ...settings, showPrompt })} />
            </div>
          </div>

          <div className="group-title">This chat</div>
          <div className="group">
            <div className="row">
              <div>
                <div className="label">Download as notes</div>
                <div className="hint">Questions, answers and sources as a Markdown file</div>
              </div>
              <button className="btn small" disabled={!messages.length}
                      onClick={() => download("study-assistant-chat.md", chatAsMarkdown(chat))}>
                Download
              </button>
            </div>
          </div>
        </Sheet>
      )}

      {passagesFor && (
        <PassagesSheet message={passagesFor} showPrompt={settings.showPrompt}
                       onOpenPage={(source, page) => setOpenPage({ source, page })}
                       onClose={() => setPassagesFor(null)} />
      )}
      {openPage && <PageSheet {...openPage} onClose={() => setOpenPage(null)} />}
    </div>
  );
}
