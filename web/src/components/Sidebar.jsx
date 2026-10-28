import { BookOpen, ChartColumn, ListChecks, MessageCircle, SquarePen, Trash2, Workflow } from "lucide-react";

export const NAV = [
  { id: "ask", label: "Ask", icon: MessageCircle },
  { id: "lectures", label: "Lectures", icon: BookOpen },
  { id: "quiz", label: "Quiz", icon: ListChecks },
  { id: "results", label: "Results", icon: ChartColumn },
  { id: "how", label: "How it works", icon: Workflow },
];

// The side panel: app name, New chat, the five pages, and recent chats.
export default function Sidebar({ page, info, chats, currentChatId, open, onNavigate, onNewChat,
                                  onOpenChat, onDeleteChat }) {
  return (
    <aside className={`sidebar${open ? " open" : ""}`} aria-label="Navigation">
      <div className="brand">
        <img src="/favicon.svg?v=2" alt="" />
        <div>
          <div className="name">Study Assistant</div>
          <div className="sub">CSL448 · RAG over your notes</div>
        </div>
      </div>

      <button className="new-chat" onClick={onNewChat}>
        <SquarePen size={17} /> New chat
      </button>

      <nav className="nav">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button key={id} className={page === id ? "active" : ""} onClick={() => onNavigate(id)}
                  aria-current={page === id ? "page" : undefined}>
            <Icon size={18} /> {label}
          </button>
        ))}
      </nav>

      <div className="recent">
        <h4>Recent chats</h4>
        {chats.length === 0 && <div className="empty">Your chats appear here.</div>}
        {chats.map((chat) => (
          <div key={chat.id} className={`item${chat.id === currentChatId && page === "ask" ? " active" : ""}`}>
            <button className="open" onClick={() => onOpenChat(chat.id)} title={chat.title}>{chat.title}</button>
            <button className="delete" onClick={() => onDeleteChat(chat.id)} aria-label={`Delete chat: ${chat.title}`}>
              <Trash2 size={15} />
            </button>
          </div>
        ))}
      </div>

      {info && (
        <div className="sidebar-foot">
          {info.lectures} lectures, {info.pages} pages<br />
          {info.model}
        </div>
      )}
    </aside>
  );
}
