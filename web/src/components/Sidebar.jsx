import { BookOpen, ChartColumn, ListChecks, MessageCircle, PanelLeftClose, PanelLeftOpen, SquarePen, Trash2,
         Workflow } from "lucide-react";

export const NAV = [
  { id: "ask", label: "Ask", icon: MessageCircle },
  { id: "lectures", label: "Lectures", icon: BookOpen },
  { id: "quiz", label: "Quiz", icon: ListChecks },
  { id: "results", label: "Results", icon: ChartColumn },
  { id: "how", label: "How it works", icon: Workflow },
];

// The side panel: app name, New chat, the five pages, and recent chats.
// On desktop it can be collapsed into a slim rail of icons (collapsed = true).
export default function Sidebar({ page, info, chats, currentChatId, open, collapsed, onToggleCollapsed,
                                  onNavigate, onNewChat, onOpenChat, onDeleteChat }) {
  return (
    <aside className={`sidebar${open ? " open" : ""}${collapsed ? " collapsed" : ""}`} aria-label="Navigation">
      <div className="brand">
        <img src="/favicon.svg?v=3" alt="" />
        <div className="name text">Study Assistant</div>
        <button className="collapse-button" onClick={onToggleCollapsed}
                aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
                title={collapsed ? "Expand sidebar" : "Collapse sidebar"}>
          {collapsed ? <PanelLeftOpen size={19} /> : <PanelLeftClose size={19} />}
        </button>
      </div>

      <button className="new-chat" onClick={onNewChat} title="New chat" aria-label="New chat">
        <SquarePen size={17} /> <span className="text">New chat</span>
      </button>

      <nav className="nav">
        {NAV.map(({ id, label, icon: Icon }) => (
          <button key={id} className={page === id ? "active" : ""} onClick={() => onNavigate(id)}
                  aria-current={page === id ? "page" : undefined} title={collapsed ? label : undefined}
                  aria-label={label}>
            <Icon size={18} /> <span className="text">{label}</span>
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
          <div className="course">CSL448 · RAG over your notes</div>
          {info.lectures} lectures, {info.pages} pages · {info.model}
        </div>
      )}
    </aside>
  );
}
