import { useCallback, useEffect, useMemo, useState } from "react";
import { Menu } from "lucide-react";
import { getJSON } from "./api.js";
import Sidebar, { NAV } from "./components/Sidebar.jsx";
import Ask from "./pages/Ask.jsx";
import HowItWorks from "./pages/HowItWorks.jsx";
import Lectures from "./pages/Lectures.jsx";
import Quiz from "./pages/Quiz.jsx";
import Results from "./pages/Results.jsx";

// Chats are saved in this browser (localStorage), so they survive a refresh.
const CHATS_KEY = "study-assistant.chats.v1";

function loadChats() {
  try {
    const chats = JSON.parse(localStorage.getItem(CHATS_KEY)) || [];
    // An answer that was still arriving when the page closed can't be finished.
    return chats.map((chat) => ({
      ...chat,
      messages: chat.messages.map((m) =>
        m.role === "assistant" && !["done", "error"].includes(m.status)
          ? { ...m, status: "error", error: "This answer was interrupted." } : m),
    }));
  } catch {
    return [];
  }
}

const pageFromHash = () => {
  const id = window.location.hash.replace("#/", "");
  return NAV.some((item) => item.id === id) ? id : "ask";
};

export default function App() {
  const [page, setPage] = useState(pageFromHash);
  const [info, setInfo] = useState(null);
  const [serverError, setServerError] = useState(null);
  const [chats, setChats] = useState(loadChats);
  const [currentChatId, setCurrentChatId] = useState(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [quizSource, setQuizSource] = useState(null);
  const [pendingQuestion, setPendingQuestion] = useState(null);

  useEffect(() => {
    getJSON("/info").then(setInfo).catch(() => setServerError(true));
  }, []);

  useEffect(() => {
    const onHashChange = () => setPage(pageFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  // Save chats shortly after they change (not on every streamed word).
  useEffect(() => {
    const timer = setTimeout(() => {
      try { localStorage.setItem(CHATS_KEY, JSON.stringify(chats)); } catch { /* storage full or blocked */ }
    }, 600);
    return () => clearTimeout(timer);
  }, [chats]);

  const navigate = (id) => {
    window.location.hash = `#/${id}`;
    setPage(id);
    setDrawerOpen(false);
  };

  const createChat = useCallback((title) => {
    const chat = { id: crypto.randomUUID(), title, updated: Date.now(), messages: [] };
    setChats((all) => [chat, ...all]);
    setCurrentChatId(chat.id);
    return chat.id;
  }, []);

  const updateChat = useCallback((id, change) => {
    setChats((all) => all.map((chat) => (chat.id === id ? change(chat) : chat)));
  }, []);

  const deleteChat = (id) => {
    setChats((all) => all.filter((chat) => chat.id !== id));
    if (id === currentChatId) setCurrentChatId(null);
  };

  const recentChats = useMemo(() => [...chats].sort((a, b) => b.updated - a.updated), [chats]);
  const currentChat = chats.find((chat) => chat.id === currentChatId) || null;
  const openDrawer = () => setDrawerOpen(true);

  return (
    <div className="app">
      <Sidebar
        page={page} info={info} chats={recentChats} currentChatId={currentChatId} open={drawerOpen}
        onNavigate={navigate}
        onNewChat={() => { setCurrentChatId(null); navigate("ask"); }}
        onOpenChat={(id) => { setCurrentChatId(id); navigate("ask"); }}
        onDeleteChat={deleteChat}
      />
      <div className={`backdrop${drawerOpen ? " open" : ""}`} onClick={() => setDrawerOpen(false)} />

      <main className="main">
        {serverError ? (
          <div className="page">
            <h1 className="large-title">Can't reach the server</h1>
            <p className="page-sub">
              The Python part of the app isn't running. Double-click <b>run_app.bat</b> in the project
              folder, wait for "ready", then reload this page.
            </p>
          </div>
        ) : page === "ask" ? (
          <Ask info={info} chat={currentChat} createChat={createChat} updateChat={updateChat}
               onNewChat={() => setCurrentChatId(null)} onMenu={openDrawer}
               pendingQuestion={pendingQuestion} clearPending={() => setPendingQuestion(null)} />
        ) : (
          <>
            <div className="topbar">
              <button className="menu-button" onClick={openDrawer} aria-label="Open menu"><Menu size={22} /></button>
              <span className="title">{NAV.find((item) => item.id === page).label}</span>
            </div>
            {page === "lectures" && (
              <Lectures
                onAsk={(question) => { setCurrentChatId(null); setPendingQuestion(question); navigate("ask"); }}
                onQuiz={(source) => { setQuizSource(source); navigate("quiz"); }} />
            )}
            {page === "quiz" && <Quiz source={quizSource} onSourceChange={setQuizSource} />}
            {page === "results" && <Results />}
            {page === "how" && <HowItWorks info={info} />}
          </>
        )}
      </main>
    </div>
  );
}
