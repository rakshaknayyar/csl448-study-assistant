// Talking to the Python server (server.py). Every call goes to /api/...

async function errorMessage(response) {
  try {
    const body = await response.json();
    return body.detail || response.statusText;
  } catch {
    return response.statusText || "The server didn't answer.";
  }
}

export async function getJSON(path) {
  const response = await fetch(`/api${path}`);
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

export async function postJSON(path, body = {}) {
  const response = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorMessage(response));
  return response.json();
}

// Ask a question. The server streams one JSON event per line
// (retrieved, token, answer, plain, error); onEvent is called for each.
export async function askStream(body, onEvent) {
  const response = await fetch("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error(await errorMessage(response));

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let newline;
    while ((newline = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, newline).trim();
      buffer = buffer.slice(newline + 1);
      if (line) onEvent(JSON.parse(line));
    }
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

export const pageImageUrl = (source, page) => `/api/page/${encodeURIComponent(source)}/${page}`;
