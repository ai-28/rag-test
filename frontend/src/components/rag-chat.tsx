"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type Citation = {
  index: number;
  page?: number;
  chunk_id?: string;
  kind?: string;
};

export type ChatMessage = {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  streaming?: boolean;
};

type NdjsonEvent =
  | { type: "context"; citations: Citation[] }
  | { type: "token"; content: string }
  | { type: "done" }
  | { type: "error"; message: string };

function parseLine(line: string): NdjsonEvent | null {
  const trimmed = line.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed) as NdjsonEvent;
  } catch {
    return null;
  }
}

function newId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

const MAX_HISTORY_TURNS = 40;

export function RagChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const endRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const canSend = useMemo(
    () => input.trim().length > 0 && !busy,
    [input, busy],
  );

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, busy]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "0px";
    const h = Math.min(el.scrollHeight, 168);
    el.style.height = `${Math.max(h, 40)}px`;
  }, [input]);

  const send = useCallback(async () => {
    const text = input.trim();
    if (!text || busy) return;

    setInput("");
    setError(null);

    const historyPayload = messages
      .filter((m) => !m.streaming && m.content.trim())
      .slice(-MAX_HISTORY_TURNS)
      .map((m) => ({ role: m.role, content: m.content }));

    const userId = newId();
    const assistantId = newId();

    setMessages((prev) => [
      ...prev,
      { id: userId, role: "user", content: text },
      {
        id: assistantId,
        role: "assistant",
        content: "",
        streaming: true,
        citations: [],
      },
    ]);

    setBusy(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: text,
          history:
            historyPayload.length > 0 ? historyPayload : undefined,
        }),
      });

      if (!res.ok) {
        const t = await res.text();
        throw new Error(t || `Request failed (${res.status})`);
      }

      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";
      const aid = assistantId;

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          const ev = parseLine(line);
          if (!ev) continue;
          if (ev.type === "error") {
            throw new Error(ev.message);
          }
          if (ev.type === "context") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aid && m.role === "assistant"
                  ? { ...m, citations: ev.citations ?? [] }
                  : m,
              ),
            );
          }
          if (ev.type === "token") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === aid && m.role === "assistant"
                  ? { ...m, content: m.content + ev.content }
                  : m,
              ),
            );
          }
        }
      }

      const tail = parseLine(buffer);
      if (tail?.type === "error") throw new Error(tail.message);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Unknown error";
      setError(msg);
      setMessages((prev) => {
        const last = prev[prev.length - 1];
        if (
          last?.role === "assistant" &&
          last.streaming &&
          !last.content.trim()
        ) {
          return prev.slice(0, -1);
        }
        if (last?.role === "assistant" && last.streaming) {
          return prev.map((m) =>
            m.id === last.id
              ? {
                  ...m,
                  streaming: false,
                  content:
                    m.content.trim() === ""
                      ? `Could not complete the reply: ${msg}`
                      : `${m.content}\n\n_(Error: ${msg})_`,
                }
              : m,
          );
        }
        return prev;
      });
    } finally {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantId && m.role === "assistant"
            ? { ...m, streaming: false }
            : m,
        ),
      );
      setBusy(false);
      textareaRef.current?.focus();
    }
  }, [busy, input, messages]);

  const newChat = useCallback(() => {
    if (busy) return;
    setMessages([]);
    setError(null);
    setInput("");
    textareaRef.current?.focus();
  }, [busy]);

  const onKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        void send();
      }
    },
    [send],
  );

  return (
    <div className="flex h-[100dvh] min-h-0 w-full flex-col bg-zinc-100 dark:bg-zinc-950">
      <header className="flex shrink-0 items-center justify-between gap-3 border-b border-zinc-200/80 bg-white/90 px-4 py-3 backdrop-blur-md dark:border-zinc-800/80 dark:bg-zinc-900/90">
        <div className="min-w-0">
          <h1 className="truncate text-lg font-semibold tracking-tight text-zinc-900 dark:text-zinc-50">
            PDF RAG chat
          </h1>
          <p className="truncate text-xs text-zinc-500 dark:text-zinc-400">
            Grounded on your ingested index · follow-ups supported
          </p>
        </div>
        <button
          type="button"
          onClick={newChat}
          disabled={busy}
          className="shrink-0 rounded-lg border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 shadow-sm transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-40 dark:border-zinc-700 dark:bg-zinc-800 dark:text-zinc-200 dark:hover:bg-zinc-700"
        >
          New chat
        </button>
      </header>

      <div className="mx-auto flex min-h-0 w-full max-w-3xl flex-1 flex-col px-3 sm:px-4">
        <div className="min-h-0 flex-1 overflow-y-auto py-4">
          {messages.length === 0 ? (
            <div className="flex h-full min-h-[12rem] flex-col items-center justify-center gap-3 px-4 text-center">
              <div className="rounded-[2rem] border border-zinc-300/90 px-8 py-10 dark:border-zinc-600/90">
                <p className="text-sm font-medium text-zinc-800 dark:text-zinc-100">
                  Start a conversation
                </p>
              </div>
            </div>
          ) : (
            <ul className="flex flex-col gap-4 pb-2">
              {messages.map((m) => (
                <li
                  key={m.id}
                  className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[min(100%,34rem)] rounded-[1.75rem] px-4 py-2.5 text-sm leading-relaxed ${
                      m.role === "user"
                        ? "border border-zinc-900/90 text-zinc-900 dark:border-zinc-100/90 dark:text-zinc-50"
                        : "border border-zinc-300/90 text-zinc-900 dark:border-zinc-600/90 dark:text-zinc-50"
                    }`}
                  >
                    <div className="mb-1 text-[10px] font-semibold uppercase tracking-wide opacity-70">
                      {m.role === "user" ? "You" : "Assistant"}
                    </div>
                    {m.role === "assistant" &&
                    m.streaming &&
                    !m.content ? (
                      <div className="flex items-center gap-1.5 py-1 text-zinc-400">
                        <span className="inline-flex gap-0.5">
                          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-400 [animation-delay:-0.2s]" />
                          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-400 [animation-delay:-0.1s]" />
                          <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-zinc-400" />
                        </span>
                      </div>
                    ) : (
                      <div className="whitespace-pre-wrap">{m.content}</div>
                    )}
                    {m.role === "assistant" &&
                    m.citations &&
                    m.citations.length > 0 ? (
                      <div className="mt-3 border-t border-zinc-200/80 pt-2 dark:border-zinc-700">
                        <p className="mb-1.5 text-[10px] font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                          Sources
                        </p>
                        <ul className="flex flex-wrap gap-1.5">
                          {m.citations.map((c) => (
                            <li
                              key={`${m.id}-${c.index}-${c.chunk_id ?? "x"}`}
                              className="rounded-full border border-zinc-200/90 px-2.5 py-0.5 text-[10px] text-zinc-700 dark:border-zinc-600/90 dark:text-zinc-300"
                            >
                              [{c.index}] p.{c.page ?? "?"}
                              {c.kind ? ` · ${c.kind}` : ""}
                            </li>
                          ))}
                        </ul>
                      </div>
                    ) : null}
                  </div>
                </li>
              ))}
            </ul>
          )}
          <div ref={endRef} className="h-px shrink-0" aria-hidden />
        </div>

        {error ? (
          <div className="shrink-0 px-1 pb-2">
            <div className="rounded-2xl border border-red-300/90 px-3 py-2 text-xs text-red-800 dark:border-red-800/80 dark:text-red-200">
              {error}
            </div>
          </div>
        ) : null}

        <div className="shrink-0 bg-transparent px-2 pb-[max(0.75rem,env(safe-area-inset-bottom))] pt-3 sm:px-3">
          <div className="flex min-h-[48px] items-end gap-1 rounded-full border border-zinc-300/90 bg-transparent py-1 pl-3 pr-1 transition-colors focus-within:border-zinc-900/70 dark:border-zinc-600/90 dark:focus-within:border-zinc-300/80">
            <textarea
              ref={textareaRef}
              rows={1}
              className="min-h-[40px] max-h-[168px] min-w-0 flex-1 resize-none rounded-full border-0 bg-transparent py-2.5 pl-1 pr-1 text-[15px] leading-relaxed text-zinc-900 outline-none placeholder:text-zinc-400 disabled:opacity-50 dark:text-zinc-100 dark:placeholder:text-zinc-500"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask about the PDF…"
              disabled={busy}
              aria-label="Message"
              autoComplete="off"
            />
            <button
              type="button"
              onClick={() => void send()}
              disabled={!canSend}
              title={busy ? "Waiting…" : "Send (Enter)"}
              className="group inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-full border border-zinc-900/90 text-zinc-900 transition hover:bg-zinc-900/5 disabled:cursor-not-allowed disabled:border-zinc-200 disabled:text-zinc-300 dark:border-zinc-100/90 dark:text-zinc-100 dark:hover:bg-zinc-100/5 dark:disabled:border-zinc-700 dark:disabled:text-zinc-600"
            >
              {busy ? (
                <span className="h-4 w-4 animate-spin rounded-full border-2 border-zinc-900/25 border-t-zinc-900 dark:border-zinc-100/30 dark:border-t-zinc-100" />
              ) : (
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  className="h-5 w-5 translate-x-px transition group-hover:translate-x-0.5 group-hover:-translate-y-px"
                  aria-hidden
                >
                  <path d="M3.478 2.404a.75.75 0 0 0-.926.941l2.432 7.905H13.5a.75.75 0 0 1 0 1.5H4.984l-2.432 7.905a.75.75 0 0 0 .926.94 60.519 60.519 0 0 0 18.445-8.986.75.75 0 0 0 0-1.218A60.517 60.517 0 0 0 3.478 2.404Z" />
                </svg>
              )}
            </button>
          </div>
          <p className="mt-2 px-1 text-center text-[11px] leading-snug text-zinc-400 dark:text-zinc-500">
            <span className="text-zinc-500 dark:text-zinc-400">Enter</span> to send
            · <span className="text-zinc-500 dark:text-zinc-400">Shift+Enter</span>{" "}
            new line · grounded on retrieved context
          </p>
        </div>
      </div>
    </div>
  );
}
