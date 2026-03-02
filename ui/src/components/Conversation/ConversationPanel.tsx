"use client";

import { useEffect, useRef } from "react";
import ChatBubble from "./ChatBubble";
import MessageInput from "./MessageInput";
import { getConversation, clearConversation } from "@/lib/api";

type Message = { role: string; content: string };

export default function ConversationPanel({
  messages,
  setMessages,
  onSend,
  loading,
  backendOnline,
}: {
  messages: Message[];
  setMessages: (m: Message[]) => void;
  onSend: (text: string) => void;
  loading: boolean;
  backendOnline?: boolean;
}) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const handleClear = async () => {
    try {
      await clearConversation();
      const res = await getConversation();
      setMessages(res.messages);
    } catch {}
  };

  const handleExtract = () => {
    const text = messages
      .map((m) => `${m.role === "user" ? "You" : "JARVIS"}: ${m.content}`)
      .join("\n\n");
    navigator.clipboard.writeText(text);
  };

  return (
    <div className="flex flex-col h-full bg-mc-bg-secondary border-l border-mc-border">
      <div className="flex items-center justify-between p-4 border-b border-mc-border">
        <div className="flex items-center gap-2">
          <span className="text-mc-text font-semibold">Conversation</span>
          {messages.length > 0 && (
            <span className="text-xs text-mc-text-secondary bg-mc-bg-tertiary px-1.5 py-0.5 rounded">
              {messages.length}
            </span>
          )}
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleClear}
            disabled={!backendOnline || messages.length === 0}
            className="px-3 py-1 text-sm rounded bg-mc-bg-tertiary hover:bg-mc-border text-mc-text disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Clear
          </button>
          <button
            onClick={handleExtract}
            disabled={messages.length === 0}
            className="px-3 py-1 text-sm rounded bg-mc-bg-tertiary hover:bg-mc-border text-mc-text disabled:opacity-40 disabled:cursor-not-allowed"
          >
            Export
          </button>
        </div>
      </div>
      <div ref={scrollRef} className="flex-1 overflow-y-auto p-4">
        {messages.length === 0 && !loading && (
          <div className="flex flex-col items-center justify-center h-full text-mc-text-secondary text-sm">
            <svg className="w-12 h-12 mb-3 opacity-30" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
            </svg>
            <p>No messages yet</p>
            <p className="text-xs mt-1">Type a message or use voice to start</p>
          </div>
        )}
        {messages.map((m, i) => (
          <ChatBubble
            key={i}
            role={m.role as "user" | "assistant"}
            content={m.content}
          />
        ))}
        {loading && (
          <div className="flex justify-start mb-4">
            <div className="bg-mc-bg-tertiary rounded-lg px-4 py-2">
              <div className="text-xs text-mc-accent mb-1">JARVIS</div>
              <div className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-mc-accent animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-1.5 h-1.5 rounded-full bg-mc-accent animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-1.5 h-1.5 rounded-full bg-mc-accent animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}
      </div>
      <MessageInput onSend={onSend} disabled={loading || !backendOnline} />
    </div>
  );
}
