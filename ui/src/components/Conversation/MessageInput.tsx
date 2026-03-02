"use client";

import { useState, FormEvent } from "react";

export default function MessageInput({
  onSend,
  disabled,
}: {
  onSend: (text: string) => void;
  disabled?: boolean;
}) {
  const [text, setText] = useState("");

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    const t = text.trim();
    if (!t || disabled) return;
    onSend(t);
    setText("");
  };

  return (
    <form onSubmit={handleSubmit} className="p-4 border-t border-mc-border">
      <div className="flex gap-2">
        <input
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type a message..."
          disabled={disabled}
          className="flex-1 bg-mc-bg-tertiary border border-mc-border rounded-lg px-4 py-2 text-mc-text placeholder-mc-text-secondary focus:outline-none focus:ring-2 focus:ring-mc-accent"
        />
        <button
          type="submit"
          disabled={disabled || !text.trim()}
          className="p-2 rounded-lg bg-mc-accent text-white hover:bg-mc-accent/80 disabled:opacity-50 disabled:cursor-not-allowed"
          aria-label="Send"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
          </svg>
        </button>
      </div>
    </form>
  );
}
