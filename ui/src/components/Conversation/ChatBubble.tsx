"use client";

type ChatBubbleProps = {
  role: "user" | "assistant";
  content: string;
  timestamp?: string;
};

/** Strip <think>...</think> blocks from LLM output for display */
function stripThinkTags(text: string): string {
  // If there's an unclosed <think> tag, hide everything after it
  if (text.includes("<think>") && !text.includes("</think>")) {
    return text.replace(/<think>[\s\S]*$/, "").trim();
  }
  return text.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
}

export default function ChatBubble({ role, content, timestamp }: ChatBubbleProps) {
  const isUser = role === "user";
  const displayContent = isUser ? content : stripThinkTags(content);

  if (!displayContent) return null;

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div
        className={`max-w-[85%] rounded-lg px-4 py-2.5 ${
          isUser
            ? "bg-mc-accent/15 text-mc-text border border-mc-accent/20"
            : "bg-mc-bg-tertiary text-mc-text"
        }`}
      >
        {!isUser && (
          <div className="text-xs font-medium text-mc-accent mb-1">JARVIS</div>
        )}
        <div className="text-sm whitespace-pre-wrap leading-relaxed">{displayContent}</div>
        {timestamp && (
          <div className="text-xs text-mc-text-secondary mt-1">{timestamp}</div>
        )}
      </div>
    </div>
  );
}
