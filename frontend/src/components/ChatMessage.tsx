"use client";

import type { ChatMessage as ChatMessageType } from "@/lib/types";

interface Props {
  message: ChatMessageType;
}

/** Strip ```json ... ``` blocks from assistant messages so users don't see raw JSON. */
function stripJsonBlocks(text: string): string {
  return text.replace(/```json\s*[\s\S]*?```/g, "").trim();
}

export default function ChatMessageBubble({ message }: Props) {
  const isUser = message.role === "user";
  const displayContent = isUser ? message.content : stripJsonBlocks(message.content);

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`
          max-w-[80%] rounded-[16px] px-4 py-2.5 text-sm leading-relaxed
          ${isUser
            ? "bg-[#1CB0F6] text-white rounded-br-[4px]"
            : "bg-[#F0F0F0] text-[#3C3C3C] rounded-bl-[4px] border border-[#E5E5E5]"
          }
        `}
      >
        {displayContent.split("\n").map((line, i) => (
          <span key={i}>
            {line}
            {i < displayContent.split("\n").length - 1 && <br />}
          </span>
        ))}
      </div>
    </div>
  );
}
