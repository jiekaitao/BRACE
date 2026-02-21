"use client";

import { useRef, useEffect, useState, KeyboardEvent } from "react";
import ChatMessageBubble from "./ChatMessage";
import DuoButton from "@/components/ui/DuoButton";
import type { ChatMessage } from "@/lib/types";

interface Props {
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  onSend: (message: string) => void;
  placeholder?: string;
}

export default function ChatPanel({ messages, loading, error, onSend, placeholder }: Props) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [input, setInput] = useState("");

  // Auto-scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages.length, loading]);

  const handleSend = () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");
    onSend(text);
  };

  const handleKeyDown = (e: KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full">
      {/* Messages area */}
      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto px-4 py-3 flex flex-col gap-3"
        style={{ minHeight: 200 }}
      >
        {messages.length === 0 && !loading && (
          <div className="text-center text-sm text-[#AFAFAF] py-8">
            Tell me about any past injuries or areas of concern...
          </div>
        )}
        {messages.map((msg, i) => (
          <ChatMessageBubble key={i} message={msg} />
        ))}
        {loading && (
          <div className="flex justify-start">
            <div className="bg-[#F0F0F0] rounded-[16px] rounded-bl-[4px] px-4 py-2.5 border border-[#E5E5E5]">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-[#AFAFAF] rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-2 h-2 bg-[#AFAFAF] rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-2 h-2 bg-[#AFAFAF] rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}
        {error && (
          <div className="text-center text-sm text-[#EA2B2B] py-2">{error}</div>
        )}
      </div>

      {/* Input bar */}
      <div className="border-t-2 border-[#E5E5E5] p-3 flex gap-2">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder || "Type a message..."}
          className="flex-1 px-4 py-2.5 text-sm border-2 border-[#E5E5E5] rounded-[12px] outline-none focus:border-[#1CB0F6] transition-colors"
          disabled={loading}
        />
        <DuoButton
          variant="blue"
          onClick={handleSend}
          disabled={loading || !input.trim()}
        >
          Send
        </DuoButton>
      </div>
    </div>
  );
}
