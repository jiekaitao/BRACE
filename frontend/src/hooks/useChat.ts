"use client";

import { useState, useCallback, useRef, useMemo } from "react";
import { getApiBase } from "@/lib/api";
import { getToken } from "@/lib/auth";
import type { ChatMessage, InjuryProfile, InjuryEntry, ChatResponse } from "@/lib/types";

// Keyword → injury type mapping for partial extraction
const INJURY_KEYWORDS: Record<string, string> = {
  acl: "acl",
  "anterior cruciate": "acl",
  shoulder: "shoulder",
  rotator: "shoulder",
  ankle: "ankle",
  "lower back": "lower_back",
  "low back": "lower_back",
  lumbar: "lower_back",
  knee: "knee_general",
  patella: "knee_general",
  hip: "hip",
  hamstring: "hamstring",
  wrist: "wrist",
  carpal: "wrist",
};

// Metric mapping for display
export const INJURY_METRIC_MAP: Record<string, string> = {
  acl: "FPPA (knee valgus)",
  shoulder: "Angular Velocity",
  ankle: "Angular Velocity",
  lower_back: "Trunk Lean + Hip Drop",
  knee_general: "FPPA (knee valgus)",
  hip: "Hip Drop",
  hamstring: "Bilateral Asymmetry",
  wrist: "Angular Velocity",
};

function extractPartialInjuries(messages: ChatMessage[]): InjuryEntry[] {
  const found = new Set<string>();
  const injuries: InjuryEntry[] = [];

  // Scan all messages (user + assistant) for injury keywords
  for (const msg of messages) {
    const text = msg.content.toLowerCase();
    for (const [keyword, type] of Object.entries(INJURY_KEYWORDS)) {
      if (text.includes(keyword) && !found.has(type)) {
        found.add(type);
        // Detect side from context
        let side: string = "unknown";
        if (text.includes("left")) side = "left";
        else if (text.includes("right")) side = "right";
        else if (text.includes("both")) side = "bilateral";

        injuries.push({ type, side, severity: "moderate", timeframe: "chronic" });
      }
    }
  }
  return injuries;
}

interface UseChatReturn {
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  extractedProfile: InjuryProfile | null;
  partialInjuries: InjuryEntry[];
  profileComplete: boolean;
  sendMessage: (content: string) => Promise<void>;
  confirmProfile: (userId?: string) => Promise<void>;
  reset: () => void;
}

export function useChat(userId?: string): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [extractedProfile, setExtractedProfile] = useState<InjuryProfile | null>(null);
  const [profileComplete, setProfileComplete] = useState(false);
  const messagesRef = useRef<ChatMessage[]>([]);

  // Partial injury extraction from conversation keywords
  const partialInjuries = useMemo(() => extractPartialInjuries(messages), [messages]);

  const sendMessage = useCallback(async (content: string) => {
    const userMsg: ChatMessage = { role: "user", content };
    const newMessages = [...messagesRef.current, userMsg];
    messagesRef.current = newMessages;
    setMessages(newMessages);
    setLoading(true);
    setError(null);

    try {
      const token = getToken();
      const res = await fetch(`${getApiBase()}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          user_id: userId || undefined,
          messages: newMessages,
        }),
      });

      if (!res.ok) {
        throw new Error("Failed to get response");
      }

      const data: ChatResponse = await res.json();
      const assistantMsg: ChatMessage = { role: "assistant", content: data.response };
      const updatedMessages = [...newMessages, assistantMsg];
      messagesRef.current = updatedMessages;
      setMessages(updatedMessages);

      if (data.extracted_profile) {
        setExtractedProfile(data.extracted_profile);
      }
      if (data.profile_complete) {
        setProfileComplete(true);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  }, [userId]);

  const confirmProfile = useCallback(async (overrideUserId?: string) => {
    if (!extractedProfile) return;

    try {
      const token = getToken();
      const res = await fetch(`${getApiBase()}/api/chat/confirm-profile`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({
          user_id: overrideUserId || userId || undefined,
          injury_profile: extractedProfile,
        }),
      });

      if (!res.ok) throw new Error("Failed to confirm profile");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to save profile");
    }
  }, [extractedProfile, userId]);

  const reset = useCallback(() => {
    setMessages([]);
    messagesRef.current = [];
    setLoading(false);
    setError(null);
    setExtractedProfile(null);
    setProfileComplete(false);
  }, []);

  return { messages, loading, error, extractedProfile, partialInjuries, profileComplete, sendMessage, confirmProfile, reset };
}
