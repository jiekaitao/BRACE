"use client";

import { useState, useCallback, useRef } from "react";
import { getApiBase } from "@/lib/api";
import { getToken } from "@/lib/auth";
import type { ChatMessage, InjuryProfile, ChatResponse } from "@/lib/types";

// Metric mapping for display — which BRACE metric each injury type maps to
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

interface UseChatReturn {
  messages: ChatMessage[];
  loading: boolean;
  error: string | null;
  extractedProfile: InjuryProfile | null;
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

  return { messages, loading, error, extractedProfile, profileComplete, sendMessage, confirmProfile, reset };
}
