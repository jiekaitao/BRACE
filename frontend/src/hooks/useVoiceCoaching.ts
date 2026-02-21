"use client";

import { useRef, useCallback, useState } from "react";

interface UseVoiceCoachingReturn {
  enabled: boolean;
  toggle: () => void;
  speak: (text: string) => void;
}

/**
 * Browser SpeechSynthesis hook for voice coaching alerts.
 * Queues alerts with client-side cooldown and dedup.
 */
export function useVoiceCoaching(): UseVoiceCoachingReturn {
  const [enabled, setEnabled] = useState(false);
  const audioQueueRef = useRef<string[]>([]);
  const speakingRef = useRef(false);
  const cooldownRef = useRef(0);

  const processQueue = useCallback(async () => {
    if (speakingRef.current || audioQueueRef.current.length === 0) return;
    speakingRef.current = true;

    const text = audioQueueRef.current.shift()!;

    try {
      if ("speechSynthesis" in window) {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.1;
        utterance.pitch = 1.0;
        utterance.volume = 0.8;

        await new Promise<void>((resolve) => {
          utterance.onend = () => resolve();
          utterance.onerror = () => resolve();
          window.speechSynthesis.speak(utterance);
        });
      }
    } catch {
      // Silently fail
    } finally {
      speakingRef.current = false;
      if (audioQueueRef.current.length > 0) {
        processQueue();
      }
    }
  }, []);

  const speak = useCallback(
    (text: string) => {
      if (!enabled) return;

      // Client-side cooldown: 5s minimum between alerts
      const now = Date.now();
      if (now - cooldownRef.current < 5000) return;
      cooldownRef.current = now;

      // Limit queue size
      if (audioQueueRef.current.length >= 2) {
        audioQueueRef.current.shift();
      }

      audioQueueRef.current.push(text);
      processQueue();
    },
    [enabled, processQueue]
  );

  const toggle = useCallback(() => {
    setEnabled((prev) => {
      if (prev && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
      return !prev;
    });
  }, []);

  return { enabled, toggle, speak };
}
