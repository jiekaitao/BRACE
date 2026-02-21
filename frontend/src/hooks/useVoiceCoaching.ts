"use client";

import { useRef, useCallback, useState } from "react";
import { getApiBase } from "@/lib/api";

interface UseVoiceCoachingReturn {
  enabled: boolean;
  toggle: () => void;
  speak: (text: string) => void;
}

/**
 * Voice coaching hook: ElevenLabs TTS via backend, with automatic
 * fallback to browser SpeechSynthesis if the endpoint is unavailable.
 */
export function useVoiceCoaching(): UseVoiceCoachingReturn {
  const [enabled, setEnabled] = useState(false);
  const audioQueueRef = useRef<string[]>([]);
  const speakingRef = useRef(false);
  const cooldownRef = useRef(0);
  // Circuit breaker: null = untested, true = working, false = failed
  const ttsAvailableRef = useRef<boolean | null>(null);
  const currentAudioRef = useRef<HTMLAudioElement | null>(null);

  const speakBrowser = useCallback((text: string): Promise<void> => {
    return new Promise<void>((resolve) => {
      if (!("speechSynthesis" in window)) {
        resolve();
        return;
      }
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.rate = 1.1;
      utterance.pitch = 1.0;
      utterance.volume = 0.8;
      utterance.onend = () => resolve();
      utterance.onerror = () => resolve();
      window.speechSynthesis.speak(utterance);
    });
  }, []);

  const processQueue = useCallback(async () => {
    if (speakingRef.current || audioQueueRef.current.length === 0) return;
    speakingRef.current = true;

    const text = audioQueueRef.current.shift()!;

    try {
      // Try ElevenLabs TTS unless circuit breaker tripped
      if (ttsAvailableRef.current !== false) {
        try {
          const resp = await fetch(
            `${getApiBase()}/api/tts?text=${encodeURIComponent(text)}`
          );
          if (resp.ok) {
            ttsAvailableRef.current = true;
            const blob = await resp.blob();
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            currentAudioRef.current = audio;
            await new Promise<void>((resolve) => {
              audio.onended = () => resolve();
              audio.onerror = () => resolve();
              audio.play().catch(() => resolve());
            });
            currentAudioRef.current = null;
            URL.revokeObjectURL(url);
            return;
          }
          // Non-200 — trip circuit breaker, fall through to browser TTS
          ttsAvailableRef.current = false;
        } catch {
          ttsAvailableRef.current = false;
        }
      }

      // Fallback: browser SpeechSynthesis
      await speakBrowser(text);
    } catch {
      // Silently fail
    } finally {
      speakingRef.current = false;
      if (audioQueueRef.current.length > 0) {
        processQueue();
      }
    }
  }, [speakBrowser]);

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
      if (prev) {
        // Stopping — cancel any in-flight audio
        if (currentAudioRef.current) {
          currentAudioRef.current.pause();
          currentAudioRef.current = null;
        }
        if ("speechSynthesis" in window) {
          window.speechSynthesis.cancel();
        }
      }
      return !prev;
    });
  }, []);

  return { enabled, toggle, speak };
}
