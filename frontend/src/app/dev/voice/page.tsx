"use client";

import { useState } from "react";
import { useVoiceCoaching } from "@/hooks/useVoiceCoaching";
import VoiceCoachingToggle from "@/components/VoiceCoachingToggle";
import DuoButton from "@/components/ui/DuoButton";
import Card from "@/components/ui/Card";

const SAMPLE_ALERTS = [
  "Warning: knee valgus detected at your left knee. Please correct your form.",
  "Watch your hips: excessive hip drop.",
  "During Squat: Warning: trunk leaning too far at your trunk. Please correct your form.",
  "Watch your both sides: movement asymmetry.",
  "Warning: joint moving too fast at your right knee. Please correct your form.",
];

export default function DevVoicePage() {
  const { enabled, toggle, speak } = useVoiceCoaching();
  const [customText, setCustomText] = useState("");

  return (
    <div className="max-w-lg mx-auto">
      <h1 className="text-2xl font-extrabold text-[#3C3C3C] mb-6">Voice Coaching Test</h1>

      <Card className="mb-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-bold text-[#3C3C3C]">Voice Coaching</h3>
          <VoiceCoachingToggle enabled={enabled} onToggle={toggle} />
        </div>
        <p className="text-xs text-[#AFAFAF] mb-4">
          {enabled ? "Voice coaching is ON. Click the sample alerts below to hear them." : "Enable voice coaching to test TTS alerts."}
        </p>
      </Card>

      <Card className="mb-4">
        <h3 className="text-sm font-bold text-[#3C3C3C] mb-3">Sample Alerts</h3>
        <div className="flex flex-col gap-2">
          {SAMPLE_ALERTS.map((alert, i) => (
            <button
              key={i}
              onClick={() => speak(alert)}
              disabled={!enabled}
              className={`text-left text-xs px-3 py-2 rounded-lg border transition-colors ${
                enabled
                  ? "border-[#E5E5E5] hover:bg-[#F7F7F7] cursor-pointer"
                  : "border-[#F0F0F0] text-[#CDCDCD] cursor-not-allowed"
              }`}
            >
              {alert}
            </button>
          ))}
        </div>
      </Card>

      <Card>
        <h3 className="text-sm font-bold text-[#3C3C3C] mb-3">Custom Text</h3>
        <div className="flex gap-2">
          <input
            type="text"
            value={customText}
            onChange={(e) => setCustomText(e.target.value)}
            placeholder="Type custom alert text..."
            className="flex-1 px-3 py-2 text-xs border-2 border-[#E5E5E5] rounded-[10px] outline-none focus:border-[#1CB0F6]"
          />
          <DuoButton variant="blue" onClick={() => speak(customText)} disabled={!enabled || !customText}>
            Speak
          </DuoButton>
        </div>
      </Card>
    </div>
  );
}
