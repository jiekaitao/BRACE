"use client";

import Card from "@/components/ui/Card";

export default function GeminiResearchPanel() {
  return (
    <Card>
      <div className="flex items-center gap-2 mb-3">
        {/* Sparkle icon */}
        <svg
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          className="flex-shrink-0"
        >
          <path
            d="M12 2L13.09 8.26L18 6L14.74 10.91L21 12L14.74 13.09L18 18L13.09 15.74L12 22L10.91 15.74L6 18L9.26 13.09L3 12L9.26 10.91L6 6L10.91 8.26L12 2Z"
            fill="#CE82FF"
          />
        </svg>
        <h3 className="text-base font-extrabold text-[#3C3C3C]">
          Research & Recommendations
        </h3>
      </div>
      <p className="text-sm text-[#777777] leading-relaxed">
        Coming soon — Gemini will analyze your injuries and find relevant
        exercises, stretches, and precautions tailored to your profile.
      </p>
    </Card>
  );
}
