"use client";

import { useEffect, useRef, useState } from "react";

interface ProgressBarProps {
  value: number; // 0-100
  color?: string;
  height?: number;
  label?: string;
}

export default function ProgressBar({
  value,
  color = "#58CC02",
  height = 16,
  label,
}: ProgressBarProps) {
  const [animated, setAnimated] = useState(false);
  const fillRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Double rAF ensures the browser paints 0% before animating
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        setAnimated(true);
      });
    });
  }, []);

  const clampedValue = Math.max(0, Math.min(100, value));

  return (
    <div>
      {label && (
        <div className="flex justify-between items-center mb-1">
          <span className="text-xs font-bold text-[#4B4B4B] uppercase tracking-[0.03em]">
            {label}
          </span>
          <span className="text-xs font-bold text-[#777777]">
            {Math.round(clampedValue)}%
          </span>
        </div>
      )}
      <div
        className="bg-[#E5E5E5] rounded-full w-full overflow-hidden"
        style={{ height }}
      >
        <div
          ref={fillRef}
          className="h-full rounded-full"
          style={{
            backgroundColor: color,
            width: animated ? `${clampedValue}%` : "0%",
            transition: animated
              ? "width 0.6s cubic-bezier(0.34, 1.56, 0.64, 1)"
              : "none",
          }}
        />
      </div>
    </div>
  );
}
