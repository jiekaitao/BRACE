"use client";

import { useState } from "react";
import DuoButton from "@/components/ui/DuoButton";
import Card from "@/components/ui/Card";
import PathCard from "@/components/ui/PathCard";
import ProgressBar from "@/components/ui/ProgressBar";

export default function DevComponentsPage() {
  const [progress, setProgress] = useState(65);

  return (
    <div className="max-w-3xl mx-auto">
      <h1 className="text-2xl font-extrabold text-[#3C3C3C] mb-6">Component Gallery</h1>

      {/* Buttons */}
      <section className="mb-8">
        <h2 className="text-lg font-bold text-[#3C3C3C] mb-3">DuoButton</h2>
        <div className="flex flex-wrap gap-3">
          <DuoButton variant="primary">Primary</DuoButton>
          <DuoButton variant="secondary">Secondary</DuoButton>
          <DuoButton variant="blue">Blue</DuoButton>
          <DuoButton variant="danger">Danger</DuoButton>
          <DuoButton disabled>Disabled</DuoButton>
        </div>
      </section>

      {/* Cards */}
      <section className="mb-8">
        <h2 className="text-lg font-bold text-[#3C3C3C] mb-3">Card</h2>
        <div className="grid grid-cols-2 gap-3">
          <Card>
            <p className="text-sm text-[#777777]">Static card</p>
          </Card>
          <Card interactive>
            <p className="text-sm text-[#777777]">Interactive card (click me)</p>
          </Card>
        </div>
      </section>

      {/* PathCards */}
      <section className="mb-8">
        <h2 className="text-lg font-bold text-[#3C3C3C] mb-3">PathCard (hover to see color transition)</h2>
        <div className="grid grid-cols-2 gap-4">
          <PathCard
            title="Personal Safety"
            subtitle="Grayscale to color on hover"
            className="bg-gradient-to-br from-[#1CB0F6] to-[#58CC02]"
          />
          <PathCard
            title="Team Monitor"
            subtitle="Grayscale to color on hover"
            className="bg-gradient-to-br from-[#CE82FF] to-[#1CB0F6]"
          />
        </div>
      </section>

      {/* ProgressBar */}
      <section className="mb-8">
        <h2 className="text-lg font-bold text-[#3C3C3C] mb-3">ProgressBar</h2>
        <ProgressBar value={progress} />
        <input
          type="range"
          min="0"
          max="100"
          step="1"
          value={progress}
          onChange={(e) => setProgress(parseFloat(e.target.value))}
          className="w-full mt-2"
        />
      </section>
    </div>
  );
}
