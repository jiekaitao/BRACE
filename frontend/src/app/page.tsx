"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import DuoButton from "@/components/ui/DuoButton";
import Card from "@/components/ui/Card";
import PathCard from "@/components/ui/PathCard";
import DemoVideoModal from "@/components/DemoVideoModal";

export default function HomePage() {
  const router = useRouter();
  const [showDemoModal, setShowDemoModal] = useState(false);

  const onDemoSelect = (filename: string) => {
    setShowDemoModal(false);
    router.push(`/analyze?mode=demo&video=${encodeURIComponent(filename)}`);
  };

  return (
    <div className="min-h-screen flex flex-col items-center px-5 py-12">
      {/* Hero */}
      <div className="scale-in text-center mb-8 max-w-lg">
        <h1 className="text-[32px] font-extrabold text-[#3C3C3C] leading-[1.2] tracking-[-0.01em] mb-2">
          BRACE
        </h1>
        <p className="text-base text-[#777777] leading-[1.5]">
          Real-time AI-powered movement analysis to keep you safe.
        </p>
      </div>

      {/* Path Cards */}
      <div
        className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full max-w-2xl mb-8 fade-up"
        style={{ "--stagger-index": 1 } as React.CSSProperties}
      >
        <PathCard
          title="Personal Workout Safety"
          subtitle="Real-time injury risk detection for your training"
          onClick={() => router.push("/onboarding?path=personal")}
          className="bg-gradient-to-br from-[#1CB0F6] to-[#58CC02]"
        >
          {/* Person icon */}
          <div className="absolute top-6 right-6 opacity-20 group-hover:opacity-30 transition-opacity duration-400">
            <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
              <circle cx="12" cy="7" r="4" />
            </svg>
          </div>
        </PathCard>

        <PathCard
          title="Team Performance Monitor"
          subtitle="Track and compare athletes across your roster"
          onClick={() => router.push("/onboarding?path=team")}
          className="bg-gradient-to-br from-[#CE82FF] to-[#1CB0F6]"
        >
          {/* Group icon */}
          <div className="absolute top-6 right-6 opacity-20 group-hover:opacity-30 transition-opacity duration-400">
            <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2" />
              <circle cx="9" cy="7" r="4" />
              <path d="M23 21v-2a4 4 0 0 0-3-3.87" />
              <path d="M16 3.13a4 4 0 0 1 0 7.75" />
            </svg>
          </div>
        </PathCard>
      </div>

      {/* Quick access CTAs */}
      <div
        className="flex flex-col sm:flex-row gap-3 mb-16 w-full max-w-2xl fade-up"
        style={{ "--stagger-index": 2 } as React.CSSProperties}
      >
        <Link href="/analyze?mode=upload" className="flex-1 no-underline">
          <DuoButton variant="primary" fullWidth>
            Upload Video
          </DuoButton>
        </Link>
        <Link href="/analyze?mode=webcam" className="flex-1 no-underline">
          <DuoButton variant="blue" fullWidth>
            Use Webcam
          </DuoButton>
        </Link>
        <div className="flex-1">
          <DuoButton
            variant="secondary"
            fullWidth
            onClick={() => setShowDemoModal(true)}
          >
            Demo Videos
          </DuoButton>
        </div>
      </div>

      {/* Feature cards */}
      <div className="stagger-children grid grid-cols-1 sm:grid-cols-3 gap-4 w-full max-w-3xl">
        <Card>
          <div className="flex flex-col items-center text-center gap-3">
            <div className="w-12 h-12 rounded-full bg-[#DDF4FF] flex items-center justify-center">
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#1CB0F6"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
                <circle cx="12" cy="12" r="3" />
              </svg>
            </div>
            <h2 className="text-base font-bold text-[#3C3C3C]">
              Detect Patterns
            </h2>
            <p className="text-sm text-[#777777]">
              AI identifies and groups your repeated movements automatically.
            </p>
          </div>
        </Card>

        <Card>
          <div className="flex flex-col items-center text-center gap-3">
            <div className="w-12 h-12 rounded-full bg-[#FFDFE0] flex items-center justify-center">
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#EA2B2B"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
                <line x1="12" y1="9" x2="12" y2="13" />
                <line x1="12" y1="17" x2="12.01" y2="17" />
              </svg>
            </div>
            <h2 className="text-base font-bold text-[#3C3C3C]">
              Flag Inconsistencies
            </h2>
            <p className="text-sm text-[#777777]">
              Spot reps that deviate from your baseline — potential injury risk.
            </p>
          </div>
        </Card>

        <Card>
          <div className="flex flex-col items-center text-center gap-3">
            <div className="w-12 h-12 rounded-full bg-[#F0FBE4] flex items-center justify-center">
              <svg
                width="24"
                height="24"
                viewBox="0 0 24 24"
                fill="none"
                stroke="#58CC02"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M20.84 4.61a5.5 5.5 0 0 0-7.78 0L12 5.67l-1.06-1.06a5.5 5.5 0 0 0-7.78 7.78l1.06 1.06L12 21.23l7.78-7.78 1.06-1.06a5.5 5.5 0 0 0 0-7.78z" />
              </svg>
            </div>
            <h2 className="text-base font-bold text-[#3C3C3C]">
              Protect Your Body
            </h2>
            <p className="text-sm text-[#777777]">
              Stay safe with real-time feedback on movement quality.
            </p>
          </div>
        </Card>
      </div>

      {/* How it works */}
      <div
        className="mt-16 w-full max-w-3xl fade-up"
        style={{ "--stagger-index": 5 } as React.CSSProperties}
      >
        <h2 className="text-xl font-bold text-[#3C3C3C] text-center mb-6">
          How It Works
        </h2>
        <div className="stagger-children grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Card>
            <div className="flex flex-col items-center text-center gap-2">
              <span className="text-2xl font-extrabold text-[#1CB0F6]">1</span>
              <h3 className="text-sm font-bold text-[#3C3C3C]">Calibrate</h3>
              <p className="text-sm text-[#777777]">
                Perform your exercise. The first 20% of frames establish your
                baseline.
              </p>
            </div>
          </Card>
          <Card>
            <div className="flex flex-col items-center text-center gap-2">
              <span className="text-2xl font-extrabold text-[#58CC02]">2</span>
              <h3 className="text-sm font-bold text-[#3C3C3C]">Analyze</h3>
              <p className="text-sm text-[#777777]">
                Each rep is compared to your pattern. Consistent reps glow green.
              </p>
            </div>
          </Card>
          <Card>
            <div className="flex flex-col items-center text-center gap-2">
              <span className="text-2xl font-extrabold text-[#EA2B2B]">3</span>
              <h3 className="text-sm font-bold text-[#3C3C3C]">Alert</h3>
              <p className="text-sm text-[#777777]">
                Deviations turn orange/red, warning you of potential injury risk.
              </p>
            </div>
          </Card>
        </div>
      </div>

      {/* Demo video modal */}
      {showDemoModal && (
        <DemoVideoModal
          onSelect={onDemoSelect}
          onClose={() => setShowDemoModal(false)}
        />
      )}
    </div>
  );
}
