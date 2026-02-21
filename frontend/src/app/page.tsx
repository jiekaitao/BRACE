"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import DuoButton from "@/components/ui/DuoButton";
import PathCard from "@/components/ui/PathCard";
import DemoVideoModal from "@/components/DemoVideoModal";
import { useAuth } from "@/contexts/AuthContext";

export default function HomePage() {
  const router = useRouter();
  const { user } = useAuth();
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
        className="flex flex-col sm:flex-row gap-3 mb-8 w-full max-w-2xl fade-up"
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

      {/* Dashboard CTA for logged-in users */}
      {user && (
        <div
          className="w-full max-w-2xl mb-8 fade-up"
          style={{ "--stagger-index": 3 } as React.CSSProperties}
        >
          <Link href="/dashboard" className="no-underline">
            <DuoButton variant="secondary" fullWidth>
              My Dashboard
            </DuoButton>
          </Link>
        </div>
      )}

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
