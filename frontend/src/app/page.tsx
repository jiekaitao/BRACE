"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import DuoButton from "@/components/ui/DuoButton";
import PathCard from "@/components/ui/PathCard";
import DemoVideoModal from "@/components/DemoVideoModal";
import TeamSportBrowser from "@/components/TeamSportBrowser";
import { useAuth } from "@/contexts/AuthContext";

interface CardRect {
  top: number;
  left: number;
  width: number;
  height: number;
}

export default function HomePage() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const [showDemoModal, setShowDemoModal] = useState(false);

  // Redirect onboarded users to dashboard
  useEffect(() => {
    if (!loading && user?.injury_profile) {
      router.replace("/dashboard");
    }
  }, [loading, user, router]);

  // Team expansion state machine
  const [teamExpanded, setTeamExpanded] = useState(false);
  const [showContent, setShowContent] = useState(true);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [cardRect, setCardRect] = useState<CardRect | null>(null);

  const contentRef = useRef<HTMLDivElement>(null);
  const teamCardRef = useRef<HTMLDivElement>(null);

  const handleTeamExpand = useCallback(() => {
    if (isTransitioning || !teamCardRef.current || !contentRef.current) return;

    // Measure team card position relative to the content container
    const cardEl = teamCardRef.current;
    const containerEl = contentRef.current;
    const cardBox = cardEl.getBoundingClientRect();
    const containerBox = containerEl.getBoundingClientRect();

    setCardRect({
      top: cardBox.top - containerBox.top,
      left: cardBox.left - containerBox.left,
      width: cardBox.width,
      height: cardBox.height,
    });

    setIsTransitioning(true);
    setShowContent(false);

    // Start expansion on next frame so initial position is set first
    requestAnimationFrame(() => {
      setTeamExpanded(true);
    });

    // Fade content in after expansion completes
    setTimeout(() => {
      setShowContent(true);
      setIsTransitioning(false);
    }, 450);
  }, [isTransitioning]);

  const handleTeamCollapse = useCallback(() => {
    if (isTransitioning) return;
    setIsTransitioning(true);
    // Phase 1: fade out expanded content
    setShowContent(false);
    // Phase 2: collapse back to card position
    setTimeout(() => {
      setTeamExpanded(false);
    }, 200);
    // Phase 3: clear overlay after collapse animation finishes
    setTimeout(() => {
      setCardRect(null);
      setShowContent(true);
      setIsTransitioning(false);
    }, 650);
  }, [isTransitioning]);

  const onDemoSelect = (filename: string) => {
    setShowDemoModal(false);
    router.push(`/analyze?mode=demo&video=${encodeURIComponent(filename)}`);
  };

  // Compute animated values: start at card position, expand to fill container
  const collapsedStyle = cardRect
    ? { top: cardRect.top, left: cardRect.left, width: cardRect.width, height: cardRect.height }
    : { top: 0, left: 0, width: "100%" as string | number, height: "100%" as string | number };

  const expandedStyle = { top: 0, left: 0, width: "100%", height: "100%" };

  return (
    <div className="min-h-screen flex flex-col items-center px-5 py-12">
      {/* Hero */}
      <div className="scale-in text-center mb-8 max-w-lg">
        <p className="text-base text-[#777777] leading-[1.5]">
          Real-time AI-powered movement analysis to keep you safe.
        </p>
      </div>

      {/* Content area - relative container for the expanding overlay */}
      <div ref={contentRef} className="relative w-full max-w-2xl mb-8">
        {/* Path Cards - always in DOM */}
        <div
          className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full mb-4 fade-up"
          style={{ "--stagger-index": 1 } as React.CSSProperties}
        >
          <PathCard
            title="Personal Workout Safety"
            subtitle="Real-time injury risk detection for your training"
            onClick={() => router.push("/onboarding?path=personal")}
            className="bg-gradient-to-br from-[#1CB0F6] to-[#58CC02]"
          >
            <div className="absolute top-6 right-6 opacity-20 group-hover:opacity-30 transition-opacity duration-400">
              <svg width="80" height="80" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
                <circle cx="12" cy="7" r="4" />
              </svg>
            </div>
          </PathCard>

          <div ref={teamCardRef}>
            <PathCard
              title="Team Performance Monitor"
              subtitle="Track and compare athletes across your roster"
              onClick={handleTeamExpand}
              className="bg-gradient-to-br from-[#CE82FF] to-[#1CB0F6]"
            >
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
        </div>

        {/* Quick access CTAs */}
        <div
          className="flex flex-col sm:flex-row gap-3 w-full fade-up"
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

        {/* Expanding overlay: starts at team card size/position, grows to fill container */}
        <AnimatePresence>
          {cardRect && (
            <motion.div
              key="team-overlay"
              initial={collapsedStyle}
              animate={teamExpanded ? expandedStyle : collapsedStyle}
              exit={collapsedStyle}
              transition={{
                top: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
                left: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
                width: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
                height: { duration: 0.4, ease: [0.16, 1, 0.3, 1] },
              }}
              className="absolute z-10 rounded-[20px] border-2 border-[#E5E5E5] shadow-[0_6px_0_#E5E5E5] bg-gradient-to-br from-[#CE82FF] to-[#1CB0F6] overflow-hidden"
            >
              <TeamSportBrowser
                onClose={handleTeamCollapse}
                showContent={showContent}
              />
            </motion.div>
          )}
        </AnimatePresence>
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
