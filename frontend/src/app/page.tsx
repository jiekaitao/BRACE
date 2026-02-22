"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { motion, AnimatePresence } from "framer-motion";
import DuoButton from "@/components/ui/DuoButton";
import PathCard from "@/components/ui/PathCard";
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

  // Compute animated values: start at card position, expand to fill container
  const collapsedStyle = cardRect
    ? { top: cardRect.top, left: cardRect.left, width: cardRect.width, height: cardRect.height }
    : { top: 0, left: 0, width: "100%" as string | number, height: "100%" as string | number };

  const expandedStyle = { top: 0, left: 0, width: "100%", height: "110%" };

  return (
    <div className="min-h-screen flex flex-col items-center px-5 py-12">
      {/* Hero */}
      <div className="scale-in text-center mb-8 max-w-2xl">
        <p className="text-xl sm:text-2xl font-bold text-[#3C3C3C] leading-[1.3]">
          Real-time AI-powered movement analysis
          <br />
          to keep you and your team <span className="underline-draw">safe</span>.
        </p>
      </div>

      {/* Content area - relative container for the expanding overlay */}
      <div ref={contentRef} className="relative w-full max-w-2xl mb-8">
        {/* Path Cards - always in DOM */}
        <div
          className="grid grid-cols-1 sm:grid-cols-2 gap-4 w-full mb-4 fade-up"
          style={{ animationDelay: "0.55s" }}
        >
          <PathCard
            title="Personal Workout Safety"
            subtitle="Real-time injury risk detection for your training"
            onClick={() => router.push("/onboarding?path=personal")}
            image="https://images.unsplash.com/photo-1614634053434-1729f6ac6bd6?q=80&w=2747&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
          />

          <motion.div
            ref={teamCardRef}
            initial={false}
            animate={{ opacity: cardRect ? 0 : 1 }}
            transition={{ duration: 0.6, ease: "easeOut" }}
          >
            <PathCard
              title="Team Performance Monitor"
              subtitle="Track and compare athletes across your roster"
              onClick={handleTeamExpand}
              image="https://images.unsplash.com/photo-1529478562208-d4c746edcb79?q=80&w=1287&auto=format&fit=crop&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
            />
          </motion.div>
        </div>

        {/* Quick access CTAs */}
        <div
          className="flex flex-col sm:flex-row gap-3 w-full fade-up"
          style={{ animationDelay: "0.65s" }}
        >
          <Link href="/vectorai" className="flex-1 no-underline">
            <DuoButton variant="primary" fullWidth>
              VectorAI Store
            </DuoButton>
          </Link>
          <Link href="/analyze?mode=webcam" className="flex-1 no-underline">
            <DuoButton variant="blue" fullWidth>
              Use Webcam
            </DuoButton>
          </Link>
          <Link href="/dev/streams" className="flex-1 no-underline">
            <DuoButton variant="danger" fullWidth>
              Debug Streams
            </DuoButton>
          </Link>
        </div>

        {/* Tech stack link */}
        <div
          className="w-full fade-up mt-3"
          style={{ animationDelay: "0.75s" }}
        >
          <Link href="/stack" className="no-underline">
            <DuoButton variant="secondary" fullWidth>
              Learn about the tech stack
            </DuoButton>
          </Link>
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
              className="absolute z-10 rounded-[20px] border-2 border-[#333] shadow-[0_6px_0_#222] bg-[#0A0A0A] overflow-hidden"
            >
              <TeamSportBrowser
                onClose={handleTeamCollapse}
                showContent={showContent}
              />
            </motion.div>
          )}
        </AnimatePresence>
      </div>

    </div>
  );
}
