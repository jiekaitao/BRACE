"use client";

import { Fragment, useState, useCallback, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { useAuth } from "@/contexts/AuthContext";
import { useChat } from "@/hooks/useChat";
import LoginForm from "@/components/LoginForm";
import ChatPanel from "@/components/ChatPanel";
import InjuryProfileCard from "@/components/InjuryProfileCard";
import DuoButton from "@/components/ui/DuoButton";
import DemoVideoModal from "@/components/DemoVideoModal";

const PERSONAL_STEPS = [
  { title: "Welcome", subtitle: "Sign in or create an account" },
  { title: "Injury History", subtitle: "Tell us about any past injuries" },
  { title: "Your Profile", subtitle: "Review your injury profile" },
  { title: "Get Started", subtitle: "Choose how to analyze your movement" },
];

const TEAM_STEPS = [
  { title: "Welcome", subtitle: "Sign in or create an account" },
  { title: "Get Started", subtitle: "Choose how to monitor your team" },
];

function OnboardingContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const path = searchParams.get("path") ?? "personal";

  const isTeam = path === "team";
  const STEPS = isTeam ? TEAM_STEPS : PERSONAL_STEPS;

  const { user } = useAuth();
  const {
    messages,
    loading: chatLoading,
    error: chatError,
    extractedProfile,
    profileComplete,
    sendMessage,
    confirmProfile,
  } = useChat(isTeam ? undefined : user?.user_id);

  const [step, setStep] = useState(0);
  const [showDemoModal, setShowDemoModal] = useState(false);
  const [profileConfirmed, setProfileConfirmed] = useState(false);

  // Map visual step index to logical step type
  const stepType = isTeam
    ? (["welcome", "mode"] as const)[step] ?? "mode"
    : (["welcome", "chat", "profile", "mode"] as const)[step] ?? "mode";

  const next = useCallback(
    () => setStep((s) => Math.min(s + 1, STEPS.length - 1)),
    [STEPS.length],
  );
  const back = useCallback(() => setStep((s) => Math.max(s - 1, 0)), []);

  const handleLoginSuccess = useCallback(() => {
    next();
  }, [next]);

  const handleSkipChat = useCallback(() => {
    // Jump to mode selection (last step)
    setStep(STEPS.length - 1);
  }, [STEPS.length]);

  const handleConfirmProfile = useCallback(async () => {
    await confirmProfile(user?.user_id);
    setProfileConfirmed(true);
    next();
  }, [confirmProfile, user, next]);

  const handleEditProfile = useCallback(() => {
    setStep(1);
  }, []);

  const handleModeSelect = useCallback(
    (mode: string, video?: string) => {
      if (video) {
        router.push(`/analyze?mode=demo&video=${encodeURIComponent(video)}`);
      } else {
        router.push(`/analyze?mode=${mode}`);
      }
    },
    [router],
  );

  const handleDemoSelect = useCallback(
    (filename: string) => {
      setShowDemoModal(false);
      handleModeSelect("demo", filename);
    },
    [handleModeSelect],
  );

  return (
    <div className="min-h-screen flex flex-col items-center px-5 py-8">
      {/* Back to home */}
      <div className="w-full max-w-md mb-4">
        <button
          onClick={() => router.push("/")}
          className="flex items-center gap-1 text-sm text-[#AFAFAF] hover:text-[#777777] transition-colors"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <polyline points="15 18 9 12 15 6" />
          </svg>
          Back to home
        </button>
      </div>

      {/* Progress bar */}
      <div className="flex items-center mb-8 w-full max-w-md">
        {STEPS.map((_, i) => (
          <Fragment key={i}>
            <div
              className={`w-8 h-8 rounded-full shrink-0 flex items-center justify-center text-sm font-bold transition-all duration-300 ${
                i <= step
                  ? "bg-[#58CC02] text-white"
                  : "bg-[#E5E5E5] text-[#AFAFAF]"
              }`}
            >
              {i < step ? (
                <svg width="14" height="14" viewBox="0 0 14 14" fill="none">
                  <path
                    d="M4 7l2.5 2.5L10 5"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              ) : (
                i + 1
              )}
            </div>
            {i < STEPS.length - 1 && (
              <div
                className={`h-1 flex-1 mx-2 rounded transition-colors duration-300 ${
                  i < step ? "bg-[#58CC02]" : "bg-[#E5E5E5]"
                }`}
              />
            )}
          </Fragment>
        ))}
      </div>

      {/* Step title */}
      <div className="text-center mb-6">
        <h1 className="text-2xl font-extrabold text-[#3C3C3C]">
          {STEPS[step].title}
        </h1>
        <p className="text-sm text-[#777777] mt-1">{STEPS[step].subtitle}</p>
      </div>

      {/* Step content */}
      <div className="w-full max-w-lg">
        {stepType === "welcome" && (
          <div className="fade-up">
            {user ? (
              <div className="text-center">
                <p className="text-base text-[#3C3C3C] mb-4">
                  Welcome back,{" "}
                  <span className="font-bold">{user.username}</span>!
                </p>
                <DuoButton variant="primary" onClick={next}>
                  Continue
                </DuoButton>
              </div>
            ) : (
              <div className="flex flex-col gap-4">
                <LoginForm onSuccess={handleLoginSuccess} />
              </div>
            )}
          </div>
        )}

        {stepType === "chat" && (
          <div className="fade-up">
            <div
              className="bg-white rounded-[16px] border-2 border-[#E5E5E5] overflow-hidden"
              style={{ height: 400 }}
            >
              <ChatPanel
                messages={messages}
                loading={chatLoading}
                error={chatError}
                onSend={sendMessage}
                placeholder="Tell me about any injuries..."
              />
            </div>
            <div className="flex justify-between mt-4">
              <DuoButton variant="secondary" onClick={back}>
                Back
              </DuoButton>
              <div className="flex gap-2">
                <button
                  onClick={handleSkipChat}
                  className="text-sm text-[#AFAFAF] hover:text-[#777777] transition-colors px-4 py-2"
                >
                  Skip
                </button>
                {(extractedProfile || profileComplete) && (
                  <DuoButton variant="primary" onClick={next}>
                    Continue
                  </DuoButton>
                )}
              </div>
            </div>
          </div>
        )}

        {stepType === "profile" && (
          <div className="fade-up">
            {extractedProfile ? (
              <InjuryProfileCard
                profile={extractedProfile}
                onConfirm={handleConfirmProfile}
                onEdit={handleEditProfile}
                confirmed={profileConfirmed}
              />
            ) : (
              <div className="text-center py-8">
                <p className="text-sm text-[#777777] mb-4">
                  No injury profile detected. You can go back to chat or skip
                  to start.
                </p>
              </div>
            )}
            <div className="flex justify-between mt-4">
              <DuoButton variant="secondary" onClick={back}>
                Back
              </DuoButton>
              {!extractedProfile && (
                <DuoButton variant="primary" onClick={next}>
                  Skip
                </DuoButton>
              )}
            </div>
          </div>
        )}

        {stepType === "mode" && (
          <div className="fade-up">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <button
                onClick={() => handleModeSelect("webcam")}
                className="flex flex-col items-center gap-2 p-6 rounded-[16px] border-2 border-[#E5E5E5] bg-white cursor-pointer shadow-[0_4px_0_#E5E5E5] hover:border-[#1CB0F6] hover:shadow-[0_4px_0_#1899D6] active:shadow-none active:translate-y-[4px] transition-all duration-100"
              >
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#1CB0F6"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M23 7l-7 5 7 5V7z" />
                  <rect x="1" y="5" width="15" height="14" rx="2" ry="2" />
                </svg>
                <span className="text-sm font-bold text-[#3C3C3C]">
                  Use Webcam
                </span>
                <span className="text-xs text-[#AFAFAF]">
                  Real-time analysis
                </span>
              </button>

              <button
                onClick={() => handleModeSelect("upload")}
                className="flex flex-col items-center gap-2 p-6 rounded-[16px] border-2 border-[#E5E5E5] bg-white cursor-pointer shadow-[0_4px_0_#E5E5E5] hover:border-[#58CC02] hover:shadow-[0_4px_0_#46A302] active:shadow-none active:translate-y-[4px] transition-all duration-100"
              >
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#58CC02"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
                <span className="text-sm font-bold text-[#3C3C3C]">
                  Upload Video
                </span>
                <span className="text-xs text-[#AFAFAF]">
                  Analyze recorded footage
                </span>
              </button>

              <button
                onClick={() => setShowDemoModal(true)}
                className="flex flex-col items-center gap-2 p-6 rounded-[16px] border-2 border-[#E5E5E5] bg-white cursor-pointer shadow-[0_4px_0_#E5E5E5] hover:border-[#CE82FF] hover:shadow-[0_4px_0_#B060E0] active:shadow-none active:translate-y-[4px] transition-all duration-100 sm:col-span-2"
              >
                <svg
                  width="32"
                  height="32"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="#CE82FF"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
                <span className="text-sm font-bold text-[#3C3C3C]">
                  Demo Videos
                </span>
                <span className="text-xs text-[#AFAFAF]">
                  Try with pre-loaded videos
                </span>
              </button>
            </div>

            <div className="flex justify-start mt-4">
              <DuoButton variant="secondary" onClick={back}>
                Back
              </DuoButton>
            </div>
          </div>
        )}
      </div>

      {/* Demo video modal */}
      {showDemoModal && (
        <DemoVideoModal
          onSelect={handleDemoSelect}
          onClose={() => setShowDemoModal(false)}
          filter={path === "team" ? "team" : "personal"}
        />
      )}
    </div>
  );
}

export default function OnboardingPage() {
  return (
    <Suspense
      fallback={
        <div className="min-h-screen flex items-center justify-center text-[#AFAFAF]">
          Loading...
        </div>
      }
    >
      <OnboardingContent />
    </Suspense>
  );
}
