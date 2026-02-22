"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";
import type { InjuryProfile, RiskModifiers } from "@/lib/types";
import InjuryProfileCard from "@/components/InjuryProfileCard";
import GeminiResearchPanel from "@/components/GeminiResearchPanel";
import AnimatedSkeletonDemo from "@/components/AnimatedSkeletonDemo";
import DuoButton from "@/components/ui/DuoButton";
import DemoVideoModal from "@/components/DemoVideoModal";
import WorkoutListItem from "@/components/dashboard/WorkoutListItem";
import TrendCharts from "@/components/dashboard/TrendCharts";
import GuidelinesPanel from "@/components/dashboard/GuidelinesPanel";
import { fetchWorkouts, type WorkoutSummary } from "@/lib/dashboard";

export default function DashboardPage() {
  const router = useRouter();
  const { user, loading } = useAuth();
  const [showDemoModal, setShowDemoModal] = useState(false);
  const [workouts, setWorkouts] = useState<WorkoutSummary[]>([]);

  useEffect(() => {
    if (!loading && (!user || !user.injury_profile)) {
      router.replace("/onboarding?path=personal");
    }
  }, [loading, user, router]);

  useEffect(() => {
    if (user?.user_id) {
      fetchWorkouts(user.user_id, 5).then(setWorkouts).catch(() => {});
    }
  }, [user]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center text-[#AFAFAF]">
        Loading...
      </div>
    );
  }

  if (!user || !user.injury_profile) {
    return null; // redirect in progress
  }

  const injuryProfile = user.injury_profile as unknown as InjuryProfile;
  const riskModifiers = (user.risk_modifiers as unknown as RiskModifiers) ?? null;

  return (
    <div className="min-h-screen flex flex-col items-center px-5 py-8">
      {/* Header */}
      <div className="w-full max-w-3xl mb-6">
        <h1 className="text-2xl font-extrabold text-[#3C3C3C]">
          Welcome back, {user.username}
        </h1>
        <p className="text-sm text-[#777777] mt-1">
          Your personalized movement safety dashboard
        </p>
      </div>

      {/* Two-column grid: Injury Profile + Research */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full max-w-3xl mb-4">
        {/* Section A: Injury Profile */}
        <div className="flex flex-col gap-3">
          <InjuryProfileCard profile={injuryProfile} confirmed />
          {riskModifiers && (
            <div className="px-4 py-3 bg-[#F7F7F7] rounded-[12px] border border-[#E5E5E5]">
              <p className="text-xs font-bold text-[#3C3C3C] mb-1 uppercase tracking-wider">
                Risk Sensitivity
              </p>
              <div className="flex flex-wrap gap-2">
                {riskModifiers.fppa_scale < 1.0 && (
                  <span className="text-xs text-[#777777]">
                    FPPA: {Math.round(riskModifiers.fppa_scale * 100)}%
                  </span>
                )}
                {riskModifiers.hip_drop_scale < 1.0 && (
                  <span className="text-xs text-[#777777]">
                    Hip drop: {Math.round(riskModifiers.hip_drop_scale * 100)}%
                  </span>
                )}
                {riskModifiers.trunk_lean_scale < 1.0 && (
                  <span className="text-xs text-[#777777]">
                    Trunk lean: {Math.round(riskModifiers.trunk_lean_scale * 100)}%
                  </span>
                )}
                {riskModifiers.asymmetry_scale < 1.0 && (
                  <span className="text-xs text-[#777777]">
                    Asymmetry: {Math.round(riskModifiers.asymmetry_scale * 100)}%
                  </span>
                )}
                {riskModifiers.angular_velocity_scale < 1.0 && (
                  <span className="text-xs text-[#777777]">
                    Angular vel: {Math.round(riskModifiers.angular_velocity_scale * 100)}%
                  </span>
                )}
              </div>
            </div>
          )}
          <DuoButton
            variant="secondary"
            onClick={() => router.push("/onboarding?path=personal&edit=true")}
          >
            Edit Profile
          </DuoButton>
        </div>

        {/* Section B: Gemini Research */}
        <div>
          <GeminiResearchPanel />
        </div>
      </div>

      {/* Section C: Animated Skeleton Demo */}
      <div className="w-full max-w-3xl mb-6">
        <AnimatedSkeletonDemo
          injuryProfile={injuryProfile}
          riskModifiers={riskModifiers}
        />
      </div>

      {/* Quick Actions */}
      <div className="flex flex-col sm:flex-row gap-3 w-full max-w-3xl">
        <Link href="/analyze?mode=webcam" className="flex-1 no-underline">
          <DuoButton variant="blue" fullWidth>
            Use Webcam
          </DuoButton>
        </Link>
        <Link href="/analyze?mode=upload" className="flex-1 no-underline">
          <DuoButton variant="primary" fullWidth>
            Upload Video
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

      {/* Section D: Recent Workouts */}
      <div className="w-full max-w-3xl mb-6">
        <p className="text-xs font-bold text-[#3C3C3C] mb-3 uppercase tracking-wider">
          Recent Workouts
        </p>
        {workouts.length === 0 ? (
          <p className="text-xs text-[#AFAFAF]">No workouts yet. Start your first session above.</p>
        ) : (
          <div className="flex flex-col gap-2">
            {workouts.map((w) => (
              <WorkoutListItem key={w._id} workout={w} />
            ))}
          </div>
        )}
      </div>

      {/* Section E: Trends + Guidelines */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 w-full max-w-3xl mb-6">
        <TrendCharts userId={user.user_id} />
        <GuidelinesPanel userId={user.user_id} />
      </div>

      {showDemoModal && (
        <DemoVideoModal
          onSelect={(filename) => {
            setShowDemoModal(false);
            router.push(
              `/analyze?mode=demo&video=${encodeURIComponent(filename)}`,
            );
          }}
          onClose={() => setShowDemoModal(false)}
        />
      )}
    </div>
  );
}
