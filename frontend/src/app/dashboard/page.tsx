"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";
import { fetchWorkouts, fetchTrends } from "@/lib/dashboard";
import type { WorkoutSummary, TrendData, InjuryProfile } from "@/lib/types";
import Card from "@/components/ui/Card";
import DuoButton from "@/components/ui/DuoButton";
import UserBadge from "@/components/UserBadge";
import InjuryProfileCard from "@/components/InjuryProfileCard";
import GuidelinesPanel from "@/components/dashboard/GuidelinesPanel";
import WorkoutListItem from "@/components/dashboard/WorkoutListItem";
import TrendCharts from "@/components/dashboard/TrendCharts";

export default function DashboardPage() {
  const router = useRouter();
  const { user, loading: authLoading } = useAuth();

  const [workouts, setWorkouts] = useState<WorkoutSummary[]>([]);
  const [trends, setTrends] = useState<TrendData | null>(null);
  const [loadingData, setLoadingData] = useState(true);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    if (authLoading) return;
    if (!user) {
      router.push("/onboarding");
      return;
    }

    let cancelled = false;
    (async () => {
      try {
        const [wRes, tRes] = await Promise.all([
          fetchWorkouts(20, 0),
          fetchTrends(20),
        ]);
        if (cancelled) return;
        setWorkouts(wRes.workouts);
        setTotal(wRes.total);
        setTrends(tRes);
      } catch (e) {
        console.error("Dashboard fetch failed:", e);
      } finally {
        if (!cancelled) setLoadingData(false);
      }
    })();
    return () => { cancelled = true; };
  }, [user, authLoading, router]);

  if (authLoading || (!user && !authLoading)) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-[#AFAFAF] text-sm">Loading...</div>
      </div>
    );
  }

  const injuryProfile = user?.injury_profile as InjuryProfile | null | undefined;

  return (
    <div className="min-h-screen px-5 py-8 max-w-4xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <Link
          href="/"
          className="text-sm font-bold text-[#1CB0F6] hover:underline no-underline"
        >
          &larr; Home
        </Link>
        <h1 className="text-xl font-extrabold text-[#3C3C3C]">My Dashboard</h1>
        <UserBadge />
      </div>

      {/* Top section: Injury Profile + Trends */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Injury Profile */}
        <div>
          {injuryProfile && injuryProfile.injuries && injuryProfile.injuries.length > 0 ? (
            <div>
              <InjuryProfileCard profile={injuryProfile} confirmed />
              {/* Guidelines per injury */}
              <div className="mt-3 space-y-2">
                {injuryProfile.injuries.map((injury, i) => (
                  <div key={i} className="pl-3 border-l-2 border-[#E5E5E5]">
                    <div className="text-sm font-bold text-[#3C3C3C] capitalize">
                      {injury.type.replace("_", " ")}
                      {injury.side !== "unknown" && ` (${injury.side})`}
                    </div>
                    <GuidelinesPanel injury={injury} />
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <Card>
              <h3 className="text-base font-extrabold text-[#3C3C3C] mb-2">
                Injury Profile
              </h3>
              <p className="text-sm text-[#777777] mb-3">
                No injury profile set up yet.
              </p>
              <DuoButton
                variant="secondary"
                onClick={() => router.push("/onboarding?path=personal")}
              >
                Set Up Profile
              </DuoButton>
            </Card>
          )}
        </div>

        {/* Trends */}
        <div>
          {loadingData ? (
            <Card>
              <div className="text-sm text-[#AFAFAF] py-8 text-center">
                Loading trends...
              </div>
            </Card>
          ) : trends && trends.dates.length > 0 ? (
            <TrendCharts trends={trends} />
          ) : (
            <Card>
              <h3 className="text-base font-extrabold text-[#3C3C3C] mb-2">
                Trends
              </h3>
              <p className="text-sm text-[#777777]">
                Complete some workouts to see your trends.
              </p>
            </Card>
          )}
        </div>
      </div>

      {/* Past Workouts */}
      <div>
        <h2 className="text-lg font-extrabold text-[#3C3C3C] mb-3">
          Past Workouts
          {total > 0 && (
            <span className="text-sm font-normal text-[#AFAFAF] ml-2">
              ({total})
            </span>
          )}
        </h2>

        {loadingData ? (
          <Card>
            <div className="text-sm text-[#AFAFAF] py-8 text-center">
              Loading workouts...
            </div>
          </Card>
        ) : workouts.length === 0 ? (
          <Card>
            <div className="text-center py-8">
              <p className="text-sm text-[#777777] mb-3">
                No workouts yet. Start your first session!
              </p>
              <div className="flex gap-2 justify-center">
                <Link href="/analyze?mode=webcam" className="no-underline">
                  <DuoButton variant="primary">Use Webcam</DuoButton>
                </Link>
                <Link href="/analyze?mode=upload" className="no-underline">
                  <DuoButton variant="secondary">Upload Video</DuoButton>
                </Link>
              </div>
            </div>
          </Card>
        ) : (
          <div className="space-y-2">
            {workouts.map((w) => (
              <WorkoutListItem
                key={w.id}
                workout={w}
                onClick={() => router.push(`/dashboard/workout/${w.id}`)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
