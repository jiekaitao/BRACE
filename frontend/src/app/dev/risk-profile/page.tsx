"use client";

// Backend API calls disabled to avoid abuse on public demo.
// To re-enable, uncomment the fetch calls and useEffect below.

// import { useEffect, useState } from "react";
// import { useAuth } from "@/contexts/AuthContext";
import Card from "@/components/ui/Card";
// import DuoButton from "@/components/ui/DuoButton";
// import { getApiBase } from "@/lib/api";
// import { getToken } from "@/lib/auth";

// const INJURY_TYPES = ["acl", "shoulder", "ankle", "lower_back", "knee_general", "hip", "hamstring"];
// const SEVERITIES = ["mild", "moderate", "severe"];
// const SIDES = ["left", "right", "bilateral"];

export default function DevRiskProfilePage() {
  // const { user } = useAuth();
  // const [injuries, setInjuries] = useState<Array<{ type: string; side: string; severity: string }>>([]);
  // const [modifiers, setModifiers] = useState<Record<string, unknown> | null>(null);
  // const [saving, setSaving] = useState(false);
  //
  // const addInjury = () => {
  //   setInjuries([...injuries, { type: "acl", side: "left", severity: "moderate" }]);
  // };
  //
  // const removeInjury = (idx: number) => {
  //   setInjuries(injuries.filter((_, i) => i !== idx));
  // };
  //
  // const updateInjury = (idx: number, field: string, value: string) => {
  //   const updated = [...injuries];
  //   updated[idx] = { ...updated[idx], [field]: value };
  //   setInjuries(updated);
  // };
  //
  // const saveProfile = async () => {
  //   setSaving(true);
  //   try {
  //     const profile = { injuries: injuries.map((i) => ({ ...i, timeframe: "chronic" })) };
  //     const res = await fetch(`${getApiBase()}/api/chat/confirm-profile`, {
  //       method: "POST",
  //       headers: {
  //         "Content-Type": "application/json",
  //         ...(getToken() ? { Authorization: `Bearer ${getToken()}` } : {}),
  //       },
  //       body: JSON.stringify({
  //         user_id: user?.user_id || undefined,
  //         injury_profile: profile,
  //       }),
  //     });
  //     const data = await res.json();
  //     setModifiers(data.risk_modifiers);
  //   } catch (e) {
  //     console.error(e);
  //   } finally {
  //     setSaving(false);
  //   }
  // };
  //
  // // Auto-compute modifiers on injury change
  // useEffect(() => {
  //   if (injuries.length === 0) {
  //     setModifiers(null);
  //     return;
  //   }
  //   const profile = { injuries: injuries.map((i) => ({ ...i, timeframe: "chronic" })) };
  //   fetch(`${getApiBase()}/api/chat/confirm-profile`, {
  //     method: "POST",
  //     headers: { "Content-Type": "application/json" },
  //     body: JSON.stringify({ injury_profile: profile }),
  //   })
  //     .then((r) => r.json())
  //     .then((data) => setModifiers(data.risk_modifiers))
  //     .catch(() => {});
  // }, [injuries]);

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-extrabold text-[#3C3C3C] mb-6">Risk Profile Test</h1>
      <Card className="flex items-center justify-center py-16">
        <div className="text-center">
          <p className="text-sm font-bold text-[#3C3C3C] mb-1">Disabled</p>
          <p className="text-xs text-[#AFAFAF]">
            Backend API calls are disabled on the public demo to prevent abuse.
          </p>
        </div>
      </Card>
    </div>
  );
}
