"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import Card from "@/components/ui/Card";
import DuoButton from "@/components/ui/DuoButton";
import { getApiBase } from "@/lib/api";
import { getToken } from "@/lib/auth";

const INJURY_TYPES = ["acl", "shoulder", "ankle", "lower_back", "knee_general", "hip", "hamstring"];
const SEVERITIES = ["mild", "moderate", "severe"];
const SIDES = ["left", "right", "bilateral"];

export default function DevRiskProfilePage() {
  const { user } = useAuth();
  const [injuries, setInjuries] = useState<Array<{ type: string; side: string; severity: string }>>([]);
  const [modifiers, setModifiers] = useState<Record<string, unknown> | null>(null);
  const [saving, setSaving] = useState(false);

  const addInjury = () => {
    setInjuries([...injuries, { type: "acl", side: "left", severity: "moderate" }]);
  };

  const removeInjury = (idx: number) => {
    setInjuries(injuries.filter((_, i) => i !== idx));
  };

  const updateInjury = (idx: number, field: string, value: string) => {
    const updated = [...injuries];
    updated[idx] = { ...updated[idx], [field]: value };
    setInjuries(updated);
  };

  const saveProfile = async () => {
    setSaving(true);
    try {
      const profile = { injuries: injuries.map((i) => ({ ...i, timeframe: "chronic" })) };
      const res = await fetch(`${getApiBase()}/api/chat/confirm-profile`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(getToken() ? { Authorization: `Bearer ${getToken()}` } : {}),
        },
        body: JSON.stringify({
          user_id: user?.user_id || undefined,
          injury_profile: profile,
        }),
      });
      const data = await res.json();
      setModifiers(data.risk_modifiers);
    } catch (e) {
      console.error(e);
    } finally {
      setSaving(false);
    }
  };

  // Auto-compute modifiers on injury change
  useEffect(() => {
    if (injuries.length === 0) {
      setModifiers(null);
      return;
    }
    const profile = { injuries: injuries.map((i) => ({ ...i, timeframe: "chronic" })) };
    fetch(`${getApiBase()}/api/chat/confirm-profile`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ injury_profile: profile }),
    })
      .then((r) => r.json())
      .then((data) => setModifiers(data.risk_modifiers))
      .catch(() => {});
  }, [injuries]);

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-extrabold text-[#3C3C3C] mb-6">Risk Profile Test</h1>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Injury editor */}
        <Card>
          <h3 className="text-sm font-bold text-[#3C3C3C] mb-3">Injuries</h3>
          {injuries.map((injury, i) => (
            <div key={i} className="flex gap-2 mb-2 items-center">
              <select
                value={injury.type}
                onChange={(e) => updateInjury(i, "type", e.target.value)}
                className="text-xs border rounded px-2 py-1 flex-1"
              >
                {INJURY_TYPES.map((t) => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
              <select
                value={injury.side}
                onChange={(e) => updateInjury(i, "side", e.target.value)}
                className="text-xs border rounded px-2 py-1"
              >
                {SIDES.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
              <select
                value={injury.severity}
                onChange={(e) => updateInjury(i, "severity", e.target.value)}
                className="text-xs border rounded px-2 py-1"
              >
                {SEVERITIES.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
              <button onClick={() => removeInjury(i)} className="text-[#EA2B2B] text-xs font-bold">
                X
              </button>
            </div>
          ))}
          <div className="flex gap-2 mt-3">
            <DuoButton variant="secondary" onClick={addInjury}>
              + Add Injury
            </DuoButton>
            {user && (
              <DuoButton variant="primary" onClick={saveProfile} disabled={saving}>
                {saving ? "Saving..." : "Save to Profile"}
              </DuoButton>
            )}
          </div>
        </Card>

        {/* Modifiers display */}
        <Card>
          <h3 className="text-sm font-bold text-[#3C3C3C] mb-3">Threshold Multipliers</h3>
          {modifiers ? (
            <div className="flex flex-col gap-2">
              {Object.entries(modifiers).map(([key, value]) => (
                <div key={key} className="flex justify-between items-center text-xs">
                  <span className="text-[#777777]">{key}</span>
                  <span className={`font-bold ${typeof value === 'number' && value < 1 ? "text-[#EA2B2B]" : "text-[#58CC02]"}`}>
                    {typeof value === 'number' ? value.toFixed(2) : JSON.stringify(value)}
                  </span>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-xs text-[#AFAFAF]">No injuries configured -- all thresholds at default (1.0)</p>
          )}
        </Card>
      </div>
    </div>
  );
}
