"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import DuoButton from "@/components/ui/DuoButton";
import Card from "@/components/ui/Card";

interface LoginFormProps {
  onSuccess?: () => void;
}

export default function LoginForm({ onSuccess }: LoginFormProps) {
  const { login } = useAuth();
  const [username, setUsername] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleContinue = async () => {
    if (!username.trim()) {
      setError("Please enter a username");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      await login(username.trim());
      onSuccess?.();
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card className="max-w-sm mx-auto">
      <div className="flex flex-col gap-4">
        <h2 className="text-xl font-extrabold text-[#3C3C3C] text-center">
          Welcome to BRACE
        </h2>
        <p className="text-sm text-[#777777] text-center">
          Enter a username to get started
        </p>
        <input
          type="text"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleContinue()}
          placeholder="Your username"
          className="w-full px-4 py-3 text-base border-2 border-[#E5E5E5] rounded-[12px] outline-none focus:border-[#1CB0F6] transition-colors"
          disabled={loading}
          autoFocus
        />
        {error && (
          <p className="text-sm text-[#EA2B2B] text-center">{error}</p>
        )}
        <DuoButton
          variant="primary"
          fullWidth
          onClick={handleContinue}
          disabled={loading}
        >
          {loading ? "..." : "Continue"}
        </DuoButton>
      </div>
    </Card>
  );
}
