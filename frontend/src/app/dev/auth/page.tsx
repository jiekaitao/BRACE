"use client";

import { useState } from "react";
import { useAuth } from "@/contexts/AuthContext";
import LoginForm from "@/components/LoginForm";
import DuoButton from "@/components/ui/DuoButton";
import Card from "@/components/ui/Card";
import { getToken } from "@/lib/auth";

export default function DevAuthPage() {
  const { user, loading, logout } = useAuth();
  const [showToken, setShowToken] = useState(false);

  return (
    <div className="max-w-lg mx-auto">
      <h1 className="text-2xl font-extrabold text-[#3C3C3C] mb-6">Auth Test</h1>

      {loading ? (
        <p className="text-[#AFAFAF]">Loading...</p>
      ) : user ? (
        <div className="flex flex-col gap-4">
          <Card>
            <h3 className="text-sm font-bold text-[#AFAFAF] mb-2">Current User</h3>
            <pre className="text-xs bg-[#F7F7F7] p-3 rounded-lg overflow-auto">
              {JSON.stringify(user, null, 2)}
            </pre>
          </Card>

          <Card>
            <h3 className="text-sm font-bold text-[#AFAFAF] mb-2">Session Token</h3>
            <button
              onClick={() => setShowToken(!showToken)}
              className="text-xs text-[#1CB0F6] hover:underline mb-2"
            >
              {showToken ? "Hide" : "Show"} token
            </button>
            {showToken && (
              <code className="text-[10px] bg-[#F7F7F7] p-2 rounded block break-all">
                {getToken()}
              </code>
            )}
          </Card>

          <DuoButton variant="danger" onClick={logout}>
            Log Out
          </DuoButton>
        </div>
      ) : (
        <LoginForm />
      )}
    </div>
  );
}
