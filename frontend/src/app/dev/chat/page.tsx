"use client";

import { useAuth } from "@/contexts/AuthContext";
import { useChat } from "@/hooks/useChat";
import ChatPanel from "@/components/ChatPanel";
import InjuryProfileCard from "@/components/InjuryProfileCard";
import Card from "@/components/ui/Card";
import DuoButton from "@/components/ui/DuoButton";

export default function DevChatPage() {
  const { user } = useAuth();
  const { messages, loading, error, extractedProfile, profileComplete, sendMessage, confirmProfile, reset } = useChat(user?.user_id);

  return (
    <div className="max-w-2xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-extrabold text-[#3C3C3C]">Chat Test</h1>
        <DuoButton variant="secondary" onClick={reset}>
          Reset
        </DuoButton>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Chat panel */}
        <div className="bg-white rounded-[16px] border-2 border-[#E5E5E5] overflow-hidden" style={{ height: 500 }}>
          <ChatPanel
            messages={messages}
            loading={loading}
            error={error}
            onSend={sendMessage}
          />
        </div>

        {/* Profile output */}
        <div className="flex flex-col gap-4">
          {extractedProfile && (
            <InjuryProfileCard
              profile={extractedProfile}
              onConfirm={() => confirmProfile(user?.user_id)}
            />
          )}

          <Card>
            <h3 className="text-sm font-bold text-[#AFAFAF] mb-2">Raw Response</h3>
            <pre className="text-[10px] bg-[#F7F7F7] p-3 rounded-lg overflow-auto max-h-60">
              {JSON.stringify({ extractedProfile, profileComplete, messageCount: messages.length }, null, 2)}
            </pre>
          </Card>
        </div>
      </div>
    </div>
  );
}
