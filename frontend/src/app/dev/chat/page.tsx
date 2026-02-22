"use client";

// import { useAuth } from "@/contexts/AuthContext";
// import { useChat } from "@/hooks/useChat";
// import ChatPanel from "@/components/ChatPanel";
// import InjuryProfileCard from "@/components/InjuryProfileCard";
import Card from "@/components/ui/Card";
// import DuoButton from "@/components/ui/DuoButton";

export default function DevChatPage() {
  // LLM chat disabled to avoid Gemini API abuse on public demo.
  // To re-enable, uncomment the imports above and restore the original JSX.
  //
  // const { user } = useAuth();
  // const { messages, loading, error, extractedProfile, profileComplete, sendMessage, confirmProfile, reset } = useChat(user?.user_id);

  return (
    <div className="max-w-2xl mx-auto">
      <h1 className="text-2xl font-extrabold text-[#3C3C3C] mb-6">Chat Test</h1>
      <Card className="flex items-center justify-center py-16">
        <div className="text-center">
          <p className="text-sm font-bold text-[#3C3C3C] mb-1">Disabled</p>
          <p className="text-xs text-[#AFAFAF]">
            LLM chat is disabled on the public demo to prevent API abuse.
          </p>
        </div>
      </Card>
    </div>
  );
}
