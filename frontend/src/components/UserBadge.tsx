"use client";

import Link from "next/link";
import { useAuth } from "@/contexts/AuthContext";

export default function UserBadge() {
  const { user, loading, logout } = useAuth();

  if (loading || !user) return null;

  return (
    <div className="flex items-center gap-2">
      <Link
        href="/dashboard"
        className="w-7 h-7 rounded-full bg-[#DDF4FF] flex items-center justify-center no-underline hover:ring-2 hover:ring-[#1CB0F6] transition-shadow"
        title="Dashboard"
      >
        <span className="text-xs font-bold text-[#1CB0F6]">
          {user.username[0].toUpperCase()}
        </span>
      </Link>
      <span className="text-sm font-bold text-[#3C3C3C]">{user.username}</span>
      <button
        onClick={logout}
        className="text-xs text-[#AFAFAF] hover:text-[#EA2B2B] transition-colors font-bold ml-1"
      >
        Log out
      </button>
    </div>
  );
}
