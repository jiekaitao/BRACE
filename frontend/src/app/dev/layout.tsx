"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const DEV_PAGES = [
  { href: "/dev/auth", label: "Auth" },
  { href: "/dev/chat", label: "Chat" },
  { href: "/dev/components", label: "Components" },
  { href: "/dev/risk-profile", label: "Risk Profile" },
  { href: "/dev/streams", label: "Streams" },
  { href: "/dev/voice", label: "Voice" },
  { href: "/dev/timeline", label: "Timeline" },
];

export default function DevLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();

  return (
    <div className="min-h-screen flex">
      {/* Sidebar */}
      <aside className="w-48 bg-[#F7F7F7] border-r-2 border-[#E5E5E5] p-4 flex-shrink-0">
        <Link href="/" className="text-sm font-bold text-[#AFAFAF] hover:text-[#3C3C3C] mb-4 block">
          &larr; Home
        </Link>
        <h2 className="text-xs font-bold text-[#AFAFAF] uppercase tracking-wider mb-3">
          Dev Pages
        </h2>
        <nav className="flex flex-col gap-1">
          {DEV_PAGES.map((page) => (
            <Link
              key={page.href}
              href={page.href}
              className={`
                px-3 py-2 rounded-lg text-sm font-bold transition-colors
                ${pathname === page.href
                  ? "bg-[#1CB0F6] text-white"
                  : "text-[#4B4B4B] hover:bg-[#E5E5E5]"
                }
              `}
            >
              {page.label}
            </Link>
          ))}
        </nav>
      </aside>

      {/* Content */}
      <main className="flex-1 p-6 overflow-auto">
        {children}
      </main>
    </div>
  );
}
