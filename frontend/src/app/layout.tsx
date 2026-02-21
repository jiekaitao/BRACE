import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/contexts/AuthContext";
import UserBadge from "@/components/UserBadge";
import Image from "next/image";
import Link from "next/link";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "BRACE",
  description: "Real-time movement consistency analysis",
  icons: {
    icon: "/favicon.ico",
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <AuthProvider>
          <header className="flex items-center justify-between px-5 py-3 border-b border-[#E5E5E5]">
            <Link href="/" className="flex items-center gap-2 no-underline">
              <Image src="/logo.png" alt="BRACE" width={28} height={42} className="object-contain" />
              <span className="text-lg font-bold text-[#3C3C3C]">BRACE</span>
            </Link>
            <UserBadge />
          </header>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
