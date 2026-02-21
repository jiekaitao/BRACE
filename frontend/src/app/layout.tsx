import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/contexts/AuthContext";
import UserBadge from "@/components/UserBadge";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "BRACE",
  description: "Real-time movement consistency analysis",
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
          <header className="flex items-center justify-end px-5 py-3 border-b border-[#E5E5E5]">
            <UserBadge />
          </header>
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
