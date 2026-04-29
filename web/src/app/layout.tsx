import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "NBA Win Probability",
  description: "Real-time NBA win probability model",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-zinc-950 text-zinc-200">
        {/* Nav */}
        <nav className="border-b border-white/5 bg-zinc-950/80 backdrop-blur-sm sticky top-0 z-50">
          <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
            <Link href="/" className="font-bold text-white tracking-tight text-lg">
              NBA Win Probability
            </Link>
            <div className="flex gap-6 text-sm">
              <Link href="/" className="text-gray-400 hover:text-white transition-colors">
                Games
              </Link>
              <Link href="/backtest" className="text-gray-400 hover:text-white transition-colors">
                Backtest
              </Link>
            </div>
          </div>
        </nav>

        {/* Main content */}
        <main className="flex-1">
          <div className="max-w-7xl mx-auto px-6 py-8">
            {children}
          </div>
        </main>
      </body>
    </html>
  );
}
