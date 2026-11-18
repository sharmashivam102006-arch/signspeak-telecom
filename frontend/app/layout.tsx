import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignSpeak Telecom — Communication Without Barriers",
  description: "One Call. Two Languages. No Communication Barrier. AI-powered bidirectional sign-language and speech communication in a video call.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-ink text-slate-50 antialiased">{children}</body>
    </html>
  );
}
