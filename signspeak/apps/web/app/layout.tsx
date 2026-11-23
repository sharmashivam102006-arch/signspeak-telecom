import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SignSpeak Telecom",
  description:
    "An accessibility layer for live calls: sign recognition, live captions and speech synthesis in the same conversation.",
};

export const viewport: Viewport = {
  // userScalable is NOT disabled. Blocking pinch-zoom is a common and serious
  // accessibility failure, and doubly wrong in a product built for people who
  // may need to magnify the screen.
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" data-theme="dark" data-contrast="normal" data-motion="normal">
      <body className="min-h-dvh antialiased">{children}</body>
    </html>
  );
}
