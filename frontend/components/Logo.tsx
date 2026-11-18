import Link from "next/link";
export function LogoMark({ size = 36 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 48 48" fill="none" aria-label="SignSpeak logo">
      <rect x="2" y="2" width="44" height="44" rx="12" fill="#101827" stroke="#22D3EE" strokeOpacity="0.35" />
      {/* open hand, abstract */}
      <path d="M17 30V18a2.5 2.5 0 0 1 5 0v-4a2.5 2.5 0 0 1 5 0v5a2.5 2.5 0 0 1 5 0v6a2.5 2.5 0 0 1 5 0v3c0 5-3.5 9-9 9-4 0-6-1.5-8-4l-3-4.5a2.4 2.4 0 0 1 0-2.5z" fill="#22D3EE" opacity="0.9" />
      {/* sound waves */}
      <path d="M35 15a8 8 0 0 1 0 18" stroke="#8B5CF6" strokeWidth="2.5" strokeLinecap="round" />
      <path d="M39 12a12 12 0 0 1 0 24" stroke="#8B5CF6" strokeWidth="2.5" strokeLinecap="round" opacity="0.6" />
    </svg>
  );
}
export function Logo() {
  return (
    <Link href="/" className="flex items-center gap-2.5">
      <LogoMark />
      <span className="text-lg font-bold tracking-tight">Sign<span className="text-cyan">Speak</span> <span className="text-muted font-medium text-sm">Telecom</span></span>
    </Link>
  );
}
