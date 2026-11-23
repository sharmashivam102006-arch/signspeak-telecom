import { ReactNode } from "react";
export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <div className={`rounded-2xl border border-slate-800 bg-panel/80 p-5 ${className}`}>{children}</div>;
}
export function Badge({ children, tone = "cyan" }: { children: ReactNode; tone?: "cyan" | "violet" | "green" | "amber" | "red" | "slate" }) {
  const map: Record<string, string> = {
    cyan: "bg-cyan/10 text-cyan border-cyan/30", violet: "bg-violet/10 text-violet-300 border-violet/30",
    green: "bg-success/10 text-green-400 border-green-500/30", amber: "bg-amber-500/10 text-amber-300 border-amber-500/30",
    red: "bg-red-500/10 text-red-300 border-red-500/30", slate: "bg-slate-700/40 text-slate-300 border-slate-600/40",
  };
  return <span className={`inline-flex items-center gap-1 rounded-full border px-2.5 py-0.5 text-xs font-medium ${map[tone]}`}>{children}</span>;
}
