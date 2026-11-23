import { ButtonHTMLAttributes, forwardRef } from "react";
type Props = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost" | "danger" | "outline"; size?: "sm" | "md" | "lg" };
const styles: Record<string, string> = {
  primary: "bg-cyan text-ink hover:bg-cyan-300 font-semibold",
  secondary: "bg-violet text-white hover:bg-violet-500 font-semibold",
  ghost: "bg-transparent text-slate-200 hover:bg-white/10",
  danger: "bg-red-500/90 text-white hover:bg-red-500 font-semibold",
  outline: "border border-slate-700 text-slate-100 hover:border-cyan/60 hover:text-cyan bg-transparent",
};
export const Button = forwardRef<HTMLButtonElement, Props>(function Button({ variant = "primary", size = "md", className = "", ...p }, ref) {
  const sz = size === "sm" ? "px-3 py-1.5 text-sm" : size === "lg" ? "px-6 py-3 text-base" : "px-4 py-2.5 text-sm";
  return <button ref={ref} className={`inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl transition disabled:opacity-50 disabled:cursor-not-allowed ${styles[variant]} ${sz} ${className}`} {...p} />;
});
