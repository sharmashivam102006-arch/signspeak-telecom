import { InputHTMLAttributes, forwardRef } from "react";
export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input(p, ref) {
  return <input ref={ref} {...p} className={`w-full rounded-xl border border-slate-700 bg-panel px-4 py-2.5 text-sm text-slate-100 placeholder:text-muted outline-none focus:border-cyan/70 focus:ring-1 focus:ring-cyan/40 ${p.className || ""}`} />;
});
export function Label({ children, htmlFor }: { children: React.ReactNode; htmlFor?: string }) {
  return <label htmlFor={htmlFor} className="mb-1.5 block text-sm font-medium text-slate-300">{children}</label>;
}
