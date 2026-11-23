"use client";

/**
 * Accessibility settings.
 *
 * These live in the top bar, one click away, not buried in a profile page.
 * A user whose captions are too small needs them bigger NOW, mid-call, not
 * after signing up and finding a settings screen.
 *
 * Settings persist to localStorage immediately and apply to the document root,
 * so a reload mid-call does not reset them.
 */

import { useEffect, useState } from "react";

type Size = "small" | "medium" | "large" | "xlarge";

interface Prefs {
  captionSize: Size;
  highContrast: boolean;
  reducedMotion: boolean;
  theme: "dark" | "light";
}

const DEFAULTS: Prefs = {
  captionSize: "medium",
  highContrast: false,
  reducedMotion: false,
  theme: "dark",
};

function load(): Prefs {
  if (typeof window === "undefined") return DEFAULTS;
  try {
    return { ...DEFAULTS, ...JSON.parse(localStorage.getItem("signspeak:a11y") ?? "{}") };
  } catch {
    return DEFAULTS;
  }
}

export default function AccessibilityMenu() {
  const [open, setOpen] = useState(false);
  const [prefs, setPrefs] = useState<Prefs>(DEFAULTS);

  useEffect(() => setPrefs(load()), []);

  useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = prefs.theme;
    root.dataset.contrast = prefs.highContrast ? "high" : "normal";
    root.dataset.motion = prefs.reducedMotion ? "reduced" : "normal";
    root.dataset.captionSize = prefs.captionSize;
    try {
      localStorage.setItem("signspeak:a11y", JSON.stringify(prefs));
    } catch { /* private mode */ }
  }, [prefs]);

  const set = <K extends keyof Prefs>(k: K, v: Prefs[K]) => setPrefs((p) => ({ ...p, [k]: v }));

  return (
    <div className="relative">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-expanded={open}
        aria-haspopup="dialog"
        className="rounded-lg border border-border px-3 py-1.5 text-sm hover:border-accent"
      >
        Display
      </button>

      {open && (
        <div
          role="dialog"
          aria-label="Display and accessibility settings"
          className="absolute right-0 z-20 mt-2 w-72 rounded-xl border border-border bg-surface p-4 shadow-xl"
        >
          <fieldset className="mb-4">
            <legend className="mb-2 text-sm font-semibold">Caption size</legend>
            <div className="grid grid-cols-4 gap-1">
              {(["small", "medium", "large", "xlarge"] as Size[]).map((s) => (
                <button
                  key={s}
                  onClick={() => set("captionSize", s)}
                  aria-pressed={prefs.captionSize === s}
                  className={`rounded border px-2 py-1 text-xs capitalize ${
                    prefs.captionSize === s ? "border-accent text-accent" : "border-border"
                  }`}
                >
                  {s === "xlarge" ? "XL" : s[0].toUpperCase()}
                </button>
              ))}
            </div>
          </fieldset>

          <Toggle label="High contrast" checked={prefs.highContrast}
                  onChange={(v) => set("highContrast", v)}
                  hint="Pure black and white, WCAG AAA colour pairs" />
          <Toggle label="Reduce motion" checked={prefs.reducedMotion}
                  onChange={(v) => set("reducedMotion", v)}
                  hint="Removes animation; captions still update instantly" />
          <Toggle label="Light theme" checked={prefs.theme === "light"}
                  onChange={(v) => set("theme", v ? "light" : "dark")}
                  hint="Some low-vision users read dark-on-light more easily" />
        </div>
      )}
    </div>
  );
}

function Toggle({
  label, checked, onChange, hint,
}: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="mb-3 flex cursor-pointer items-start gap-3">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        className="mt-1 h-4 w-4 accent-[rgb(var(--accent))]"
      />
      <span>
        <span className="block text-sm font-medium">{label}</span>
        {hint && <span className="block text-xs text-muted">{hint}</span>}
      </span>
    </label>
  );
}
