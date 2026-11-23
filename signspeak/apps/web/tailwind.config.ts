import type { Config } from "tailwindcss";

export default {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "rgb(var(--bg) / <alpha-value>)",
        surface: "rgb(var(--surface) / <alpha-value>)",
        border: "rgb(var(--border) / <alpha-value>)",
        fg: "rgb(var(--fg) / <alpha-value>)",
        muted: "rgb(var(--muted) / <alpha-value>)",
        accent: "rgb(var(--accent) / <alpha-value>)",
        high: "rgb(var(--high) / <alpha-value>)",
        medium: "rgb(var(--medium) / <alpha-value>)",
        low: "rgb(var(--low) / <alpha-value>)",
      },
      fontSize: {
        // Caption scale. Driven by a user setting, not a breakpoint - the
        // person who needs bigger captions needs them on every screen size.
        "caption-sm": ["1rem", { lineHeight: "1.5rem" }],
        "caption-md": ["1.375rem", { lineHeight: "1.9rem" }],
        "caption-lg": ["1.875rem", { lineHeight: "2.5rem" }],
        "caption-xl": ["2.5rem", { lineHeight: "3.2rem" }],
      },
    },
  },
  plugins: [],
} satisfies Config;
