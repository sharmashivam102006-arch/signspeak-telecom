import type { Config } from "tailwindcss";
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        ink: "#080B14", panel: "#101827", card: "#0F172A",
        cyan: { DEFAULT: "#22D3EE" }, violet: { DEFAULT: "#8B5CF6" },
        success: "#22C55E", muted: "#94A3B8",
      },
      fontFamily: { sans: ["Inter", "system-ui", "sans-serif"] },
    },
  },
  plugins: [],
};
export default config;
