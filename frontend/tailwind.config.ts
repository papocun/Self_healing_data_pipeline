import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        surface: "var(--surface)",
        border: "var(--border)",
        ink: "var(--ink)",
        muted: "var(--muted)",
        pos: "var(--pos)",
        neg: "var(--neg)",
        neu: "var(--neu)",
        warn: "var(--warn)",
        accent: "var(--accent)",
      },
    },
  },
  plugins: [],
};
export default config;
