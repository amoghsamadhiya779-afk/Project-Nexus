import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#030712",
        secondaryBg: "#0F172A",
        surface: "rgba(255, 255, 255, 0.04)",
        borderCustom: "rgba(255, 255, 255, 0.08)",
        primaryText: "#F8FAFC",
        secondaryText: "#94A3B8",
        accent: "#60A5FA",
        glow: "rgba(96, 165, 250, 0.4)",
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "Inter", "sans-serif"],
        mono: ["var(--font-geist-mono)", "Fira Code", "monospace"],
      },
      boxShadow: {
        hudGlow: "0 0 20px rgba(96, 165, 250, 0.15)",
        btnGlow: "0 0 15px rgba(96, 165, 250, 0.3)",
      },
      animation: {
        "scanline": "scanline 8s linear infinite",
        "pulse-slow": "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "fade-in-up": "fadeInUp 1s cubic-bezier(0.22, 1, 0.36, 1) forwards",
      },
      keyframes: {
        scanline: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" },
        },
        fadeInUp: {
          "0%": { opacity: "0", transform: "translateY(20px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
    },
  },
  plugins: [],
};

export default config;
