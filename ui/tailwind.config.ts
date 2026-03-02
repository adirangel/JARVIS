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
        "mc-bg": "#0d1117",
        "mc-bg-secondary": "#161b22",
        "mc-bg-tertiary": "#21262d",
        "mc-border": "#30363d",
        "mc-text": "#c9d1d9",
        "mc-text-secondary": "#8b949e",
        "mc-accent": "#58a6ff",
        "mc-accent-green": "#3fb950",
      },
    },
  },
  plugins: [],
};
export default config;
