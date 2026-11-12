/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: { colors: { ink: "#09111f", surface: "#101b2d", cyan: "#4de1d2", lavender: "#a99bff", coral: "#ff9478" }, fontFamily: { sans: ["Inter", "ui-sans-serif", "system-ui", "sans-serif"], display: ["Plus Jakarta Sans", "Inter", "sans-serif"] } } },
  plugins: [],
};
