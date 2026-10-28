import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  build: {
    // Recharts and its d3 helpers are ~560 kB minified (~160 kB gzipped) in one vendor chunk.
    chunkSizeWarningLimit: 700,
    rollupOptions: {
      output: {
        // React and the charting stack change rarely; separate chunks keep the app chunk small and cacheable.
        manualChunks(id) {
          if (!id.includes("node_modules")) return undefined;
          return /node_modules[\\/](react|react-dom|scheduler)[\\/]/.test(id) ? "react" : "vendor";
        },
      },
    },
  },
  server: {
    // The backend's CORS allow-list contains http://localhost:5173 and http://127.0.0.1:5173.
    // strictPort fails loudly instead of silently moving to a port the backend would reject.
    port: 5173,
    strictPort: true,
  },
  test: {
    environment: "jsdom",
    setupFiles: ["./src/test/setup.js"],
    css: false,
  },
});
