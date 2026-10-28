import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// While developing (npm run dev, on http://localhost:5173), API calls are
// forwarded to the Python server (python server.py, on port 8501).
// "npm run build" puts the finished app in dist/, which server.py serves itself.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: { "/api": "http://localhost:8501" },
  },
});
