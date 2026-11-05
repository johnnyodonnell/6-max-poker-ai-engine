import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Pure client-side app with a local random-move engine — no backend and no
// model assets to serve.
export default defineConfig({
  plugins: [react()],
});
