import { createReadStream, existsSync, readdirSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

import { defineConfig } from "vite";
import type { Plugin } from "vite";
import react from "@vitejs/plugin-react";

// onnxruntime-web dynamically imports its wasm glue (.mjs + .wasm) at runtime.
// Vite's dev server refuses module imports of files under /public/, so we
// serve them straight from node_modules under a virtual /onnx-wasm/ prefix —
// for dev (middleware) and production (emitted assets). Same approach as
// fox-lite-ai-engine and chess-ai-engine.
const ORT_DIR = resolve("node_modules/onnxruntime-web/dist");
const ORT_FILES = existsSync(ORT_DIR)
  ? readdirSync(ORT_DIR).filter((f) => /^ort-wasm.*\.(wasm|mjs)$/.test(f))
  : [];

const onnxWasmAssets: Plugin = {
  name: "serve-onnx-wasm",
  configureServer(server) {
    server.middlewares.use("/onnx-wasm", (req, res, next) => {
      const filename = (req.url ?? "").split("?")[0].replace(/^\//, "");
      if (!ORT_FILES.includes(filename)) return next();
      res.setHeader(
        "Content-Type",
        filename.endsWith(".wasm") ? "application/wasm" : "text/javascript",
      );
      createReadStream(resolve(ORT_DIR, filename)).pipe(res);
    });
  },
  generateBundle() {
    for (const file of ORT_FILES) {
      this.emitFile({
        type: "asset",
        fileName: `onnx-wasm/${file}`,
        source: readFileSync(resolve(ORT_DIR, file)),
      });
    }
  },
};

export default defineConfig({
  plugins: [react(), onnxWasmAssets],
});
