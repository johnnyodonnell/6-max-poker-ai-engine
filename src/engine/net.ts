// onnxruntime-web session for the poker policy net. Lazily loads
// /models/current.onnx on first use and caches the session; tries WebGPU,
// falls back to WASM (same pattern as fox-lite / chess-ai-engine).
//
//   await loadModel(source?)      // optional pre-warm (URL or model bytes)
//   await evaluate(inputF32)      // -> logits over the NUM_ACTIONS discrete actions

import * as ort from "onnxruntime-web";

import { INPUT_SIZE } from "./encode";

// vite.config.ts serves onnxruntime-web's wasm/mjs assets under /onnx-wasm/.
ort.env.wasm.wasmPaths = "/onnx-wasm/";

const DEFAULT_MODEL_URL = "/models/current.onnx";

let sessionPromise: Promise<ort.InferenceSession> | null = null;

export function loadModel(
  source: string | Uint8Array = DEFAULT_MODEL_URL,
): Promise<ort.InferenceSession> {
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    const providers: string[] = [];
    if (typeof navigator !== "undefined" && "gpu" in navigator) providers.push("webgpu");
    providers.push("wasm");
    const create = (eps: string[]) =>
      typeof source === "string"
        ? ort.InferenceSession.create(source, { executionProviders: eps })
        : ort.InferenceSession.create(source, { executionProviders: eps });
    try {
      return await create(providers);
    } catch (err) {
      console.warn("PokerNet: WebGPU failed, falling back to WASM", err);
      return await create(["wasm"]);
    }
  })();
  sessionPromise.catch(() => {
    sessionPromise = null; // allow a retry after a failed load
  });
  return sessionPromise;
}

// Policy logits, one per discrete action (see actions.ts), unmasked.
export async function evaluate(x: Float32Array): Promise<Float32Array> {
  const session = await loadModel();
  const out = await session.run({ input: new ort.Tensor("float32", x, [1, INPUT_SIZE]) });
  return out.logits.data as Float32Array;
}
