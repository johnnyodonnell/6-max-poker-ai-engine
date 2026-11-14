// onnxruntime-web session for the poker policy/value net. Lazily loads
// /models/current.onnx on first use and caches the session; tries WebGPU,
// falls back to WASM (same pattern as fox-lite / chess-ai-engine).
//
//   await loadModel(source?)      // optional pre-warm (URL or model bytes)
//   await evaluate(inputF32)      // -> { typeLogits, size: [mu, sigma], value }

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

export interface NetOutput {
  typeLogits: Float32Array; // fold, check/call, raise, all-in
  mu: number; // raise size center (log pot-fraction)
  sigma: number;
  value: number;
}

export async function evaluate(x: Float32Array): Promise<NetOutput> {
  const session = await loadModel();
  const out = await session.run({ input: new ort.Tensor("float32", x, [1, INPUT_SIZE]) });
  const size = out.size_params.data as Float32Array;
  return {
    typeLogits: out.type_logits.data as Float32Array,
    mu: size[0],
    sigma: size[1],
    value: (out.value.data as Float32Array)[0],
  };
}
