// ONNX-vs-tch parity: run the Rust-exported ONNX through onnxruntime-web
// (wasm, in Node — the same runtime the browser uses) on the fixture's
// positions and compare against the outputs tch computed for them.
//
//   # on the box:
//   poker export --weights <w>.safetensors (or --init 7) --out model.onnx --fixture fx.json
//   # here (after copying both into training/fixtures/):
//   npx tsx training/scripts/test_onnx_parity.ts [model.onnx] [fx.json]

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import * as ort from "onnxruntime-web";

import { INPUT_SIZE } from "../../src/engine/encode.ts";

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, "../..");
ort.env.wasm.wasmPaths = resolve(root, "node_modules/onnxruntime-web/dist") + "/";
ort.env.wasm.numThreads = 1;

const modelPath = process.argv[2] ?? resolve(here, "../fixtures/parity_model.onnx");
const fxPath = process.argv[3] ?? resolve(here, "../fixtures/parity_fx.json");

const fx = JSON.parse(readFileSync(fxPath, "utf8"));
const n: number = fx.n;
const input = new Float32Array(n * INPUT_SIZE);
fx.inputs.forEach((row: number[], i: number) => input.set(row, i * INPUT_SIZE));

const session = await ort.InferenceSession.create(readFileSync(modelPath), {
  executionProviders: ["wasm"],
});
const out = await session.run({ input: new ort.Tensor("float32", input, [n, INPUT_SIZE]) });

let worst = 0;
for (const name of ["type_logits", "size_params", "value"]) {
  const got = out[name].data as Float32Array;
  const want: number[] = fx[name];
  if (got.length !== want.length) throw new Error(`${name}: length ${got.length} vs ${want.length}`);
  let maxd = 0;
  for (let i = 0; i < want.length; i++) maxd = Math.max(maxd, Math.abs(got[i] - want[i]));
  console.log(`${name.padEnd(12)} max|Δ| = ${maxd.toExponential(2)}`);
  worst = Math.max(worst, maxd);
}
const ok = worst < 1e-4;
console.log(ok ? "ONNX PARITY OK" : "ONNX PARITY FAILED");
process.exit(ok ? 0 : 1);
