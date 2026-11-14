// The neural AI engine — an async drop-in for random.ts.
//
//   await bestMove(state)  // -> an Action for state.toAct (one forward pass, no search)
//   preload()              // optional: warm the ONNX session before the first move
//
// Encode the acting bot's view, run the net, and *sample* from the policy:
// an action type from the masked softmax, and for a raise a size from the
// truncated normal over log pot-fraction. Sampling (not argmax) keeps the
// mixed strategy the bot was trained to play — bluffs, varied sizes.

import { encode } from "./encode";
import { NUM_TYPES, RAISE, sampleTruncNormal, toEngineAction } from "./actions";
import type { Action, EngineState } from "./game";
import { evaluate, loadModel } from "./net";
import { bestMove as randomMove } from "./random";

export async function bestMove(state: EngineState): Promise<Action> {
  const e = encode(state);
  if (!e) return { type: "check" };
  const { x, space } = e;

  let net;
  try {
    net = await evaluate(x);
  } catch (err) {
    // Keep the game playable if the model can't load.
    console.error("PokerNet: inference failed, playing randomly", err);
    return randomMove(state);
  }

  // Masked softmax over the legal action types.
  const legal = [...Array(NUM_TYPES).keys()].filter((t) => space.mask[t] > 0);
  const mx = Math.max(...legal.map((t) => net.typeLogits[t]));
  const w = legal.map((t) => Math.exp(net.typeLogits[t] - mx));
  const total = w.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  let type = legal[legal.length - 1];
  for (let k = 0; k < legal.length; k++) {
    r -= w[k];
    if (r < 0) {
      type = legal[k];
      break;
    }
  }

  let s = space.sLo;
  if (type === RAISE && space.sHi > space.sLo) {
    s = sampleTruncNormal(net.mu, net.sigma, space.sLo, space.sHi);
  }
  return toEngineAction(space, type, s);
}

export function preload(): Promise<unknown> {
  return loadModel();
}
