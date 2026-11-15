// The neural AI engine — an async drop-in for random.ts.
//
//   await bestMove(state)  // -> an Action for state.toAct (one forward pass, no search)
//   preload()              // optional: warm the ONNX session before the first move
//
// Encode the acting bot's view, run the net, and *sample* a discrete action
// (fold, check/call, a bet size, all-in) from the masked softmax. Sampling
// (not argmax) keeps the mixed strategy the bot was trained to play.

import { encode } from "./encode";
import { NUM_ACTIONS, toEngineAction } from "./actions";
import type { Action, EngineState } from "./game";
import { evaluate, loadModel } from "./net";
import { bestMove as randomMove } from "./random";

export async function bestMove(state: EngineState): Promise<Action> {
  const e = encode(state);
  if (!e) return { type: "check" };
  const { x, space } = e;

  let logits;
  try {
    logits = await evaluate(x);
  } catch (err) {
    // Keep the game playable if the model can't load.
    console.error("PokerNet: inference failed, playing randomly", err);
    return randomMove(state);
  }

  // Masked softmax over the legal actions.
  const legal = [...Array(NUM_ACTIONS).keys()].filter((a) => space.amask[a] > 0);
  const mx = Math.max(...legal.map((a) => logits[a]));
  const w = legal.map((a) => Math.exp(logits[a] - mx));
  const total = w.reduce((s, v) => s + v, 0);
  let r = Math.random() * total;
  let choice = legal[legal.length - 1];
  for (let k = 0; k < legal.length; k++) {
    r -= w[k];
    if (r < 0) {
      choice = legal[k];
      break;
    }
  }
  return toEngineAction(space, choice);
}

export function preload(): Promise<unknown> {
  return loadModel();
}
