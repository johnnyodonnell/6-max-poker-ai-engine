// The v1 AI engine: pick a uniform-random legal action.
//
// Contract — the same shape as chess-ai-engine and fox-lite-ai-engine's
// random.js: bestMove(state) -> an action { type, amount? } for state.toAct.
// Caller must only invoke when it's a bot's turn (phase 'betting').
//
// "Random" here means: choose uniformly among the legal action *types*, and for
// a bet/raise choose a uniform-random legal raise-to amount. It's a placeholder
// — a search- or network-based engine can replace this file without the UI
// noticing, since the action contract is all the caller depends on.

import { legalActions } from "./game";
import type { EngineState, Action } from "./game";

export function bestMove(state: EngineState): Action {
  const legal = legalActions(state);
  if (!legal) return { type: "check" };

  const type = legal.actions[Math.floor(Math.random() * legal.actions.length)];

  if (type === "bet" || type === "raise") {
    const { minRaiseTo, maxRaiseTo } = legal;
    const to =
      minRaiseTo + Math.floor(Math.random() * (maxRaiseTo - minRaiseTo + 1));
    return { type, amount: to };
  }

  return { type };
}
