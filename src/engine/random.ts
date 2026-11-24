// The AI engine: pick a uniform-random legal action.
//
// Contract — the same shape as chess-ai-engine and fox-lite-ai-engine's
// random.js: bestMove(state) -> an action { type, amount? } for state.toAct.
// Caller must only invoke when it's a bot's turn (phase 'betting').
//
// "Random" here means: choose uniformly among the legal action *types*, and for
// a bet/raise choose a uniform-random legal raise-to amount. It's a placeholder
// — a search- or network-based engine can replace this file without the UI
// noticing, since the action contract is all the caller depends on.
//
//   bestMove(state)        // -> an Action for state.toAct
//   onDecision(listener)   // every decision's options and pick (the debug view)

import { legalActions } from "./game";
import type { EngineState, Action, ActionType, LegalActions } from "./game";
import { dollars } from "../money";

export interface DecisionOption {
  label: string;
  prob: number;
}

// One bot decision, for the debug view.
export interface BotDecision {
  hand: number;
  seat: number;
  name: string;
  street: string;
  // The legal actions with the chance of picking each.
  options: DecisionOption[];
  chosen: string;
}

type Listener = (d: BotDecision) => void;
let listeners: Listener[] = [];

export function onDecision(listener: Listener): () => void {
  listeners.push(listener);
  return () => {
    listeners = listeners.filter((l) => l !== listener);
  };
}

// An action type as the table would say it (a bet or raise as its range).
function optionLabel(type: ActionType, legal: LegalActions): string {
  const range = `${dollars(legal.minRaiseTo)}–${dollars(legal.maxRaiseTo)}`;
  switch (type) {
    case "fold":
      return "Fold";
    case "check":
      return "Check";
    case "call":
      return `Call ${dollars(legal.toCall)}`;
    case "bet":
      return legal.minRaiseTo < legal.maxRaiseTo ? `Bet ${range}` : `Bet ${dollars(legal.maxRaiseTo)}`;
    case "raise":
      return legal.minRaiseTo < legal.maxRaiseTo
        ? `Raise to ${range}`
        : `Raise to ${dollars(legal.maxRaiseTo)}`;
    default:
      return "All-in";
  }
}

function chosenLabel(action: Action, legal: LegalActions): string {
  const allIn = action.amount === legal.maxRaiseTo ? " (all-in)" : "";
  if (action.type === "bet") return `Bet ${dollars(action.amount!)}${allIn}`;
  if (action.type === "raise") return `Raise to ${dollars(action.amount!)}${allIn}`;
  return optionLabel(action.type, legal);
}

export function bestMove(state: EngineState): Action {
  const legal = legalActions(state);
  if (!legal) return { type: "check" };

  const type = legal.actions[Math.floor(Math.random() * legal.actions.length)];

  let action: Action = { type };
  if (type === "bet" || type === "raise") {
    const { minRaiseTo, maxRaiseTo } = legal;
    const to =
      minRaiseTo + Math.floor(Math.random() * (maxRaiseTo - minRaiseTo + 1));
    action = { type, amount: to };
  }

  if (listeners.length) {
    const p = state.players[state.toAct];
    const decision: BotDecision = {
      hand: state.handNumber,
      seat: p.id,
      name: p.name,
      street: state.street,
      options: legal.actions.map((t) => ({
        label: optionLabel(t, legal),
        prob: 1 / legal.actions.length,
      })),
      chosen: chosenLabel(action, legal),
    };
    for (const l of listeners) l(decision);
  }

  return action;
}
