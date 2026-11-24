// The neural AI engine — an async drop-in for random.ts.
//
//   await bestMove(state)  // -> an Action for state.toAct (one forward pass, no search)
//   preload()              // optional: warm the ONNX session before the first move
//   onDecision(listener)   // every decision's options and pick (the debug view)
//
// Encode the acting bot's view, run the net, and *sample* a discrete action
// (fold, check/call, a bet size, all-in) from the masked softmax. Sampling
// (not argmax) keeps the mixed strategy the bot was trained to play.

import { encode } from "./encode";
import { A_ALLIN, A_CALL, A_FOLD, A_MINRAISE, NUM_ACTIONS, POT_FRACS, toEngineAction } from "./actions";
import type { ActionSpace } from "./actions";
import type { Action, EngineState } from "./game";
import { evaluate, loadModel } from "./net";
import { bestMove as randomMove } from "./random";
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
  // The legal actions with the net's probability of each (empty on fallback).
  options: DecisionOption[];
  chosen: string;
  // Set when the net couldn't run and the move was random.
  fallback: string | null;
}

type Listener = (d: BotDecision) => void;
let listeners: Listener[] = [];

export function onDecision(listener: Listener): () => void {
  listeners.push(listener);
  return () => {
    listeners = listeners.filter((l) => l !== listener);
  };
}

function emit(state: EngineState, d: Pick<BotDecision, "options" | "chosen" | "fallback">): void {
  const p = state.players[state.toAct];
  const decision = { hand: state.handNumber, seat: p.id, name: p.name, street: state.street, ...d };
  for (const l of listeners) l(decision);
}

// A discrete action as the table would say it.
function optionLabel(state: EngineState, space: ActionSpace, a: number): string {
  const p = state.players[state.toAct];
  const verb = space.currentBet === 0 ? "Bet" : "Raise to";
  if (a === A_FOLD) return "Fold";
  if (a === A_CALL) {
    return space.legal.canCheck ? "Check" : `Call ${dollars(Math.min(space.legal.toCall, p.stack))}`;
  }
  if (a === A_ALLIN) return `All-in ${dollars(p.bet + p.stack)}`;
  const size = a === A_MINRAISE ? "min" : `${Math.round(POT_FRACS[a - 3] * 100)}% pot`;
  return `${verb} ${dollars(space.amounts[a])} (${size})`;
}

// A plain description of an engine action (for the random fallback).
function actionLabel(action: Action): string {
  switch (action.type) {
    case "bet":
      return `Bet ${dollars(action.amount ?? 0)}`;
    case "raise":
      return `Raise to ${dollars(action.amount ?? 0)}`;
    case "allin":
      return "All-in";
    default:
      return action.type[0].toUpperCase() + action.type.slice(1);
  }
}

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
    const action = randomMove(state);
    emit(state, {
      options: [],
      chosen: actionLabel(action),
      fallback: err instanceof Error ? err.message : String(err),
    });
    return action;
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
  if (listeners.length) {
    emit(state, {
      options: legal.map((a, k) => ({ label: optionLabel(state, space, a), prob: w[k] / total })),
      chosen: optionLabel(state, space, choice),
      fallback: null,
    });
  }
  return toEngineAction(space, choice);
}

export function preload(): Promise<unknown> {
  return loadModel();
}
