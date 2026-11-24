// The bots' engine: a precomputed CFR strategy served from the box (cfr/,
// `sixmax serve`). Each decision is one POST /api/act with the bot's view of
// the hand. There is deliberately no fallback: if the engine can't be
// reached, bestMove rejects and the table shows the error. (The random bot,
// engine/random.ts, stays in the code base but is not served.)
//
//   await bestMove(state)       // -> an Action for state.toAct
//   onDecision(listener)        // every decision, for the debug view
//   logHand(state)              // send a finished hand to the box's hand log
//   newMatch()                  // New Game: later hands get new ids

import type { Action, EngineState } from "./game";
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
  // The real actions the strategy chose between, with the chance of each.
  options: DecisionOption[];
  chosen: string;
  // How the engine got there: the strategy it used, the bot's hand group,
  // and why it fell back to its equity rule if it did.
  details: string[];
}

// What the engine sends back with each move (cfr/sixmax_cfr/src/engine.rs).
interface EngineDebug {
  table: string;
  effective_bb: number;
  strategy_bb: number;
  street: string;
  bucket_label: string | null;
  path: string[];
  options: { label: string; action: string; prob: number }[];
  chosen: string;
  fallback: string | null;
}

export class EngineError extends Error {}

// Hand ids: one random id per page load, a game count (New Game starts the
// hand count over) and the hand number, so ids are unique across matches.
// The engine seeds a hand's bet-size translation from its id.
const matchId = Math.random().toString(36).slice(2, 10);
let game = 1;

function handId(state: EngineState): string {
  return `${matchId}-${game}-${state.handNumber}`;
}

export function newMatch(): void {
  game++;
}

type Listener = (d: BotDecision) => void;
let listeners: Listener[] = [];

export function onDecision(listener: Listener): () => void {
  listeners.push(listener);
  return () => {
    listeners = listeners.filter((l) => l !== listener);
  };
}

// The engine's action labels ("raise to 17", "all-in", ...) as the table says them.
function realLabel(a: string): string {
  const m = /^(bet|raise to) (\d+)$/.exec(a);
  if (m) return `${m[1] === "bet" ? "Bet" : "Raise to"} ${dollars(Number(m[2]))}`;
  if (a === "all-in") return "All-in";
  return a.charAt(0).toUpperCase() + a.slice(1);
}

function decision(state: EngineState, d: EngineDebug): BotDecision {
  const p = state.players[state.toAct];
  // Abstract actions that become the same real action are shown together.
  const options: DecisionOption[] = [];
  for (const o of d.options) {
    const label = realLabel(o.action);
    const same = options.find((x) => x.label === label);
    if (same) same.prob += o.prob;
    else options.push({ label, prob: o.prob });
  }
  const details = [
    `${d.table}: ${d.strategy_bb}bb strategy (effective stack ${d.effective_bb}bb)`,
  ];
  if (d.bucket_label) details.push(`Hand: ${d.bucket_label}`);
  if (d.fallback) details.push(`Equity rule: ${d.fallback}`);
  const chosen = d.fallback ? d.chosen : realLabel(d.chosen);
  return { hand: state.handNumber, seat: p.id, name: p.name, street: state.street, options, chosen, details };
}

export async function bestMove(state: EngineState): Promise<Action> {
  const bot = state.toAct;
  const body = {
    handId: handId(state),
    bot,
    button: state.button,
    // Stacks at the start of the hand, before the blinds (0 = sitting out).
    stacks: state.players.map((p) => p.stack + p.commit),
    hole: state.players[bot].holeCards.map((c) => c.id),
    board: state.community.map((c) => c.id),
    actions: state.actions,
  };
  let res: Response;
  try {
    res = await fetch("/api/act", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (err) {
    throw new EngineError(`Can't reach the engine (${String(err)}).`);
  }
  if (!res.ok) {
    let detail = await res.text();
    try {
      detail = JSON.parse(detail).error ?? detail;
    } catch {
      // not JSON: keep the text
    }
    throw new EngineError(`The engine returned ${res.status}: ${detail}`);
  }
  const data = (await res.json()) as {
    action: { type: Action["type"]; amount?: number };
    debug: EngineDebug;
  };
  const action: Action =
    data.action.amount === undefined
      ? { type: data.action.type }
      : { type: data.action.type, amount: data.action.amount };
  const d = decision(state, data.debug);
  for (const l of listeners) l(d);
  return action;
}

const logged = new Set<string>();

// Send a finished hand (phase hand-over or match-over) to the box's hand log
// ($SIXMAX_HOME/logs/hands.jsonl): every player's cards, the board, every
// action at its real size, and the result. Once per hand; failures are
// ignored (the log is for analysis, never in the way of play).
export function logHand(state: EngineState): void {
  if (state.phase !== "hand-over" && state.phase !== "match-over") return;
  const id = handId(state);
  if (logged.has(id)) return;
  logged.add(id);
  const won = new Map<number, number>();
  for (const w of state.results?.winners ?? []) {
    won.set(w.seat, (won.get(w.seat) ?? 0) + w.amountWon);
  }
  const record = {
    handId: id,
    handNumber: state.handNumber,
    time: new Date().toISOString(),
    button: state.button,
    // By seat. A player's commit stays on them until the next deal, so the
    // stack the hand started with is what they end with, less what they
    // won, plus what they put in.
    players: state.players.map((p) => ({
      name: p.name,
      human: p.isHuman,
      startStack: p.stack - (won.get(p.id) ?? 0) + p.commit,
      endStack: p.stack,
      hole: p.holeCards.map((c) => c.id),
      status: p.status,
    })),
    board: state.community.map((c) => c.id),
    actions: state.actions,
    results: state.results && {
      reveal: state.results.reveal,
      mucked: state.results.mucked,
      winners: state.results.winners,
      pots: state.results.pots.map((p) => ({ amount: p.amount, winners: p.winners })),
    },
    matchOver: state.phase === "match-over",
  };
  void fetch("/api/hand", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(record),
  }).catch(() => {});
}
