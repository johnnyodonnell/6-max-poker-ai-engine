// The match carried across page refreshes: each player's stack, the button and
// the hand count, saved in the browser between hands. A hand in progress isn't
// saved, so a refresh mid-hand deals again from the stacks it started with.

import { createTable, SEATS, STARTING_STACK } from "./game";
import type { EngineState } from "./game";

const KEY = "6max-match";

interface SavedMatch {
  stacks: number[];
  button: number;
  handNumber: number;
}

function isValid(m: unknown): m is SavedMatch {
  if (typeof m !== "object" || m === null) return false;
  const { stacks, button, handNumber } = m as Partial<SavedMatch>;
  return (
    Array.isArray(stacks) &&
    stacks.length === SEATS &&
    stacks.every((s) => Number.isInteger(s) && s >= 0) &&
    // No chips made or lost.
    stacks.reduce((a, b) => a + b, 0) === SEATS * STARTING_STACK &&
    // The match is still on: you have chips, and so does someone else.
    stacks[0] > 0 &&
    stacks.filter((s) => s > 0).length >= 2 &&
    Number.isInteger(button) &&
    button! >= -1 &&
    button! < SEATS &&
    Number.isInteger(handNumber) &&
    handNumber! >= 0
  );
}

// A seated table at the saved stacks (the busted sitting out), or a fresh one
// if nothing usable is saved.
export function loadTable(): EngineState {
  const table = createTable();
  let saved: unknown = null;
  try {
    saved = JSON.parse(localStorage.getItem(KEY) ?? "null");
  } catch {
    // storage unavailable or corrupt: start fresh
  }
  if (!isValid(saved)) return table;
  return {
    ...table,
    players: table.players.map((p, i) => ({
      ...p,
      stack: saved.stacks[i],
      status: saved.stacks[i] > 0 ? "active" : "out",
    })),
    button: saved.button,
    handNumber: saved.handNumber,
  };
}

// Save the match between hands; forget it once the match is over.
export function saveTable(state: EngineState): void {
  try {
    if (state.phase === "match-over") {
      localStorage.removeItem(KEY);
    } else if (state.phase === "waiting" || state.phase === "hand-over") {
      const saved: SavedMatch = {
        stacks: state.players.map((p) => p.stack),
        button: state.button,
        handNumber: state.handNumber,
      };
      localStorage.setItem(KEY, JSON.stringify(saved));
    }
  } catch {
    // storage unavailable: the match just won't survive a refresh
  }
}
