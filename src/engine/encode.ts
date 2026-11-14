// Canonical NN input encoding for the 6-max poker engine. This is the source
// of truth; the Rust trainer (training/poker_core/src/encode.rs) is
// parity-tested against it.
//
// Everything is from the point of view of the player to act ("the mover"):
// seats are numbered clockwise from the mover (relative seat 0 = the mover),
// and suits are relabeled by order of first appearance (hole cards, then flop,
// turn, river; ties by rank, then real suit) so suit-isomorphic spots encode
// identically.
//
// One flat Float32Array, sliced inside the network:
//   [ cards (7) | history (HIST_TOKENS x TOKEN_FEATS) | static (STATIC_SIZE) ]
//
// cards:   canonical card indices (canonSuit * 13 + rank - 2) for hole 1-2,
//          flop 1-3, turn, river; NO_CARD for cards not dealt yet.
// history: one token per betting action this hand, most recent first; slots
//          past the end are all-zero (valid = 0). Older actions beyond
//          HIST_TOKENS are dropped.
// static:  table state (see `encodeStatic`).

import { BIG_BLIND, SEATS } from "./game";
import type { EngineState, HandAction, Street } from "./game";
import type { Card } from "./cards";
import { SUITS } from "./cards";
import { actionSpace } from "./actions";
import type { ActionSpace } from "./actions";

export const CARD_SLOTS = 7;
export const NO_CARD = 52;
export const HIST_TOKENS = 48;
export const TOKEN_FEATS = 22;
export const SEAT_FEATS = 12;
export const STATIC_SIZE = 4 + 5 + 2 + SEATS * SEAT_FEATS + 4 + 4 + 5; // 96
export const HIST_OFFSET = CARD_SLOTS;
export const STATIC_OFFSET = CARD_SLOTS + HIST_TOKENS * TOKEN_FEATS;
export const INPUT_SIZE = STATIC_OFFSET + STATIC_SIZE; // 1159

const STREETS: Street[] = ["preflop", "flop", "turn", "river"];
const TYPES = ["fold", "check", "call", "bet", "raise"] as const;

const lbb = (chips: number): number => Math.log1p(chips / BIG_BLIND);

// Cards in scan order: hole (rank desc, suit), flop (rank desc, suit), turn, river.
function orderedGroup(cards: Card[]): Card[] {
  return cards
    .slice()
    .sort((a, b) => b.rank - a.rank || SUITS.indexOf(a.suit) - SUITS.indexOf(b.suit));
}

// Canonical card indices for the mover's hole cards + the board.
export function canonicalCards(hole: Card[], community: Card[]): number[] {
  const groups = [
    orderedGroup(hole),
    orderedGroup(community.slice(0, 3)),
    community.slice(3, 4),
    community.slice(4, 5),
  ];
  const suitMap = new Map<string, number>();
  const out: number[] = [];
  for (const g of groups) {
    for (const c of g) {
      if (!suitMap.has(c.suit)) suitMap.set(c.suit, suitMap.size);
      out.push(suitMap.get(c.suit)! * 13 + (c.rank - 2));
    }
  }
  // hole(2) flop(3) turn(1) river(1), padded with NO_CARD
  const slots: number[] = [];
  let k = 0;
  for (const [g, n] of [[groups[0], 2], [groups[1], 3], [groups[2], 1], [groups[3], 1]] as const) {
    for (let j = 0; j < n; j++) slots.push(j < g.length ? out[k++] : NO_CARD);
  }
  return slots;
}

function encodeToken(x: Float32Array, off: number, a: HandAction, mover: number): void {
  x[off + STREETS.indexOf(a.street)] = 1;
  x[off + 4 + ((a.seat - mover + SEATS) % SEATS)] = 1;
  x[off + 10 + TYPES.indexOf(a.type)] = 1;
  x[off + 15] = a.allin ? 1 : 0;
  const betBefore = a.to - a.added;
  if (a.type === "bet" || a.type === "raise") {
    const potAfterCall = a.potBefore + (a.currentBetBefore - betBefore);
    x[off + 16] = Math.log((a.to - a.currentBetBefore) / potAfterCall);
  }
  if (a.type === "call") x[off + 17] = a.added / a.potBefore;
  x[off + 18] = lbb(a.added);
  x[off + 19] = lbb(a.potBefore);
  x[off + 20] = lbb(a.stackBefore);
  x[off + 21] = 1;
}

function encodeStatic(x: Float32Array, off: number, s: EngineState, space: ActionSpace): void {
  const mover = s.toAct;
  const me = s.players[mover];
  let o = off;
  x[o + STREETS.indexOf(s.street)] = 1;
  o += 4;
  const dealt = s.players.filter((p) => p.holeCards.length > 0).length; // 2..6
  x[o + Math.min(Math.max(dealt, 2), 6) - 2] = 1;
  o += 5;
  const inHand = s.players.filter((p) => p.status === "active" || p.status === "allin").length;
  const canAct = s.players.filter((p) => p.status === "active").length;
  x[o] = inHand / SEATS;
  x[o + 1] = canAct / SEATS;
  o += 2;
  for (let r = 0; r < SEATS; r++) {
    const i = (mover + r) % SEATS;
    const p = s.players[i];
    const st = ["out", "active", "allin", "folded"].indexOf(p.status);
    x[o + st] = 1;
    x[o + 4] = i === s.button ? 1 : 0;
    x[o + 5] = i === s.sbSeat ? 1 : 0;
    x[o + 6] = i === s.bbSeat ? 1 : 0;
    x[o + 7] = lbb(p.stack);
    x[o + 8] = lbb(p.bet);
    x[o + 9] = lbb(p.commit);
    x[o + 10] = p.stack / 200;
    x[o + 11] = p.hasActed ? 1 : 0;
    o += SEAT_FEATS;
  }
  const potTotal = s.pot + s.players.reduce((n, q) => n + q.bet, 0);
  const toCall = space.legal.toCall;
  x[o] = lbb(potTotal);
  x[o + 1] = lbb(toCall);
  x[o + 2] = toCall / (potTotal + toCall);
  x[o + 3] = Math.log1p(me.stack / potTotal);
  o += 4;
  x[o] = space.sLo;
  x[o + 1] = space.sHi;
  x[o + 2] = lbb(space.minRaiseTo);
  x[o + 3] = lbb(space.maxRaiseTo);
  o += 4;
  x[o] = space.mask[0];
  x[o + 1] = space.legal.canCheck ? 1 : 0;
  x[o + 2] = space.mask[2];
  x[o + 3] = space.mask[3];
  x[o + 4] = space.legal.canCheck ? 0 : 1;
}

// Encode the state for the player to act. Returns null if nobody is to act.
export function encode(s: EngineState): { x: Float32Array; space: ActionSpace } | null {
  const space = actionSpace(s);
  if (!space) return null;
  const mover = s.toAct;
  const x = new Float32Array(INPUT_SIZE);
  const cards = canonicalCards(s.players[mover].holeCards, s.community);
  for (let k = 0; k < CARD_SLOTS; k++) x[k] = cards[k];
  const acts = s.actions;
  for (let t = 0; t < Math.min(HIST_TOKENS, acts.length); t++) {
    encodeToken(x, HIST_OFFSET + t * TOKEN_FEATS, acts[acts.length - 1 - t], mover);
  }
  encodeStatic(x, STATIC_OFFSET, s, space);
  return { x, space };
}
