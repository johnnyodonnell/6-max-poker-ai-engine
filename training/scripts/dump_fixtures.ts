// Dump parity fixtures from the authoritative TS engine (src/engine/*.ts) for
// the Rust port in training/poker_core:
//
//   training/fixtures/rules_traces.json   full random matches, step by step:
//       every startHand (with the deck it shuffled), every action (with the
//       legal-action set it was chosen from), every advanceStreet, the state
//       after each step, and each hand's results + action history.
//       Every other decision also carries the encoder's output (sparse
//       [index, value] pairs), the action space, and raise-to amounts for a
//       few sizes, for encoder / action-space parity.
//   training/fixtures/eval_hands.json     random 7-card hands with their
//       evaluate7 score arrays (for ordering parity).
//
//   npx tsx training/scripts/dump_fixtures.ts [matches=300] [evalHands=20000]
//
// Then training/sync_to_box.sh and, on the box, cargo test -p poker_core.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { createDeck, shuffle } from "../../src/engine/cards.ts";
import { evaluate7 } from "../../src/engine/evaluator.ts";
import {
  advanceStreet,
  applyAction,
  createTable,
  legalActions,
  startHand,
} from "../../src/engine/game.ts";
import type { EngineState } from "../../src/engine/game.ts";
import { bestMove } from "../../src/engine/random.ts";
import { raiseToFor } from "../../src/engine/actions.ts";
import { encode } from "../../src/engine/encode.ts";

const here = dirname(fileURLToPath(import.meta.url));
const outDir = resolve(here, "../fixtures");

function summary(s: EngineState) {
  return {
    phase: s.phase,
    street: s.street,
    toAct: s.toAct,
    pot: s.pot,
    currentBet: s.currentBet,
    minRaise: s.minRaise,
    button: s.button,
    sbSeat: s.sbSeat,
    bbSeat: s.bbSeat,
    community: s.community.map((c) => c.id),
    // [stack, bet, commit, status, hasActed, actedAt, holeCards]
    players: s.players.map((p) => [
      p.stack,
      p.bet,
      p.commit,
      p.status,
      p.hasActed,
      p.actedAt,
      p.holeCards.map((c) => c.id),
    ]),
  };
}

// The full shuffled deck startHand dealt from: it pops hole cards off the end
// (two per seat, seat order), so put them back in reverse.
function fullDeck(s: EngineState): string[] {
  const dealt: string[] = [];
  for (const p of s.players) for (const c of p.holeCards) dealt.push(c.id);
  return [...s.deck.map((c) => c.id), ...dealt.reverse()];
}

function results(s: EngineState) {
  return {
    pots: s.results!.pots.map((p) => ({ amount: p.amount, winners: p.winners })),
    stacks: s.players.map((p) => p.stack),
    history: s.actions,
  };
}

let decision = 0;

// Encoder + action-space snapshot for the player to act (every other decision).
function encoding(s: EngineState) {
  if (decision++ % 2 !== 0) return undefined;
  const e = encode(s)!;
  const enc: [number, number][] = [];
  e.x.forEach((v, i) => {
    if (v !== 0) enc.push([i, v]);
  });
  const sp = e.space;
  const sizes: [number, number][] = [];
  if (sp.mask[2]) {
    for (const t of [-0.25, 0, 0.37, 0.81, 1.25]) {
      const sv = sp.sLo + t * (sp.sHi - sp.sLo);
      sizes.push([sv, raiseToFor(sp, sv)]);
    }
  }
  return {
    enc,
    space: {
      mask: sp.mask,
      potAfterCall: sp.potAfterCall,
      minRaiseTo: sp.minRaiseTo,
      maxRaiseTo: sp.maxRaiseTo,
      sLo: sp.sLo,
      sHi: sp.sHi,
    },
    sizes,
  };
}

function recordMatch() {
  const events: unknown[] = [];
  let s = startHand(createTable());
  events.push({ kind: "start", deck: fullDeck(s), after: summary(s) });
  for (let guard = 0; ; guard++) {
    if (guard > 200000) throw new Error("runaway match");
    if (s.phase === "match-over") break;
    if (s.phase === "hand-over") {
      s = startHand(s);
      events.push({ kind: "start", deck: s.phase === "match-over" ? null : fullDeck(s), after: summary(s) });
      continue;
    }
    if (s.phase === "street-complete") {
      s = advanceStreet(s);
      events.push({ kind: "advance", after: summary(s) });
    } else {
      const legal = legalActions(s)!;
      const enc = encoding(s);
      const action = bestMove(s);
      s = applyAction(s, action);
      events.push({ kind: "act", action, legal, ...(enc ? { enc } : {}), after: summary(s) });
    }
    if (s.phase === "hand-over" || s.phase === "match-over") {
      events.push({ kind: "end", results: results(s) });
    }
  }
  return { events };
}

const nMatches = Number(process.argv[2] ?? 300);
const nEval = Number(process.argv[3] ?? 20000);

mkdirSync(outDir, { recursive: true });

const matches = [];
for (let i = 0; i < nMatches; i++) matches.push(recordMatch());
writeFileSync(resolve(outDir, "rules_traces.json"), JSON.stringify({ matches }));

const hands = [];
for (let i = 0; i < nEval; i++) {
  const cards = shuffle(createDeck()).slice(0, 7);
  hands.push({ cards: cards.map((c) => c.id), score: evaluate7(cards) });
}
writeFileSync(resolve(outDir, "eval_hands.json"), JSON.stringify({ hands }));

const nEvents = matches.reduce((n, m) => n + m.events.length, 0);
console.log(`wrote ${nMatches} matches (${nEvents} events), ${nEval} eval hands to ${outDir}`);
