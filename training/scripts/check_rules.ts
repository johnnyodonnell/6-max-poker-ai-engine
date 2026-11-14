// Scenario checks for game.ts rule details the AI engine depends on:
//   1. a short all-in doesn't reopen the betting for players who already acted
//      (but several short all-ins that add up to a full raise do);
//   2. odd chips in a split pot go to the first winner left of the button;
//   3. the structured per-hand action history.
//
//   npx tsx training/scripts/check_rules.ts

import assert from "node:assert/strict";

import { cardId } from "../../src/engine/cards.ts";
import type { Card, Suit } from "../../src/engine/cards.ts";
import {
  advanceStreet,
  applyAction,
  createTable,
  legalActions,
  startHand,
} from "../../src/engine/game.ts";
import type { Action, EngineState } from "../../src/engine/game.ts";

function card(rank: number, suit: Suit): Card {
  return { rank, suit, id: cardId(rank, suit) };
}

// Deal a hand with the button on `button` and the given stacks.
function deal(button: number, stacks: number[]): EngineState {
  const t = createTable();
  t.button = (button + 5) % 6; // startHand moves the button one seat on
  t.players.forEach((p, i) => (p.stack = stacks[i]));
  return startHand(t);
}

function act(s: EngineState, seat: number, a: Action): EngineState {
  assert.equal(s.phase, "betting");
  assert.equal(s.toAct, seat, `expected seat ${seat} to act, got ${s.toAct}`);
  return applyAction(s, a);
}

const canRaise = (s: EngineState): boolean => {
  const l = legalActions(s)!;
  return l.actions.includes("raise") || l.actions.includes("bet");
};

// --- 1a. A single short all-in doesn't reopen the betting ------------------
{
  // Button 0, SB 1, BB 2; seat 3 opens.
  let s = deal(0, [200, 200, 200, 200, 14, 200]);
  s = act(s, 3, { type: "raise", amount: 10 }); // full raise (min raise now 8)
  s = act(s, 4, { type: "allin" }); // to 14: a raise of 4, short of 8
  assert.ok(canRaise(s), "seat 5 hasn't acted yet, so it may still raise");
  s = act(s, 5, { type: "fold" });
  s = act(s, 0, { type: "fold" });
  s = act(s, 1, { type: "fold" });
  s = act(s, 2, { type: "fold" });
  assert.equal(s.toAct, 3);
  assert.deepEqual(legalActions(s)!.actions, ["fold", "call"]);
  // A raise attempt is downgraded to a call.
  s = applyAction(s, { type: "raise", amount: 40 });
  assert.equal(s.players[3].bet, 14);
  assert.equal(s.phase, "street-complete");
  console.log("ok  short all-in does not reopen the betting");
}

// --- 1b. Short all-ins that add up to a full raise do reopen it -------------
{
  let s = deal(0, [200, 200, 200, 200, 14, 19]);
  s = act(s, 3, { type: "raise", amount: 10 }); // min raise now 8
  s = act(s, 4, { type: "allin" }); // to 14 (short)
  s = act(s, 5, { type: "allin" }); // to 19 (short), 9 above seat 3's 10
  s = act(s, 0, { type: "fold" });
  s = act(s, 1, { type: "fold" });
  s = act(s, 2, { type: "fold" });
  assert.equal(s.toAct, 3);
  assert.ok(canRaise(s), "19 - 10 = 9 >= 8: seat 3 faces a full raise");
  console.log("ok  cumulative short all-ins reopen the betting");
}

// --- 2. Odd chip goes to the first winner left of the button ----------------
{
  // Button 3, SB 4, BB 5. Seat 0 limps, SB folds, BB checks: pot 5, split
  // between seats 0 and 5. Left of the button the order is 4, 5, 0, ... so
  // seat 5 gets the odd chip.
  let s = deal(3, [200, 200, 200, 200, 200, 200]);
  s.players[0].holeCards = [card(2, "c"), card(3, "d")];
  s.players[5].holeCards = [card(2, "d"), card(3, "c")];
  // Board As Ks Qs Js 9h: both play A-K-Q-J-9 (deck pops from the end).
  s.deck = [card(9, "h"), card(11, "s"), card(12, "s"), card(13, "s"), card(14, "s")];
  s = act(s, 0, { type: "call" });
  s = act(s, 1, { type: "fold" });
  s = act(s, 2, { type: "fold" });
  s = act(s, 3, { type: "fold" });
  s = act(s, 4, { type: "fold" });
  s = act(s, 5, { type: "check" });
  while (s.phase !== "hand-over") {
    s = s.phase === "street-complete" ? advanceStreet(s) : applyAction(s, { type: "check" });
  }
  assert.equal(s.players[5].stack, 201, "BB (first left of the button) gets 3");
  assert.equal(s.players[0].stack, 200, "seat 0 gets 2");
  console.log("ok  odd chip goes left of the button");
}

// --- 3. Structured action history -------------------------------------------
{
  let s = deal(0, [200, 200, 200, 200, 14, 200]);
  s = act(s, 3, { type: "raise", amount: 10 });
  s = act(s, 4, { type: "allin" });
  s = act(s, 5, { type: "call" });
  assert.deepEqual(s.actions, [
    { seat: 3, street: "preflop", type: "raise", allin: false, to: 10, added: 10,
      potBefore: 3, currentBetBefore: 2, stackBefore: 200 },
    { seat: 4, street: "preflop", type: "raise", allin: true, to: 14, added: 14,
      potBefore: 13, currentBetBefore: 10, stackBefore: 14 },
    { seat: 5, street: "preflop", type: "call", allin: false, to: 14, added: 14,
      potBefore: 27, currentBetBefore: 14, stackBefore: 200 },
  ]);
  assert.deepEqual(startHand(s).actions, [], "history resets each hand");
  console.log("ok  action history");
}

console.log("all rule checks passed");
