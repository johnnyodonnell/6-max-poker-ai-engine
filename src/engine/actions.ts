// The neural engine's action space, shared (by parity tests) with the Rust
// trainer (training/poker_core/src/actions.rs).
//
// The policy picks one of NUM_ACTIONS discrete actions:
//
//   0 fold   1 check/call   2 min-raise   3..8 bet/raise to a pot fraction
//   (POT_FRACS of the pot after calling: raiseTo = currentBet + f * potAfterCall,
//   rounded to a chip)   9 all-in
//
// A pot-fraction size is offered only when it lands strictly between the
// previous offered size and all-in, so every legal action is a distinct
// amount. The coarse action types (fold / call / raise / all-in) and the legal
// raise band [sLo, sHi] (s = ln((raiseTo - currentBet) / potAfterCall)) are
// still part of the encoder's input.

import { legalActions } from "./game";
import type { Action, EngineState, LegalActions } from "./game";

export const FOLD = 0;
export const CALL = 1; // check or call
export const RAISE = 2; // a sized (non-all-in) bet or raise
export const ALLIN = 3;
export const NUM_TYPES = 4;

export const A_FOLD = 0;
export const A_CALL = 1;
export const A_MINRAISE = 2;
export const A_ALLIN = 9;
export const NUM_ACTIONS = 10;
export const POT_FRACS = [0.33, 0.5, 0.75, 1.0, 1.5, 2.5];

export interface ActionSpace {
  legal: LegalActions;
  mask: [number, number, number, number]; // coarse type legality (encoder input)
  currentBet: number;
  potAfterCall: number; // gathered pot + street bets + the mover's call
  minRaiseTo: number; // smallest legal (non-all-in) raise-to
  maxRaiseTo: number; // largest legal non-all-in raise-to (all-in minus one)
  sLo: number; // s of minRaiseTo (0 if raising isn't allowed)
  sHi: number; // s of maxRaiseTo
  amask: number[]; // legality of each discrete action (NUM_ACTIONS)
  amounts: number[]; // raise-to of each legal sized action, else 0
}

export function actionSpace(state: EngineState): ActionSpace | null {
  const legal = legalActions(state);
  if (!legal) return null;
  const p = state.players[state.toAct];
  const potTotal = state.pot + state.players.reduce((n, q) => n + q.bet, 0);
  const potAfterCall = potTotal + (state.currentBet - p.bet);
  const canRaise = legal.actions.includes("bet") || legal.actions.includes("raise");
  // A non-all-in raise needs room between the min-raise and the shove.
  const sized = canRaise && legal.minRaiseTo < legal.maxRaiseTo;
  const minTo = legal.minRaiseTo;
  const maxTo = legal.maxRaiseTo - 1;
  const cb = state.currentBet;
  const s = (to: number): number => Math.log((to - cb) / potAfterCall);

  const canFold = legal.actions.includes("fold");
  const amask = new Array(NUM_ACTIONS).fill(0);
  const amounts = new Array(NUM_ACTIONS).fill(0);
  amask[A_FOLD] = canFold ? 1 : 0;
  amask[A_CALL] = 1;
  amask[A_ALLIN] = canRaise ? 1 : 0;
  if (sized) {
    amask[A_MINRAISE] = 1;
    amounts[A_MINRAISE] = minTo;
    let last = minTo;
    POT_FRACS.forEach((f, k) => {
      const to = Math.floor(cb + f * potAfterCall + 0.5);
      if (to > last && to <= maxTo) {
        amask[3 + k] = 1;
        amounts[3 + k] = to;
        last = to;
      }
    });
  }
  return {
    legal,
    mask: [canFold ? 1 : 0, 1, sized ? 1 : 0, canRaise ? 1 : 0],
    currentBet: cb,
    potAfterCall,
    minRaiseTo: sized ? minTo : 0,
    maxRaiseTo: sized ? maxTo : 0,
    sLo: sized ? s(minTo) : 0,
    sHi: sized ? s(maxTo) : 0,
    amask,
    amounts,
  };
}

// The engine action for a discrete action.
export function toEngineAction(space: ActionSpace, a: number): Action {
  switch (a) {
    case A_FOLD:
      return { type: "fold" };
    case A_CALL:
      return { type: space.legal.canCheck ? "check" : "call" };
    case A_ALLIN:
      return { type: "allin" };
    default:
      return { type: space.currentBet === 0 ? "bet" : "raise", amount: space.amounts[a] };
  }
}
