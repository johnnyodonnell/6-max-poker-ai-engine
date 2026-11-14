// The neural engine's action space, shared (by parity tests) with the Rust
// trainer (training/poker_core/src/actions.rs).
//
// The policy picks one of four action types — fold, check/call, raise,
// all-in — and, for a raise, a continuous size:
//
//   s = ln((raiseTo - currentBet) / potAfterCall)
//
// the raise increment as a fraction of the pot after calling, on a log scale
// ("half pot" is ln 0.5 in every spot). The legal raises (min-raise up to one
// chip short of all-in) form the interval [sLo, sHi]; the network outputs a
// normal (mu, sigma) over s that is truncated to that interval. All-in is its
// own action type.

import { legalActions } from "./game";
import type { Action, EngineState, LegalActions } from "./game";

export const FOLD = 0;
export const CALL = 1; // check or call
export const RAISE = 2; // bet or raise, sized by s
export const ALLIN = 3;
export const NUM_TYPES = 4;

export interface ActionSpace {
  legal: LegalActions;
  mask: [number, number, number, number]; // 1 = allowed, per action type
  currentBet: number;
  potAfterCall: number; // gathered pot + street bets + the mover's call
  minRaiseTo: number; // smallest legal (non-all-in) raise-to
  maxRaiseTo: number; // largest legal non-all-in raise-to (all-in minus one)
  sLo: number; // s of minRaiseTo (0 if raising isn't allowed)
  sHi: number; // s of maxRaiseTo
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
  const s = (to: number): number => Math.log((to - state.currentBet) / potAfterCall);
  return {
    legal,
    mask: [legal.actions.includes("fold") ? 1 : 0, 1, sized ? 1 : 0, canRaise ? 1 : 0],
    currentBet: state.currentBet,
    potAfterCall,
    minRaiseTo: sized ? minTo : 0,
    maxRaiseTo: sized ? maxTo : 0,
    sLo: sized ? s(minTo) : 0,
    sHi: sized ? s(maxTo) : 0,
  };
}

// The raise-to amount for size s (rounded to a chip, clamped to the legal band).
export function raiseToFor(space: ActionSpace, s: number): number {
  const to = Math.round(space.currentBet + Math.exp(s) * space.potAfterCall);
  return Math.min(Math.max(to, space.minRaiseTo), space.maxRaiseTo);
}

// The engine action for an action type (+ size for RAISE).
export function toEngineAction(space: ActionSpace, type: number, s = 0): Action {
  switch (type) {
    case FOLD:
      return { type: "fold" };
    case CALL:
      return { type: space.legal.canCheck ? "check" : "call" };
    case RAISE:
      return {
        type: space.currentBet === 0 ? "bet" : "raise",
        amount: raiseToFor(space, s),
      };
    default:
      return { type: "allin" };
  }
}

// --- truncated-normal sampling ----------------------------------------------

// Complementary error function (Numerical Recipes erfcc; |rel err| < 1.2e-7).
function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 1 / (1 + 0.5 * z);
  const r =
    t *
    Math.exp(
      -z * z -
        1.26551223 +
        t *
          (1.00002368 +
            t *
              (0.37409196 +
                t *
                  (0.09678418 +
                    t *
                      (-0.18628806 +
                        t *
                          (0.27886807 +
                            t *
                              (-1.13520398 +
                                t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))),
    );
  return x >= 0 ? r : 2 - r;
}

// Standard normal CDF.
function ndtr(x: number): number {
  return 0.5 * erfc(-x / Math.SQRT2);
}

// Inverse standard normal CDF (Acklam's rational approximation).
function ndtri(p: number): number {
  const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
  const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
  const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
  const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
  const lo = 0.02425;
  if (p < lo) {
    const q = Math.sqrt(-2 * Math.log(p));
    return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
      ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
  }
  if (p > 1 - lo) return -ndtri(1 - p);
  const q = p - 0.5;
  const r = q * q;
  return ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) /
    (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

// Draw from Normal(mu, sigma) truncated to [lo, hi] by inverse CDF. Works in
// the lower half (flipping when the interval lies above mu) so tail
// probabilities keep their precision.
export function sampleTruncNormal(
  mu: number,
  sigma: number,
  lo: number,
  hi: number,
  rand: () => number = Math.random,
): number {
  if (!(hi > lo)) return lo;
  if ((lo - mu) / sigma > 0) return -sampleTruncNormal(-mu, sigma, -hi, -lo, rand);
  const pa = ndtr((lo - mu) / sigma);
  const pb = ndtr((hi - mu) / sigma);
  if (!(pb > pa)) return hi; // all mass far below the interval: nearest edge
  const u = pa + rand() * (pb - pa);
  const x = mu + sigma * ndtri(Math.min(Math.max(u, 1e-300), 1 - 1e-16));
  return Math.min(Math.max(x, lo), hi);
}
