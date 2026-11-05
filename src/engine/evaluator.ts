// Seven-card hand evaluation for Texas Hold'em.
//
// evaluate7(cards) returns a "score" array whose lexicographic order matches
// poker strength: a bigger array (compared element-by-element) is the stronger
// hand. The first element is the hand category (see CATEGORY below); the rest
// are kicker ranks, high to low, enough to break every tie within a category.
//
// We just enumerate the 21 five-card subsets of the seven and keep the best.
// It's trivially correct and plenty fast for one table.

import type { Card } from "./cards";

export type Score = number[];

export const CATEGORY = {
  HIGH_CARD: 0,
  PAIR: 1,
  TWO_PAIR: 2,
  TRIPS: 3,
  STRAIGHT: 4,
  FLUSH: 5,
  FULL_HOUSE: 6,
  QUADS: 7,
  STRAIGHT_FLUSH: 8,
} as const;

export const CATEGORY_NAMES = [
  "High Card",
  "One Pair",
  "Two Pair",
  "Three of a Kind",
  "Straight",
  "Flush",
  "Full House",
  "Four of a Kind",
  "Straight Flush",
];

// Index combinations for "choose 5 of n", memoized per n. Hold'em only ever
// asks for 5, 6, or 7 (a hand can end before the board is complete, leaving a
// lone player with as few as 2+0 cards — handled by the length<=5 shortcut).
const combosCache = new Map<number, number[][]>();
function fiveOf(n: number): number[][] {
  const cached = combosCache.get(n);
  if (cached) return cached;
  const combos: number[][] = [];
  for (let a = 0; a < n - 4; a++)
    for (let b = a + 1; b < n - 3; b++)
      for (let c = b + 1; c < n - 2; c++)
        for (let d = c + 1; d < n - 1; d++)
          for (let e = d + 1; e < n; e++) combos.push([a, b, c, d, e]);
  combosCache.set(n, combos);
  return combos;
}

// Highest rank that completes a straight for a given set of present ranks,
// or 0 if there is no straight. Handles the wheel (A-2-3-4-5, high card 5):
// when checking a 5-high straight, rank 1 is treated as an ace.
function straightHigh(ranks: number[]): number {
  const present = new Set(ranks);
  for (let high = 14; high >= 5; high--) {
    let ok = true;
    for (let r = high; r > high - 5; r--) {
      const rank = r === 1 ? 14 : r;
      if (!present.has(rank)) {
        ok = false;
        break;
      }
    }
    if (ok) return high;
  }
  return 0;
}

// Score a single 5-card hand.
function score5(cards: Card[]): Score {
  const ranks = cards.map((c) => c.rank).sort((a, b) => b - a);
  const suits = cards.map((c) => c.suit);
  const isFlush = suits.every((s) => s === suits[0]);

  // Rank -> count, plus a strength-ordered list of [rank, count].
  const counts = new Map<number, number>();
  for (const r of ranks) counts.set(r, (counts.get(r) || 0) + 1);
  const groups = [...counts.entries()].sort((a, b) => {
    if (b[1] !== a[1]) return b[1] - a[1]; // by count first
    return b[0] - a[0]; // then by rank
  });

  const straight = straightHigh(ranks);

  if (isFlush && straight) return [CATEGORY.STRAIGHT_FLUSH, straight];

  if (groups[0][1] === 4) {
    const quad = groups[0][0];
    const kicker = groups.find((g) => g[0] !== quad)![0];
    return [CATEGORY.QUADS, quad, kicker];
  }

  if (groups[0][1] === 3 && groups[1] && groups[1][1] >= 2) {
    return [CATEGORY.FULL_HOUSE, groups[0][0], groups[1][0]];
  }

  if (isFlush) return [CATEGORY.FLUSH, ...ranks];

  if (straight) return [CATEGORY.STRAIGHT, straight];

  if (groups[0][1] === 3) {
    const trip = groups[0][0];
    const kickers = ranks.filter((r) => r !== trip);
    return [CATEGORY.TRIPS, trip, ...kickers];
  }

  if (groups[0][1] === 2 && groups[1] && groups[1][1] === 2) {
    const highPair = groups[0][0];
    const lowPair = groups[1][0];
    const kicker = groups.find((g) => g[1] === 1)![0];
    return [CATEGORY.TWO_PAIR, highPair, lowPair, kicker];
  }

  if (groups[0][1] === 2) {
    const pair = groups[0][0];
    const kickers = ranks.filter((r) => r !== pair);
    return [CATEGORY.PAIR, pair, ...kickers];
  }

  return [CATEGORY.HIGH_CARD, ...ranks];
}

// Lexicographic comparison of two score arrays. >0 if a is stronger.
export function compareScores(a: Score, b: Score): number {
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) {
    const x = a[i] ?? 0;
    const y = b[i] ?? 0;
    if (x !== y) return x - y;
  }
  return 0;
}

// Best 5-card score out of 7 (or fewer, for defensive use).
export function evaluate7(cards: Card[]): Score {
  if (cards.length <= 5) return score5(cards);
  let best: Score | null = null;
  for (const combo of fiveOf(cards.length)) {
    const hand = combo.map((i) => cards[i]);
    const s = score5(hand);
    if (!best || compareScores(s, best) > 0) best = s;
  }
  return best!;
}

export function categoryName(score: Score): string {
  return CATEGORY_NAMES[score[0]];
}
