// The 52-card deck plus small helpers. No React, no DOM.
//
// A card is a plain object: { rank, suit, id }.
//   rank: 2..14 (11=J, 12=Q, 13=K, 14=A)  — numeric so comparisons are trivial.
//   suit: 'c' | 'd' | 'h' | 's'
//   id:   e.g. 'Ah', '10s', '2c' — stable key for React and for logging.

export type Suit = "c" | "d" | "h" | "s";

export interface Card {
  rank: number;
  suit: Suit;
  id: string;
}

export const SUITS: Suit[] = ["c", "d", "h", "s"];
export const RANKS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];

// How a rank prints on a card face.
const RANK_LABELS: Record<number, string> = {
  11: "J",
  12: "Q",
  13: "K",
  14: "A",
};

export function rankLabel(rank: number): string {
  return RANK_LABELS[rank] ?? String(rank);
}

export const SUIT_SYMBOLS: Record<Suit, string> = {
  c: "♣",
  d: "♦",
  h: "♥",
  s: "♠",
};

export function cardId(rank: number, suit: Suit): string {
  return `${rankLabel(rank)}${suit}`;
}

export function cardLabel(card: Card): string {
  return `${rankLabel(card.rank)}${SUIT_SYMBOLS[card.suit]}`;
}

export function createDeck(): Card[] {
  const deck: Card[] = [];
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      deck.push({ rank, suit, id: cardId(rank, suit) });
    }
  }
  return deck;
}

// Fisher–Yates. Returns a fresh shuffled copy; leaves the input untouched.
export function shuffle(deck: Card[]): Card[] {
  const a = deck.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}
