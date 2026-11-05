// The UI-facing game-state contract, ported from 8-max-poker-ai-engine. The
// components render off these types; the adapter (engine/adapt.ts) maps the
// engine's internal state into this shape.

export interface Card {
  rank: number; // 2-14 (2=Two, ..., 14=Ace)
  suit: "h" | "d" | "c" | "s";
  label: string; // e.g. "As", "Td", "2c"
}

export type PlayerStatus = "active" | "all_in" | "folded" | "eliminated";

export interface Player {
  seat: number;
  name: string;
  chips: number;
  isHuman: boolean;
  status: PlayerStatus;
  holeCards: Card[];
  currentBet: number;
  totalBetThisHand: number;
  isDealer: boolean;
  handDescription?: string;
  // Seat label: the player's last action this street, if any.
  lastAction: SeatAction | null;
  // Blind posted this hand, shown as a badge until the player acts.
  blind: "SB" | "BB" | null;
}

export type SeatAction = "fold" | "check" | "call" | "bet" | "raise" | "allin";

export interface SidePot {
  amount: number;
  eligibleSeats: number[];
}

export type Phase =
  | "waiting"
  | "preflop"
  | "flop"
  | "turn"
  | "river"
  | "showdown"
  | "hand_complete";

export type BetAction =
  | { type: "fold" }
  | { type: "check" }
  | { type: "call" }
  | { type: "raise"; amount: number };

export interface Winner {
  seat: number;
  amountWon: number;
  handDescription: string;
}

export interface GameState {
  handNumber: number;
  phase: Phase;
  communityCards: Card[];
  players: Player[];
  pots: SidePot[];
  totalPot: number;
  dealerSeat: number;
  humanSeat: number;
  // True only when the hand reached a real showdown. On a fold-win (everyone
  // else folded) it's false, so the winner's hole cards stay concealed.
  showdown: boolean;
  winners: Winner[];
  actionOnSeat: number;
  humanCanAct: boolean;
  minRaiseAmount: number;
  maxRaiseAmount: number;
  canCheck: boolean;
  callAmount: number;
  // The human's current best hand ("One Pair"), while they're in the hand.
  humanHandDescription: string | null;
  // Set once the match is over: who won, or that the human busted.
  gameOverMessage: string | null;
}
