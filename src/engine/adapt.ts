// Bridges the internal engine (engine/game.ts) to the UI's GameState contract
// (types.ts). The components only ever see the mapped GameState; the engine
// stays the source of truth.

import type {
  EngineState,
  EngineStatus,
  EnginePlayer,
  Action,
} from "./game";
import { legalActions, buildPots, matchWinner } from "./game";
import { evaluate7, categoryName } from "./evaluator";
import type {
  GameState,
  Player,
  PlayerStatus,
  Phase,
  Card as UICard,
  SidePot,
  BetAction,
} from "../types";
import type { Card } from "./cards";

function mapStatus(status: EngineStatus): PlayerStatus {
  switch (status) {
    case "allin":
      return "all_in";
    case "out":
      return "eliminated";
    default:
      return status; // 'active' | 'folded'
  }
}

function mapPhase(s: EngineState): Phase {
  if (s.phase === "waiting") return "waiting";
  if (s.phase === "hand-over" || s.phase === "match-over") return "hand_complete";
  return s.street; // 'preflop' | 'flop' | 'turn' | 'river'
}

function mapCard(c: Card): UICard {
  return { rank: c.rank, suit: c.suit, label: c.id };
}

export function toGameState(s: EngineState): GameState {
  // A hand that ends the match (phase 'match-over') can still be a showdown.
  const showdown =
    (s.phase === "hand-over" || s.phase === "match-over") && !!s.results?.reveal;

  const players: Player[] = s.players.map((p: EnginePlayer) => {
    const inHand = p.status === "active" || p.status === "allin";
    const handDescription =
      showdown && inHand
        ? categoryName(evaluate7([...p.holeCards, ...s.community]))
        : undefined;
    return {
      seat: p.id,
      name: p.name,
      chips: p.stack,
      isHuman: p.isHuman,
      status: mapStatus(p.status),
      holeCards: p.holeCards.map(mapCard),
      currentBet: p.bet,
      totalBetThisHand: p.commit,
      isDealer: s.button === p.id,
      handDescription,
      lastAction: p.lastAction,
      blind:
        s.handNumber > 0 && s.street === "preflop" && s.phase === "betting"
          ? p.id === s.bbSeat
            ? "BB"
            : p.id === s.sbSeat
              ? "SB"
              : null
          : null,
    };
  });

  // Side-pot breakdown is only meaningful (and only shown) once someone is
  // all-in; otherwise the SB/BB size difference would surface as a spurious
  // "side pot" every hand. Off-showdown we show one consolidated pot.
  const anyAllIn = s.players.some((p) => p.status === "allin");
  const pots: SidePot[] = anyAllIn
    ? buildPots(s.players).map((pot) => ({
        amount: pot.amount,
        eligibleSeats: pot.eligible,
      }))
    : [
        {
          amount: s.players.reduce((sum, p) => sum + p.commit, 0),
          eligibleSeats: s.players
            .filter((p) => p.status === "active" || p.status === "allin")
            .map((p) => p.id),
        },
      ];

  const humanToAct =
    s.phase === "betting" && s.players[s.toAct]?.isHuman === true;
  const legal = humanToAct ? legalActions(s) : null;
  const canRaise =
    !!legal &&
    (legal.actions.includes("bet") || legal.actions.includes("raise"));

  return {
    handNumber: s.handNumber,
    phase: mapPhase(s),
    communityCards: s.community.map(mapCard),
    players,
    pots,
    totalPot: s.pot,
    dealerSeat: s.button,
    humanSeat: 0,
    showdown,
    winners: s.results?.winners ?? [],
    actionOnSeat: s.phase === "betting" ? s.toAct : -1,
    humanCanAct: humanToAct,
    canCheck: legal?.canCheck ?? false,
    callAmount: legal?.toCall ?? 0,
    minRaiseAmount: canRaise ? legal!.minRaiseTo : 0,
    maxRaiseAmount: canRaise ? legal!.maxRaiseTo : 0,
    humanHandDescription: humanHand(s),
    gameOverMessage: gameOverMessage(s),
  };
}

function humanHand(s: EngineState): string | null {
  const me = s.players.find((p) => p.isHuman);
  if (!me || me.holeCards.length === 0) return null;
  if (me.status !== "active" && me.status !== "allin") return null;
  return categoryName(evaluate7([...me.holeCards, ...s.community]));
}

function gameOverMessage(s: EngineState): string | null {
  if (s.phase !== "match-over") return null;
  const me = s.players.find((p) => p.isHuman);
  if (me && me.stack === 0) return "You busted — game over";
  const winner = matchWinner(s);
  if (!winner) return "Game over";
  return winner.isHuman ? "You win the match!" : `${winner.name} wins the match`;
}

// The UI only emits fold/check/call/raise (an opening bet is a "raise"). Map
// that onto my engine's action, which distinguishes an opening 'bet' from a
// 'raise'. `amount` is a raise-to total in both models.
export function fromBetAction(a: BetAction, s: EngineState): Action {
  switch (a.type) {
    case "fold":
      return { type: "fold" };
    case "check":
      return { type: "check" };
    case "call":
      return { type: "call" };
    case "raise":
      return {
        type: s.currentBet === 0 ? "bet" : "raise",
        amount: a.amount,
      };
  }
}
