import { useState, useCallback } from "react";
import type { BetAction, GameState } from "../types";
import {
  createTable,
  startHand,
  applyAction,
  advanceStreet,
} from "../engine/game";
import type { EngineState } from "../engine/game";
import { bestMove } from "../engine/random";
import { toGameState, fromBetAction } from "../engine/adapt";

// Run the engine forward until a human decision is required or the hand/match
// ends. Instant pacing: bots and street run-outs resolve with no delay (the
// reference relied on network latency, which we don't have). Bounded by a
// generous step cap purely as a runaway-loop backstop.
function advance(state: EngineState): EngineState {
  let s = state;
  for (let step = 0; step < 10000; step++) {
    if (s.phase === "street-complete") {
      s = advanceStreet(s);
      continue;
    }
    if (s.phase === "betting") {
      const actor = s.players[s.toAct];
      if (actor && !actor.isHuman) {
        s = applyAction(s, bestMove(s));
        continue;
      }
    }
    break; // human to act, or hand-over / match-over / waiting
  }
  return s;
}

export function useGameState(): {
  gameState: GameState;
  dealHand: () => void;
  performAction: (action: BetAction) => void;
  newGame: () => void;
  gameOver: boolean;
  isProcessingBots: boolean;
} {
  const [engine, setEngine] = useState<EngineState>(() => createTable());

  const dealHand = useCallback(() => {
    setEngine((s) =>
      s.phase === "waiting" || s.phase === "hand-over"
        ? advance(startHand(s))
        : s,
    );
  }, []);

  const performAction = useCallback((action: BetAction) => {
    setEngine((s) => {
      if (s.phase !== "betting" || !s.players[s.toAct]?.isHuman) return s;
      return advance(applyAction(s, fromBetAction(action, s)));
    });
  }, []);

  const newGame = useCallback(() => {
    setEngine(createTable());
  }, []);

  return {
    gameState: toGameState(engine),
    dealHand,
    performAction,
    newGame,
    gameOver: engine.phase === "match-over",
    isProcessingBots: false, // instant — bots never leave a pending state
  };
}
