import { useState, useCallback, useEffect, useRef } from "react";
import type { BetAction, GameState } from "../types";
import {
  createTable,
  startHand,
  applyAction,
  advanceStreet,
} from "../engine/game";
import type { EngineState } from "../engine/game";
import { bestMove, preload } from "../engine/neural";
import { toGameState, fromBetAction } from "../engine/adapt";

// Run the engine forward until a human decision is required or the hand/match
// ends. The bots' moves are async (one ONNX forward each, a few ms), but there
// is no artificial pacing: the table updates once, when the run-out is done.
// Bounded by a generous step cap purely as a runaway-loop backstop.
async function advance(state: EngineState): Promise<EngineState> {
  let s = state;
  for (let step = 0; step < 10000; step++) {
    if (s.phase === "street-complete") {
      s = advanceStreet(s);
      continue;
    }
    if (s.phase === "betting") {
      const actor = s.players[s.toAct];
      if (actor && !actor.isHuman) {
        s = applyAction(s, await bestMove(s));
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
  const [busy, setBusy] = useState(false);
  const engineRef = useRef(engine);
  // Bumped by every new run (and New Game): a bot run that finishes after
  // it has been superseded is discarded.
  const generation = useRef(0);

  useEffect(() => {
    preload().catch((err) => console.error("PokerNet: model failed to load", err));
  }, []);

  const set = useCallback((s: EngineState) => {
    engineRef.current = s;
    setEngine(s);
  }, []);

  // Show `start`, then let the bots play until the human is up again.
  const run = useCallback(
    async (start: EngineState) => {
      const my = ++generation.current;
      set(start);
      setBusy(true);
      const end = await advance(start);
      if (generation.current !== my) return;
      set(end);
      setBusy(false);
    },
    [set],
  );

  const dealHand = useCallback(() => {
    const s = engineRef.current;
    if (busy || (s.phase !== "waiting" && s.phase !== "hand-over")) return;
    void run(startHand(s));
  }, [busy, run]);

  const performAction = useCallback(
    (action: BetAction) => {
      const s = engineRef.current;
      if (busy || s.phase !== "betting" || !s.players[s.toAct]?.isHuman) return;
      void run(applyAction(s, fromBetAction(action, s)));
    },
    [busy, run],
  );

  const newGame = useCallback(() => {
    generation.current++;
    set(createTable());
    setBusy(false);
  }, [set]);

  return {
    gameState: toGameState(engine),
    dealHand,
    performAction,
    newGame,
    gameOver: engine.phase === "match-over",
    isProcessingBots: busy,
  };
}
