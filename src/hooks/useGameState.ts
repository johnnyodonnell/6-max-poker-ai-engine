import { useState, useCallback, useEffect, useRef } from "react";
import type { BetAction, GameState } from "../types";
import {
  createTable,
  startHand,
  applyAction,
  advanceStreet,
} from "../engine/game";
import type { EngineState } from "../engine/game";
import { loadTable, saveTable } from "../engine/saved";
// The bots' engine. Swap this one import to plug in a different engine
// (engine/random.ts is the original random bot, not served).
import { bestMove, logHand, newMatch, onDecision } from "../engine/remote";
import type { BotDecision } from "../engine/remote";
import { toGameState, fromBetAction } from "../engine/adapt";

// Thrown when a bot can't move; carries the state reached so far (e.g. a
// street dealt before the bot's decision) so the table shows where it stopped.
class AdvanceError extends Error {
  constructor(
    message: string,
    public state: EngineState,
  ) {
    super(message);
  }
}

// Run the engine forward until a human decision is required or the hand/match
// ends. Bots' moves are awaited (they come from the engine service), but there
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
        let action;
        try {
          action = await bestMove(s);
        } catch (err) {
          throw new AdvanceError(err instanceof Error ? err.message : String(err), s);
        }
        s = applyAction(s, action);
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
  retry: () => void;
  gameOver: boolean;
  isProcessingBots: boolean;
  engineError: string | null;
  decisions: BotDecision[];
} {
  const [engine, setEngine] = useState<EngineState>(loadTable);
  const [busy, setBusy] = useState(false);
  const [engineError, setEngineError] = useState<string | null>(null);
  const [decisions, setDecisions] = useState<BotDecision[]>([]);
  const engineRef = useRef(engine);
  // Bumped by every new run (and New Game): a bot run that finishes after
  // it has been superseded is discarded.
  const generation = useRef(0);

  // The bots' decisions this hand, for the debug view.
  useEffect(
    () =>
      onDecision((d) =>
        setDecisions((prev) => (prev.length && prev[0].hand !== d.hand ? [d] : [...prev, d])),
      ),
    [],
  );

  const set = useCallback((s: EngineState) => {
    engineRef.current = s;
    setEngine(s);
    saveTable(s);
  }, []);

  // Show `start`, then let the bots play until the human is up again.
  const run = useCallback(
    async (start: EngineState) => {
      const my = ++generation.current;
      set(start);
      setBusy(true);
      setEngineError(null);
      try {
        const end = await advance(start);
        if (generation.current !== my) return;
        set(end);
        logHand(end);
      } catch (err) {
        if (generation.current !== my) return;
        if (err instanceof AdvanceError) {
          set(err.state);
          setEngineError(err.message);
        } else {
          setEngineError(String(err));
        }
      }
      setBusy(false);
    },
    [set],
  );

  const dealHand = useCallback(() => {
    const s = engineRef.current;
    if (busy || engineError || (s.phase !== "waiting" && s.phase !== "hand-over")) return;
    setDecisions([]);
    void run(startHand(s));
  }, [busy, engineError, run]);

  const performAction = useCallback(
    (action: BetAction) => {
      const s = engineRef.current;
      if (busy || engineError || s.phase !== "betting" || !s.players[s.toAct]?.isHuman) return;
      void run(applyAction(s, fromBetAction(action, s)));
    },
    [busy, engineError, run],
  );

  // Ask the engine again from where the bot was stuck.
  const retry = useCallback(() => {
    if (busy) return;
    void run(engineRef.current);
  }, [busy, run]);

  const newGame = useCallback(() => {
    generation.current++;
    newMatch();
    set(createTable());
    setBusy(false);
    setEngineError(null);
    setDecisions([]);
  }, [set]);

  return {
    gameState: toGameState(engine),
    dealHand,
    performAction,
    newGame,
    retry,
    gameOver: engine.phase === "match-over",
    isProcessingBots: busy,
    engineError,
    decisions,
  };
}
