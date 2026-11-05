import { dollars } from "../money";
import { useState, useEffect } from "react";
import type { Phase, BetAction } from "../types";
import "./ActionPanel.css";

interface ActionPanelProps {
  phase: Phase;
  humanCanAct: boolean;
  canCheck: boolean;
  callAmount: number;
  minRaiseAmount: number;
  maxRaiseAmount: number;
  onAction: (action: BetAction) => void;
  onDeal: () => void;
  gameOver: boolean;
  onNewGame: () => void;
  isProcessingBots: boolean;
}

export default function ActionPanel({
  phase,
  humanCanAct,
  canCheck,
  callAmount,
  minRaiseAmount,
  maxRaiseAmount,
  onAction,
  onDeal,
  gameOver,
  onNewGame,
  isProcessingBots,
}: ActionPanelProps) {
  const [raiseAmount, setRaiseAmount] = useState(minRaiseAmount);

  useEffect(() => {
    setRaiseAmount(minRaiseAmount);
  }, [minRaiseAmount]);

  if (gameOver) {
    return (
      <div className="action-panel">
        <button className="btn btn-deal" onClick={onNewGame}>
          New Game
        </button>
      </div>
    );
  }

  if (phase === "waiting" || phase === "hand_complete") {
    return (
      <div className="action-panel">
        <button className="btn btn-deal" onClick={onDeal}>
          Deal
        </button>
      </div>
    );
  }

  if (!humanCanAct || isProcessingBots) {
    return null;
  }

  const canRaise = minRaiseAmount > 0 && maxRaiseAmount > 0;
  const showSlider = canRaise && minRaiseAmount < maxRaiseAmount;
  // Nothing to call after the flop means this is an opening bet, not a raise.
  const raiseVerb = canCheck && phase !== "preflop" ? "Bet" : "Raise to";
  const allIn = raiseAmount >= maxRaiseAmount;

  return (
    <div className="action-panel">
      {showSlider && (
        <div className="raise-controls">
          <input
            type="range"
            className="raise-slider"
            min={minRaiseAmount}
            max={maxRaiseAmount}
            value={raiseAmount}
            onChange={(e) => setRaiseAmount(Number(e.target.value))}
          />
          <span className="raise-amount">{dollars(raiseAmount)}</span>
        </div>
      )}
      <div className="action-buttons">
        {canCheck ? (
          <button className="btn btn-check" onClick={() => onAction({ type: "check" })}>
            Check
          </button>
        ) : (
          <>
            <button className="btn btn-fold" onClick={() => onAction({ type: "fold" })}>
              Fold
            </button>
            <button className="btn btn-call" onClick={() => onAction({ type: "call" })}>
              Call {dollars(callAmount)}
            </button>
          </>
        )}

        {canRaise &&
          (showSlider ? (
            <button
              className="btn btn-raise"
              onClick={() => onAction({ type: "raise", amount: raiseAmount })}
            >
              {allIn ? "All-In" : raiseVerb} {dollars(raiseAmount)}
            </button>
          ) : (
            <button
              className="btn btn-raise"
              onClick={() => onAction({ type: "raise", amount: maxRaiseAmount })}
            >
              All-In {dollars(maxRaiseAmount)}
            </button>
          ))}
      </div>
    </div>
  );
}
