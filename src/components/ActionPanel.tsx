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

// Chip denominations, smallest first (one chip unit is one dollar).
const DENOMINATIONS = [1, 5, 25, 100] as const;

function Chip({
  value,
  onClick,
  disabled = false,
  label,
}: {
  value: number;
  onClick: () => void;
  disabled?: boolean;
  label: string;
}) {
  return (
    <button
      type="button"
      className={`bet-chip bet-chip-${value}`}
      onClick={onClick}
      disabled={disabled}
      aria-label={label}
      title={label}
    >
      <span className="bet-chip-face">{value}</span>
    </button>
  );
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
  // The "raise to" / "bet" total, kept between the minimum raise and all-in.
  const [raiseAmount, setRaiseAmount] = useState(minRaiseAmount);

  // Start every decision at the minimum raise.
  useEffect(() => {
    setRaiseAmount(minRaiseAmount);
  }, [phase, humanCanAct, isProcessingBots, minRaiseAmount, maxRaiseAmount]);

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

  if (isProcessingBots) {
    // While the bots move, the controls stay put, greyed out and without
    // amounts (what you'll face depends on their moves), rather than vanishing
    // and reappearing.
    return (
      <div className="action-panel is-waiting" aria-busy="true">
        {(["Take chips off the bet", "Add chips to the bet"] as const).map((rowLabel, r) => (
          <div key={rowLabel} className="chip-row" aria-label={rowLabel}>
            <span className="chip-row-sign">{r === 0 ? "−" : "+"}</span>
            {DENOMINATIONS.map((value) => (
              <Chip key={value} value={value} onClick={() => {}} disabled label={rowLabel} />
            ))}
          </div>
        ))}
        <div className="action-buttons">
          <button className="btn btn-fold" disabled>
            Fold
          </button>
          <button className="btn btn-call" disabled>
            Call
          </button>
          <button className="btn btn-raise" disabled>
            Raise
          </button>
        </div>
      </div>
    );
  }

  if (!humanCanAct) {
    return null;
  }

  const canRaise = minRaiseAmount > 0 && maxRaiseAmount > 0;
  // With a choice of sizes, chips step the bet up and down; otherwise the
  // only raise is all-in.
  const showChips = canRaise && minRaiseAmount < maxRaiseAmount;
  // Nothing to call after the flop means this is an opening bet, not a raise.
  const raiseVerb = canCheck && phase !== "preflop" ? "Bet" : "Raise to";
  const allIn = raiseAmount >= maxRaiseAmount;

  // Room left to move the bet up (to all-in) and down (to the minimum raise).
  const roomUp = maxRaiseAmount - raiseAmount;
  const roomDown = raiseAmount - minRaiseAmount;
  // A chip works if it fits in the room, and so does the smallest chip that
  // reaches or passes the end, which stops the bet there: adding goes all-in,
  // removing goes back to the minimum raise. Larger chips are dimmed.
  const usable = (index: number, room: number) =>
    room > 0 && (index === 0 || DENOMINATIONS[index - 1] < room);
  const addChip = (value: number) =>
    setRaiseAmount((r) => Math.min(r + value, maxRaiseAmount));
  const removeChip = (value: number) =>
    setRaiseAmount((r) => Math.max(r - value, minRaiseAmount));

  return (
    <div className="action-panel">
      {showChips && (
        <>
          <div className="chip-row" aria-label="Take chips off the bet">
            <span className="chip-row-sign">−</span>
            {DENOMINATIONS.map((value, i) => (
              <Chip
                key={value}
                value={value}
                onClick={() => removeChip(value)}
                disabled={!usable(i, roomDown)}
                label={value >= roomDown ? "Back to the minimum raise" : `Remove ${dollars(value)}`}
              />
            ))}
          </div>
          <div className="chip-row" aria-label="Add chips to the bet">
            <span className="chip-row-sign">+</span>
            {DENOMINATIONS.map((value, i) => (
              <Chip
                key={value}
                value={value}
                onClick={() => addChip(value)}
                disabled={!usable(i, roomUp)}
                label={value >= roomUp ? "All-in" : `Add ${dollars(value)}`}
              />
            ))}
          </div>
        </>
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
          (showChips ? (
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
