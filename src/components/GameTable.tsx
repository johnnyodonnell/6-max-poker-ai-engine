import { useEffect, useState } from "react";
import type { GameState, BetAction } from "../types";
import PlayerSeat from "./PlayerSeat";
import PokerTable from "./PokerTable";
import CommunityCards from "./CommunityCards";
import PotDisplay from "./PotDisplay";
import ChipStack from "./ChipStack";
import DealerButton from "./DealerButton";
import ActionPanel from "./ActionPanel";
import HandResult from "./HandResult";
import DebugPanel from "./DebugPanel";
import type { BotDecision } from "../engine/remote";
import { STAGE_WIDTH, STAGE_HEIGHT, BET_POS } from "./layout";
import "./GameTable.css";

interface GameTableProps {
  gameState: GameState;
  onDeal: () => void;
  onAction: (action: BetAction) => void;
  gameOver: boolean;
  onNewGame: () => void;
  isProcessingBots?: boolean;
  faceUp?: boolean;
  // Debug view: the toggle, and the bots' decisions this hand.
  debug?: boolean;
  onToggleDebug?: () => void;
  decisions?: BotDecision[];
  // Set when a bot couldn't move (engine unreachable / error).
  engineError?: string | null;
  onRetry?: () => void;
}

// Window width kept clear on the left for the open debug panel (its width
// plus margins; see DebugPanel.css), so it never covers the left-hand seats.
const DEBUG_SIDEBAR = 410;

// Scale the fixed-size stage to fit the window (less `reserved` pixels on the
// left), preserving its aspect ratio.
function useStageScale(reserved: number): number {
  const compute = () =>
    Math.min((window.innerWidth - reserved) / STAGE_WIDTH, window.innerHeight / STAGE_HEIGHT);
  const [scale, setScale] = useState(compute);
  useEffect(() => {
    const onResize = () => setScale(compute());
    onResize();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [reserved]);
  return scale;
}

export default function GameTable({
  gameState,
  onDeal,
  onAction,
  gameOver,
  onNewGame,
  isProcessingBots = false,
  faceUp = false,
  debug = false,
  onToggleDebug,
  decisions = [],
  engineError = null,
  onRetry,
}: GameTableProps) {
  const reserved = debug && onToggleDebug ? DEBUG_SIDEBAR : 0;
  const scale = useStageScale(reserved);
  const winnerSeats = new Set(gameState.winners.map((w) => w.seat));

  const currentBetsTotal = gameState.players.reduce(
    (sum, p) => sum + p.currentBet,
    0,
  );
  const liveTotalPot = gameState.totalPot + currentBetsTotal;
  const handOver = gameState.phase === "hand_complete";

  return (
    <div className="viewport" style={{ paddingLeft: reserved }}>
      {onToggleDebug && (
        <DebugPanel enabled={debug} onToggle={onToggleDebug} decisions={decisions} />
      )}

      <div
        className="stage"
        style={{
          width: STAGE_WIDTH,
          height: STAGE_HEIGHT,
          transform: `scale(${scale})`,
        }}
      >
        <PokerTable>
          <PotDisplay
            pots={gameState.pots}
            collected={gameState.totalPot}
            liveTotal={liveTotalPot}
          />
          <CommunityCards cards={gameState.communityCards} />

          {gameState.players.map((p) =>
            p.currentBet > 0 && !handOver ? (
              <ChipStack key={p.seat} amount={p.currentBet} pos={BET_POS[p.seat]} />
            ) : null,
          )}

          {gameState.handNumber > 0 && <DealerButton seat={gameState.dealerSeat} />}

          {gameState.players.map((p) => (
            <PlayerSeat
              key={p.seat}
              player={p}
              isWinner={winnerSeats.has(p.seat)}
              phase={gameState.phase}
              position={p.seat}
              isActing={gameState.actionOnSeat === p.seat}
              showdown={gameState.showdown}
              faceUp={faceUp}
            />
          ))}
          {handOver && (
            <HandResult
              winners={gameState.winners}
              players={gameState.players}
              gameOverMessage={gameState.gameOverMessage}
            />
          )}
        </PokerTable>

        {engineError && (
          <div className="engine-error">
            <span>The bots can't move: {engineError}</span>
            {onRetry && <button onClick={onRetry}>Retry</button>}
          </div>
        )}

        <div className="action-dock">
          <ActionPanel
            phase={gameState.phase}
            humanCanAct={gameState.humanCanAct}
            canCheck={gameState.canCheck}
            callAmount={gameState.callAmount}
            minRaiseAmount={gameState.minRaiseAmount}
            maxRaiseAmount={gameState.maxRaiseAmount}
            onAction={onAction}
            onDeal={onDeal}
            gameOver={gameOver}
            onNewGame={onNewGame}
            isProcessingBots={isProcessingBots}
          />
        </div>
      </div>
    </div>
  );
}
