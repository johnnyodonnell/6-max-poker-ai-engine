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
}

// Scale the fixed-size stage to fit the window, preserving its aspect ratio.
function useStageScale(): number {
  const compute = () =>
    Math.min(window.innerWidth / STAGE_WIDTH, window.innerHeight / STAGE_HEIGHT);
  const [scale, setScale] = useState(compute);
  useEffect(() => {
    const onResize = () => setScale(compute());
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);
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
}: GameTableProps) {
  const scale = useStageScale();
  const winnerSeats = new Set(gameState.winners.map((w) => w.seat));

  const currentBetsTotal = gameState.players.reduce(
    (sum, p) => sum + p.currentBet,
    0,
  );
  const liveTotalPot = gameState.totalPot + currentBetsTotal;
  const handOver = gameState.phase === "hand_complete";

  return (
    <div className="viewport">
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
              handStrength={p.isHuman ? gameState.humanHandDescription : null}
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
