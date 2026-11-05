import { useGameState } from "./hooks/useGameState";
import GameTable from "./components/GameTable";

export default function App() {
  const {
    gameState,
    dealHand,
    performAction,
    newGame,
    gameOver,
    isProcessingBots,
  } = useGameState();

  return (
    <GameTable
      gameState={gameState}
      onDeal={dealHand}
      onAction={performAction}
      gameOver={gameOver}
      onNewGame={newGame}
      isProcessingBots={isProcessingBots}
    />
  );
}
