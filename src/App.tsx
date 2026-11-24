import { useState } from "react";
import { useGameState } from "./hooks/useGameState";
import GameTable from "./components/GameTable";

const DEBUG_KEY = "6max-debug";

function loadDebug(): boolean {
  try {
    return localStorage.getItem(DEBUG_KEY) === "1";
  } catch {
    return false;
  }
}

export default function App() {
  const {
    gameState,
    dealHand,
    performAction,
    newGame,
    retry,
    gameOver,
    isProcessingBots,
    engineError,
    decisions,
  } = useGameState();
  const [debug, setDebug] = useState(loadDebug);

  const toggleDebug = () => {
    const next = !debug;
    setDebug(next);
    try {
      localStorage.setItem(DEBUG_KEY, next ? "1" : "0");
    } catch {
      // storage unavailable: the toggle still works for this page
    }
  };

  return (
    <GameTable
      gameState={gameState}
      onDeal={dealHand}
      onAction={performAction}
      gameOver={gameOver}
      onNewGame={newGame}
      isProcessingBots={isProcessingBots}
      faceUp={debug}
      debug={debug}
      onToggleDebug={toggleDebug}
      decisions={decisions}
      engineError={engineError}
      onRetry={retry}
    />
  );
}
