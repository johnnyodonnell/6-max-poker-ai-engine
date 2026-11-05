import type { Winner, Player } from "../types";
import { RESULT_POS, at } from "./layout";
import "./HandResult.css";

interface HandResultProps {
  winners: Winner[];
  players: Player[];
  gameOverMessage: string | null;
}

export default function HandResult({ winners, players, gameOverMessage }: HandResultProps) {
  if (winners.length === 0 && !gameOverMessage) return null;

  return (
    <div className="hand-result" style={at(RESULT_POS)}>
      {winners.map((w, i) => {
        const player = players[w.seat];
        const verb = player.isHuman ? "win" : "wins";
        const desc =
          w.handDescription === "opponents folded"
            ? "(opponents folded)"
            : `with ${w.handDescription}`;
        return (
          <div key={i} className="winner-line">
            {player.name} {verb} {w.amountWon.toLocaleString()} {desc}
          </div>
        );
      })}
      {gameOverMessage && <div className="game-over-line">{gameOverMessage}</div>}
    </div>
  );
}
