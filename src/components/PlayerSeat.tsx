import type { Player as PlayerType, Phase } from "../types";
import Card from "./Card";
import { SEAT_POS, at } from "./layout";
import "./PlayerSeat.css";

interface PlayerSeatProps {
  player: PlayerType;
  isWinner: boolean;
  phase: Phase;
  position: number;
  isActing: boolean;
  showdown: boolean;
  faceUp?: boolean;
}

export default function PlayerSeat({
  player,
  isWinner,
  phase,
  position,
  isActing,
  showdown,
  faceUp,
}: PlayerSeatProps) {
  const folded = player.status === "folded";
  const out = player.status === "eliminated";

  // Reveal opponents' cards only at a real showdown. A hand won because everyone
  // else folded is not a showdown, so the winner keeps their cards concealed.
  const revealAll = showdown || phase === "showdown" ? !folded && !out : false;
  const showCards = faceUp ? !out : revealAll || player.isHuman;
  // Like CoinPoker, a folded hand leaves the table.
  const hasCards = player.holeCards.length > 0 && !folded && !out;

  const classNames = [
    "seat",
    player.isHuman ? "is-human" : "",
    out ? "eliminated" : "",
    folded ? "folded" : "",
    isWinner ? "is-winner" : "",
    isActing ? "is-acting" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={classNames} style={at(SEAT_POS[position])}>
      {hasCards && (
        <div className="seat-cards">
          {player.holeCards.map((card, i) => (
            <Card
              key={i}
              card={card}
              faceDown={!showCards}
              size={player.isHuman ? "lg" : "sm"}
            />
          ))}
        </div>
      )}
      <div className="seat-plate">
        <div className="seat-name">{player.name}</div>
        <div className="seat-stack">{player.chips.toLocaleString()}</div>
      </div>
      {isActing && <div className="seat-timer" />}
    </div>
  );
}
