import { dollars } from "../money";
import type { Player as PlayerType, Phase } from "../types";
import Card from "./Card";
import { SEAT_POS, at } from "./layout";
import "./PlayerSeat.css";

const ACTION_LABELS: Record<NonNullable<PlayerType["lastAction"]>, string> = {
  fold: "Fold",
  check: "Check",
  call: "Call",
  bet: "Bet",
  raise: "Raise",
  allin: "All-In",
};

// The tab above the plate: the last action this street, a FOLD that sticks for
// the rest of the hand, ALL-IN, or the SB/BB badge until the blind acts.
function seatLabel(player: PlayerType): { text: string; kind: string } | null {
  if (player.status === "eliminated") return { text: "Out", kind: "muted" };
  if (player.status === "folded") return { text: "Fold", kind: "muted" };
  if (player.lastAction) {
    const a = player.lastAction;
    const kind =
      a === "allin" ? "allin" : a === "bet" || a === "raise" ? "aggressive" : "passive";
    return { text: ACTION_LABELS[a], kind };
  }
  if (player.status === "all_in") return { text: "All-In", kind: "allin" };
  if (player.blind) return { text: player.blind, kind: "blind" };
  return null;
}

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
  // Like CoinPoker, a folded hand leaves the table...
  const hasCards = player.holeCards.length > 0 && !folded && !out;
  // ...but you can still peek at your own folded cards by hovering the seat.
  const canPeek = player.holeCards.length > 0 && folded && showCards;

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

  const label = seatLabel(player);

  return (
    <div className={classNames} style={at(SEAT_POS[position])}>
      {(hasCards || canPeek) && (
        <div className={canPeek ? "seat-cards folded-peek" : "seat-cards"}>
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
        <div className="seat-stack">{dollars(player.chips)}</div>
      </div>
      {label && <div className={`seat-label label-${label.kind}`}>{label.text}</div>}
    </div>
  );
}
