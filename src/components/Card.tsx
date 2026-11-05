import type { Card as CardType } from "../types";
import "./Card.css";

const SUIT_SYMBOLS: Record<CardType["suit"], string> = {
  h: "♥",
  d: "♦",
  c: "♣",
  s: "♠",
};

const RANK_DISPLAY: Record<number, string> = {
  11: "J",
  12: "Q",
  13: "K",
  14: "A",
};

interface CardProps {
  card?: CardType;
  faceDown?: boolean;
  size?: "sm" | "lg";
}

export default function Card({ card, faceDown, size = "lg" }: CardProps) {
  if (faceDown || !card) {
    return <div className={`card card-${size} face-down`} />;
  }

  const isRed = card.suit === "h" || card.suit === "d";
  const suit = SUIT_SYMBOLS[card.suit];

  return (
    <div className={`card card-${size} ${isRed ? "red" : "black"}`}>
      <span className="card-rank">{RANK_DISPLAY[card.rank] ?? card.rank}</span>
      <span className="card-suit-small">{suit}</span>
      <span className="card-suit-big">{suit}</span>
    </div>
  );
}
