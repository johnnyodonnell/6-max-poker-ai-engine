import type { Card as CardType } from "../types";
import Card from "./Card";
import { BOARD_POS, at } from "./layout";
import "./CommunityCards.css";

interface CommunityCardsProps {
  cards: CardType[];
}

export default function CommunityCards({ cards }: CommunityCardsProps) {
  if (cards.length === 0) return null;

  return (
    <div className="community-cards" style={at(BOARD_POS)}>
      {cards.map((card) => (
        <Card key={card.label} card={card} />
      ))}
    </div>
  );
}
