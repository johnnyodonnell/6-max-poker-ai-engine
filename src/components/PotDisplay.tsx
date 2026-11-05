import { dollars } from "../money";
import type { SidePot } from "../types";
import ChipStack from "./ChipStack";
import { POT_LABEL_POS, SIDE_POTS_POS, COLLECTED_POT_POS, at } from "./layout";
import "./PotDisplay.css";

interface PotDisplayProps {
  pots: SidePot[];
  // Chips gathered from earlier streets (shown as a stack under the board).
  collected: number;
  // Everything in the middle including this street's bets ("Pot N" label).
  liveTotal: number;
}

function potName(i: number): string {
  if (i === 0) return "Main pot";
  return i === 1 ? "Side pot" : `Side pot ${i}`;
}

export default function PotDisplay({ pots, collected, liveTotal }: PotDisplayProps) {
  if (liveTotal === 0) return null;

  return (
    <>
      <div className="pot-label" style={at(POT_LABEL_POS)}>
        <span className="pot-word">Pot</span> {dollars(liveTotal)}
      </div>
      {pots.length > 1 && (
        <div className="side-pots" style={at(SIDE_POTS_POS)}>
          {pots.map((pot, i) => (
            <span key={i} className="side-pot">
              <span className="side-pot-name">{potName(i)}:</span>{" "}
              <span className="side-pot-amount">{dollars(pot.amount)}</span>
            </span>
          ))}
        </div>
      )}
      {collected > 0 && <ChipStack amount={collected} pos={COLLECTED_POT_POS} />}
    </>
  );
}
