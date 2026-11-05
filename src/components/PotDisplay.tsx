import type { SidePot } from "../types";
import ChipStack from "./ChipStack";
import { POT_LABEL_POS, COLLECTED_POT_POS, at } from "./layout";
import "./PotDisplay.css";

interface PotDisplayProps {
  pots: SidePot[];
  // Chips gathered from earlier streets (shown as a stack under the board).
  collected: number;
  // Everything in the middle including this street's bets ("Pot N" label).
  liveTotal: number;
}

export default function PotDisplay({ pots, collected, liveTotal }: PotDisplayProps) {
  if (liveTotal === 0) return null;

  return (
    <>
      <div className="pot-label" style={at(POT_LABEL_POS)}>
        <span className="pot-word">Pot</span> {liveTotal.toLocaleString()}
        {pots.length > 1 && (
          <div className="side-pots">
            {pots.map((pot, i) => (
              <span key={i} className="side-pot">
                {i === 0 ? "Main" : `Side ${i}`} {pot.amount.toLocaleString()}
              </span>
            ))}
          </div>
        )}
      </div>
      {collected > 0 && <ChipStack amount={collected} pos={COLLECTED_POT_POS} />}
    </>
  );
}
