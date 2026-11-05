import { at } from "./layout";
import type { Point } from "./layout";
import "./ChipStack.css";

interface ChipStackProps {
  amount: number;
  pos: Point;
}

// A red chip over an amount pill, centered on `pos` — used for each player's
// bet and for the pot gathered from earlier streets.
export default function ChipStack({ amount, pos }: ChipStackProps) {
  return (
    <div className="chip-stack" style={at(pos)}>
      <div className="chip" />
      <div className="chip-amount">{amount.toLocaleString()}</div>
    </div>
  );
}
