import { DEALER_POS, at } from "./layout";
import "./DealerButton.css";

export default function DealerButton({ seat }: { seat: number }) {
  if (seat < 0) return null;
  return (
    <div className="dealer-button" style={at(DEALER_POS[seat])}>
      D
    </div>
  );
}
