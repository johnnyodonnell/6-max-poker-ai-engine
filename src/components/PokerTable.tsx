import type { ReactNode } from "react";
import "./PokerTable.css";

interface PokerTableProps {
  children: ReactNode;
}

// The rail + felt oval. Everything on the table is positioned absolutely on
// the stage, so this just draws the furniture behind it.
export default function PokerTable({ children }: PokerTableProps) {
  return (
    <>
      <div className="table-rail">
        <div className="table-felt" />
      </div>
      {children}
    </>
  );
}
