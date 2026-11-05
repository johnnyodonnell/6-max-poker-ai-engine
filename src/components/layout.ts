// Fixed-size "stage" geometry, modelled on the CoinPoker 6-max table. The
// whole stage is laid out in these design pixels and then scaled to fit the
// window (see GameTable), so every position here is absolute.
//
// Seats run clockwise from the human at bottom-center, matching the engine's
// seat order (action passes to seat i+1):
//
//            3
//      2           4
//
//      1           5
//            0

export const STAGE_WIDTH = 1340;
export const STAGE_HEIGHT = 940;

export interface Point {
  x: number;
  y: number;
}

// Center of each seat's name/stack plate.
export const SEAT_POS: Point[] = [
  { x: 670, y: 845 },
  { x: 160, y: 655 },
  { x: 185, y: 255 },
  { x: 670, y: 172 },
  { x: 1155, y: 255 },
  { x: 1180, y: 655 },
];

// Where each seat's current-street bet sits on the felt.
export const BET_POS: Point[] = [
  { x: 670, y: 628 },
  { x: 345, y: 595 },
  { x: 365, y: 285 },
  { x: 670, y: 245 },
  { x: 975, y: 285 },
  { x: 995, y: 595 },
];

// Where the dealer button sits when that seat has the button.
export const DEALER_POS: Point[] = [
  { x: 545, y: 700 },
  { x: 258, y: 528 },
  { x: 300, y: 338 },
  { x: 808, y: 236 },
  { x: 1040, y: 338 },
  { x: 1082, y: 528 },
];

export const POT_LABEL_POS: Point = { x: 670, y: 300 };
export const SIDE_POTS_POS: Point = { x: 670, y: 336 };
export const BOARD_POS: Point = { x: 670, y: 434 };
export const COLLECTED_POT_POS: Point = { x: 670, y: 552 };
export const RESULT_POS: Point = { x: 670, y: 612 };

export function at(p: Point): React.CSSProperties {
  return { left: p.x, top: p.y };
}
