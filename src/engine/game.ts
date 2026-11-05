// Pure No-Limit Texas Hold'em rules for a 6-max table. No React, no DOM.
//
// The whole game is a plain immutable-ish state object; every function here
// takes a state and returns a new one (we shallow-clone what we touch). The UI
// (via the useGameState hook + adapter) owns orchestration — whose turn it is,
// running out streets — and calls into these pure transitions:
//
//   createTable()           -> seated table, no hand dealt (phase 'waiting')
//   createGame()            -> table with the first hand already dealt
//   startHand(state)        -> deal the next hand
//   legalActions(state)     -> what the player to act may do
//   applyAction(state, act) -> apply one player's action
//   advanceStreet(state)    -> deal the next street / run out / go to showdown
//
// Chip model: blinds 1/2, everyone starts at 200 (100bb). A player at 0 chips
// is 'out' and sits out future hands. When one player has all the chips the
// match is over — and so is it the moment the human busts.

import { createDeck, shuffle } from "./cards";
import type { Card } from "./cards";
import { evaluate7, compareScores, categoryName } from "./evaluator";
import type { Score } from "./evaluator";

export const SEATS = 6;
export const SMALL_BLIND = 1;
export const BIG_BLIND = 2;
export const STARTING_STACK = 200;

export const BOT_NAMES = [
  "You",
  "Ada",
  "Boyd",
  "Cleo",
  "Dex",
  "Esme",
];

// --- types ------------------------------------------------------------------

// Player.status:
//   'active' — in the hand, has chips, can still act
//   'allin'  — in the hand, no chips left to act with
//   'folded' — out of this hand
//   'out'    — busted, sitting out until a new game
export type EngineStatus = "active" | "allin" | "folded" | "out";
export type Street = "preflop" | "flop" | "turn" | "river";
// phase drives the orchestration layer:
//   'waiting'         — table seated, no hand dealt yet (Deal button)
//   'betting'         — waiting on state.toAct to act
//   'street-complete' — betting closed; caller runs advanceStreet
//   'hand-over'       — hand decided (fold-out or showdown); await next hand
//   'match-over'      — one player holds every chip
export type EnginePhase =
  | "waiting"
  | "betting"
  | "street-complete"
  | "hand-over"
  | "match-over";

export interface EnginePlayer {
  id: number;
  name: string;
  isHuman: boolean;
  stack: number;
  holeCards: Card[];
  status: EngineStatus;
  bet: number; // chips committed on the current street
  commit: number; // chips committed across the whole hand (for side pots)
  hasActed: boolean;
  // The player's most recent action on the current street (drives the seat
  // label: CALL / CHECK / FOLD / BET / RAISE / ALL-IN). Cleared each street.
  lastAction: ActionType | null;
}

export type ActionType = "fold" | "check" | "call" | "bet" | "raise" | "allin";
export interface Action {
  type: ActionType;
  amount?: number;
}

export interface LegalActions {
  actions: ActionType[];
  toCall: number;
  callTo: number;
  minRaiseTo: number;
  maxRaiseTo: number;
  canCheck: boolean;
}

export interface PotLayer {
  amount: number;
  eligible: number[];
}
export interface PotResult {
  amount: number;
  winners: number[];
  score: Score;
}
export interface WinnerRecord {
  seat: number;
  amountWon: number;
  handDescription: string;
}
export interface Results {
  pots: PotResult[];
  winners: WinnerRecord[];
  // At showdown the still-in players' cards are turned face up; a hand won by
  // everyone else folding stays concealed.
  reveal: boolean;
}

export interface EngineState {
  players: EnginePlayer[];
  button: number;
  sbSeat: number;
  bbSeat: number;
  handNumber: number;
  street: Street;
  community: Card[];
  deck: Card[];
  pot: number; // chips already gathered from earlier streets
  currentBet: number; // highest per-street bet to match
  minRaise: number; // smallest legal raise increment
  toAct: number;
  phase: EnginePhase;
  results: Results | null;
  log: string[];
}

// --- table / hand setup -----------------------------------------------------

function createPlayers(): EnginePlayer[] {
  return BOT_NAMES.map((name, i) => ({
    id: i,
    name,
    isHuman: i === 0,
    stack: STARTING_STACK,
    holeCards: [],
    status: "active",
    bet: 0,
    commit: 0,
    hasActed: false,
    lastAction: null,
  }));
}

// A seated table with no hand dealt — the Deal button starts hand 1.
export function createTable(): EngineState {
  return {
    players: createPlayers(),
    button: -1, // startHand advances it, so the first hand's button is seat 0
    sbSeat: -1,
    bbSeat: -1,
    handNumber: 0,
    street: "preflop",
    community: [],
    deck: [],
    pot: 0,
    currentBet: 0,
    minRaise: BIG_BLIND,
    toAct: -1,
    phase: "waiting",
    results: null,
    log: [],
  };
}

// A table with the first hand already dealt (used by tests/headless play).
export function createGame(): EngineState {
  return startHand(createTable());
}

// --- seat-order helpers -----------------------------------------------------

function contenders(players: EnginePlayer[]): EnginePlayer[] {
  return players.filter((p) => p.stack > 0);
}

// Next seat index (clockwise) after `from` whose player passes `pred`.
// Returns -1 if none.
function nextSeat(
  players: EnginePlayer[],
  from: number,
  pred: (p: EnginePlayer) => boolean,
): number {
  for (let step = 1; step <= SEATS; step++) {
    const i = (from + step) % SEATS;
    if (pred(players[i])) return i;
  }
  return -1;
}

export function startHand(prev: EngineState): EngineState {
  const players: EnginePlayer[] = prev.players.map((p) => ({
    ...p,
    holeCards: [],
    bet: 0,
    commit: 0,
    hasActed: false,
    lastAction: null,
    // Anyone with chips comes back in; the busted sit out.
    status: p.stack > 0 ? "active" : "out",
  }));

  const live = contenders(players);
  if (live.length < 2 || humanBusted(players)) {
    return { ...prev, players, phase: "match-over" };
  }

  const button = nextSeat(players, prev.button, (p) => p.stack > 0);
  const headsUp = live.length === 2; // heads-up: button posts the small blind

  const sbSeat = headsUp
    ? button
    : nextSeat(players, button, (p) => p.stack > 0);
  const bbSeat = nextSeat(players, sbSeat, (p) => p.stack > 0);

  postBlind(players[sbSeat], SMALL_BLIND);
  postBlind(players[bbSeat], BIG_BLIND);

  // Deal two hole cards to every contender. Deal order is cosmetic in a
  // shuffled deck, so we simply go seat by seat.
  const deck = shuffle(createDeck());
  for (const p of players) {
    if (p.status !== "out") {
      p.holeCards = [deck.pop()!, deck.pop()!];
    }
  }

  const firstToAct = nextSeat(players, bbSeat, (p) => p.status === "active");

  const log = [`--- Hand #${prev.handNumber + 1} ---`];
  log.push(`${players[sbSeat].name} posts SB ${SMALL_BLIND}`);
  log.push(`${players[bbSeat].name} posts BB ${BIG_BLIND}`);

  return {
    ...prev,
    players,
    button,
    sbSeat,
    bbSeat,
    handNumber: prev.handNumber + 1,
    street: "preflop",
    community: [],
    deck,
    pot: 0,
    currentBet: BIG_BLIND,
    minRaise: BIG_BLIND,
    toAct: firstToAct,
    phase: "betting",
    results: null,
    log,
  };
}

// Post a blind, capping at the player's stack (short stacks post all-in).
function postBlind(player: EnginePlayer, amount: number): void {
  const posted = Math.min(amount, player.stack);
  player.stack -= posted;
  player.bet = posted;
  player.commit = posted;
  if (player.stack === 0) player.status = "allin";
}

// --- legal actions ----------------------------------------------------------

// Describes what the player to act may legally do. Bet/raise amounts are
// expressed as a *raise-to* total (the player's `bet` after the action), which
// keeps the caller's arithmetic simple.
export function legalActions(state: EngineState): LegalActions | null {
  const p = state.players[state.toAct];
  if (!p || p.status !== "active") return null;

  const toCall = state.currentBet - p.bet;
  const maxTo = p.bet + p.stack; // shove: commit the whole stack

  const actions: ActionType[] = [];

  // Fold is offered only when there's something to call, matching how a real
  // table plays (folding when checking is free is pointless).
  if (toCall > 0) actions.push("fold");

  if (toCall <= 0) actions.push("check");
  else actions.push("call");

  // Opening bet vs. raising an existing bet.
  const opening = state.currentBet === 0;
  // Smallest legal raise-to; capped by an all-in that can't reach it.
  const minTo = Math.min(
    opening ? BIG_BLIND : state.currentBet + state.minRaise,
    maxTo,
  );
  // A raise/bet is only possible if the player can put more in than a call.
  if (maxTo > state.currentBet) {
    actions.push(opening ? "bet" : "raise");
  }

  return {
    actions,
    toCall: Math.min(toCall, p.stack),
    callTo: Math.min(state.currentBet, maxTo),
    minRaiseTo: minTo,
    maxRaiseTo: maxTo,
    canCheck: toCall <= 0,
  };
}

// --- applying an action -----------------------------------------------------

// For bet/raise, `action.amount` is the raise-to total. 'allin' is shorthand
// for a bet/raise/call to the player's maximum.
export function applyAction(state: EngineState, action: Action): EngineState {
  const players = state.players.map((p) => ({ ...p }));
  const i = state.toAct;
  const p = players[i];
  const legal = legalActions(state);
  if (!legal) return state;

  let currentBet = state.currentBet;
  let minRaise = state.minRaise;
  const log = state.log.slice();

  const maxTo = p.bet + p.stack;
  const commit = (to: number): void => {
    const delta = to - p.bet;
    p.stack -= delta;
    p.commit += delta;
    p.bet = to;
    if (p.stack === 0) p.status = "allin";
  };

  let type: ActionType = action.type;
  let amount = action.amount;
  // Normalize an 'allin' into the concrete action it represents.
  if (type === "allin") {
    amount = maxTo;
    if (maxTo <= currentBet) type = currentBet - p.bet > 0 ? "call" : "check";
    else type = currentBet === 0 ? "bet" : "raise";
  }

  switch (type) {
    case "fold":
      p.status = "folded";
      p.hasActed = true;
      log.push(`${p.name} folds`);
      break;

    case "check":
      p.hasActed = true;
      log.push(`${p.name} checks`);
      break;

    case "call": {
      const to = Math.min(currentBet, maxTo);
      const put = to - p.bet;
      commit(to);
      p.hasActed = true;
      log.push(`${p.name} calls ${put}${p.status === "allin" ? " (all in)" : ""}`);
      break;
    }

    case "bet":
    case "raise": {
      let to = amount ?? legal.minRaiseTo;
      // Clamp to the legal band. An all-in that can't reach a full raise is
      // still allowed (it just may not reopen the betting — see below).
      to = Math.max(Math.min(to, maxTo), Math.min(legal.minRaiseTo, maxTo));
      const raiseSize = to - currentBet;
      commit(to);
      p.hasActed = true;
      // A raise of at least a full min-raise reopens the action: everyone who
      // already acted must respond again. A short all-in does not.
      if (raiseSize >= minRaise) {
        minRaise = raiseSize;
        for (const other of players) {
          if (other.id !== p.id && other.status === "active")
            other.hasActed = false;
        }
      }
      currentBet = Math.max(currentBet, to);
      const verb = type === "bet" ? `bets ${to}` : `raises to ${to}`;
      log.push(`${p.name} ${verb}${p.status === "allin" ? " (all in)" : ""}`);
      break;
    }

    default:
      return state;
  }

  p.lastAction = p.status === "allin" ? "allin" : type;

  const next: EngineState = { ...state, players, currentBet, minRaise, log };

  // Is the betting round finished?
  if (bettingClosed(next)) {
    return { ...next, phase: "street-complete" };
  }
  next.toAct = nextActor(next, i);
  next.phase = "betting";
  return next;
}

// The betting round closes when no 'active' player still owes an action — i.e.
// everyone active has acted and matched the current bet. Also closes the
// instant only one non-folded player remains.
function bettingClosed(state: EngineState): boolean {
  const inHand = state.players.filter(
    (p) => p.status === "active" || p.status === "allin",
  );
  if (inHand.length <= 1) return true;

  for (const p of state.players) {
    if (p.status !== "active") continue;
    if (!p.hasActed || p.bet < state.currentBet) return false;
  }
  return true;
}

// Next seat after `from` that still owes an action.
function nextActor(state: EngineState, from: number): number {
  return nextSeat(
    state.players,
    from,
    (p) => p.status === "active" && (!p.hasActed || p.bet < state.currentBet),
  );
}

// --- moving between streets -------------------------------------------------

const NEXT_STREET: Record<Street, Street | undefined> = {
  preflop: "flop",
  flop: "turn",
  turn: "river",
  river: undefined,
};
const DEAL_COUNT: Record<Street, number> = {
  preflop: 0,
  flop: 3,
  turn: 1,
  river: 1,
};

// Called once the current street's betting is complete (phase
// 'street-complete'). Gathers bets into the pot, then either ends the hand or
// deals the next street. When two-plus players are all-in with nothing left to
// decide, it deals ONE street at a time and stays in 'street-complete' so the
// run-out can be revealed street by street.
export function advanceStreet(state: EngineState): EngineState {
  const players = state.players.map((p) => ({ ...p }));
  // Sweep this street's bets into the central pot.
  let pot = state.pot;
  for (const p of players) {
    pot += p.bet;
    p.bet = 0;
    p.hasActed = false;
  }

  const notFolded = players.filter(
    (p) => p.status !== "folded" && p.status !== "out",
  );

  // Everyone folded to one player: they win without a showdown.
  if (notFolded.length === 1) {
    return settle({ ...state, players, pot }, false);
  }

  // River already dealt and its betting done -> showdown.
  if (state.street === "river") {
    return settle({ ...state, players, pot }, true);
  }

  // Deal the next street.
  const deck = state.deck.slice();
  const street = NEXT_STREET[state.street]!;
  // New street, fresh seat labels (a fold stays visible via the folded status).
  for (const p of players) p.lastAction = null;
  const dealt: Card[] = [];
  for (let k = 0; k < DEAL_COUNT[street]; k++) dealt.push(deck.pop()!);
  const community = [...state.community, ...dealt];

  const log = state.log.slice();
  log.push(
    `${street[0].toUpperCase() + street.slice(1)}: ${dealt
      .map((c) => c.id)
      .join(" ")}`,
  );

  const canAct = players.filter((p) => p.status === "active");
  const base: EngineState = {
    ...state,
    players,
    pot,
    deck,
    community,
    street,
    currentBet: 0,
    minRaise: BIG_BLIND,
    log,
  };

  // Fewer than two players can still act -> no betting; keep running out.
  if (canAct.length < 2) {
    return { ...base, toAct: -1, phase: "street-complete" };
  }

  return {
    ...base,
    toAct: nextSeat(players, state.button, (p) => p.status === "active"),
    phase: "betting",
  };
}

// --- showdown / settling the pot -------------------------------------------

// Build main + side pots from each player's total commitment. Returns
// [{ amount, eligible: [ids] }], smallest layer first.
export function buildPots(players: EnginePlayer[]): PotLayer[] {
  const contrib = players
    .filter((p) => p.commit > 0)
    .map((p) => ({ id: p.id, amt: p.commit, folded: p.status === "folded" }));

  const pots: (PotLayer & { key: string })[] = [];
  while (true) {
    const positive = contrib.filter((c) => c.amt > 0);
    if (positive.length === 0) break;
    const layer = Math.min(...positive.map((c) => c.amt));
    let amount = 0;
    for (const c of positive) {
      amount += layer;
      c.amt -= layer;
    }
    const eligible = positive.filter((c) => !c.folded).map((c) => c.id);
    // Merge into the previous pot if the eligible set is identical.
    const key = eligible.join(",");
    const last = pots[pots.length - 1];
    if (last && last.key === key) last.amount += amount;
    else pots.push({ amount, eligible, key });
  }
  return pots.map(({ amount, eligible }) => ({ amount, eligible }));
}

// Decide the hand, distribute chips, and set results/phase.
function settle(state: EngineState, reveal: boolean): EngineState {
  const players = state.players.map((p) => ({ ...p }));
  const pots = buildPots(players);

  // Best hand score per still-in player.
  const scores = new Map<number, Score>();
  for (const p of players) {
    if (p.status === "folded" || p.status === "out") continue;
    scores.set(p.id, evaluate7([...p.holeCards, ...state.community]));
  }

  const potResults: PotResult[] = [];
  const awarded = new Map<number, number>(); // seat -> chips won
  for (const pot of pots) {
    const eligible = pot.eligible.filter((id) => scores.has(id));
    if (eligible.length === 0) continue;
    let best: Score | null = null;
    let winners: number[] = [];
    for (const id of eligible) {
      const s = scores.get(id)!;
      const cmp = best ? compareScores(s, best) : 1;
      if (cmp > 0) {
        best = s;
        winners = [id];
      } else if (cmp === 0) {
        winners.push(id);
      }
    }
    // Split, awarding odd chips to the earliest seats left of the button.
    const share = Math.floor(pot.amount / winners.length);
    let remainder = pot.amount - share * winners.length;
    for (const id of winners) {
      let amt = share;
      if (remainder > 0) {
        amt += 1;
        remainder -= 1;
      }
      players[id].stack += amt;
      awarded.set(id, (awarded.get(id) ?? 0) + amt);
    }
    potResults.push({ amount: pot.amount, winners, score: best! });
  }

  const winners: WinnerRecord[] = [...awarded.entries()].map(
    ([seat, amountWon]) => ({
      seat,
      amountWon,
      handDescription: reveal
        ? categoryName(scores.get(seat)!)
        : "opponents folded",
    }),
  );

  const log = state.log.slice();
  for (const w of winners) {
    log.push(`${players[w.seat].name} wins ${w.amountWon}`);
  }

  // Players who busted out are re-seated (as 'out') by the next startHand;
  // here we only need to know whether the match is decided: one player holds
  // every chip, or the human has busted.
  const matchOver =
    players.filter((p) => p.stack > 0).length < 2 || humanBusted(players);
  return {
    ...state,
    players,
    pot: 0,
    toAct: -1,
    phase: matchOver ? "match-over" : "hand-over",
    results: { pots: potResults, winners, reveal },
    log,
  };
}

function humanBusted(players: EnginePlayer[]): boolean {
  return players.some((p) => p.isHuman && p.stack === 0);
}

// The overall match winner (valid once phase === 'match-over').
export function matchWinner(state: EngineState): EnginePlayer | null {
  return state.players.find((p) => p.stack > 0) ?? null;
}
