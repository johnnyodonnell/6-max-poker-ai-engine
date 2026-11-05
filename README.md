# 6-Max Poker — AI Engine

A single-table, 6-handed **No-Limit Texas Hold'em** game in the browser
(React 19 + TypeScript 7 + Vite 8) with a swappable AI engine. You take the
bottom seat; the other five are bots.

The v1 AI engine is **random**: on its turn a bot picks a uniform-random legal
action (and a uniform-random legal size for a bet/raise). The engine module
exposes a single `bestMove(state)` function, so a smarter engine can replace it
later without touching the UI — the same pattern as the sibling
`chess-ai-engine`, `fox-lite-ai-engine`, and `8-max-poker-ai-engine`.

The rules engine is ported from `8-max-poker-ai-engine`. The table UI is
modelled on the CoinPoker 6-max client: a dark room, an oval green felt, name
and stack plates with action labels, card backs for the bots, chips in front of
each bettor, a pot label above the board, and action buttons at the bottom
right. Click **Deal** to start each hand; bots and street run-outs resolve
instantly.

## Running locally

```sh
npm install
npm run dev        # or: ./run-local.sh
```

Then open the printed URL. `npm run build` type-checks and produces a production
build in `dist/`. Desktop browsers only — the table is a fixed-size stage that
scales to fit the window.

## Rules / setup

- **Table**: 6 seats (You, Ada, Boyd, Cleo, Dex, Esme), blinds **1 / 2**,
  everyone starts with **200** (100 bb).
- **Play**: standard NLHE — pre-flop, flop, turn, river, showdown. Fold, check,
  call, bet, raise, all-in; min-raise and side-pot rules enforced.
- **Button** rotates each hand; blinds are posted by the next live seats (with
  the heads-up small-blind-on-the-button rule once the table shrinks).
- **Freeze-out**: a bot at 0 chips sits out. The game ends when **you** bust or
  when one player holds every chip — hit **New Game** to re-seat everyone.

The bots play *randomly*, so expect wild all-ins and short games.

## Project layout

```
src/
  main.tsx              entry — mounts <App>
  App.tsx               wires the useGameState hook to <GameTable>
  App.css               global reset + fonts
  types.ts              UI-facing GameState/Player/Card/BetAction contract
  hooks/
    useGameState.ts     holds engine state; runs bots + streets to the next
                        human decision (instant)
  engine/
    cards.ts            52-card deck, shuffle, card formatting
    evaluator.ts        best-5-of-7 hand ranking + comparison
    game.ts             pure NLHE rules — no React, no side effects
    random.ts           v1 AI — bestMove(state) returns a random legal action
    adapt.ts            maps engine state <-> the UI's GameState / BetAction
  components/
    layout.ts           stage size and every seat/bet/dealer/pot position
    GameTable           scaled stage; composes everything below
    PokerTable          rail + felt
    PlayerSeat          plate, hole cards, action label
    Card, CommunityCards, PotDisplay, ChipStack, DealerButton,
    HandResult, ActionPanel
```

`GameTable` accepts a `faceUp` prop (default `false`) that shows every bot's
hole cards; nothing sets it yet.

## The engine contract

`hooks/useGameState.ts` imports `bestMove` from exactly one engine file. To plug
in a different engine, change that one line:

```ts
import { bestMove } from "../engine/random";   // current
// import { bestMove } from "../engine/<other>";
```

```ts
bestMove(state)   // state: EngineState (see engine/game.ts)
                  // -> an Action { type, amount? } for state.toAct
```

Engine actions are `{ type: 'fold' | 'check' | 'call' | 'bet' | 'raise' | 'allin' }`,
where `amount` (for bet/raise) is the *raise-to* total. `legalActions(state)`
returns exactly what's permitted for the player to act, including the legal
bet/raise band — a bot need only choose among those.
