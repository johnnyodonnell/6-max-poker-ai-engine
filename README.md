# 6-Max Poker — AI Engine

A single-table, 6-handed **No-Limit Texas Hold'em** game in the browser
(React 19 + TypeScript 7 + Vite 8) with a swappable AI engine. You take the
bottom seat; the other five are bots.

The bots are a **neural network** trained by search-free self-play
(PPO with a training-only critic). On its turn a bot runs one forward pass of
`public/models/current.onnx` in the browser (onnxruntime-web, WebGPU with a
WASM fallback). It then *samples* its move from the policy over ten discrete
actions: fold, check/call, min-raise, six bet sizes (33–250% of the pot) and
all-in. Training lives in `training/` (all Rust; see `training/README.md`).

The engine module exposes a single `bestMove(state)` function, so the engine can
be swapped without touching the UI. This is the same pattern as the sibling
`chess-ai-engine`, `fox-lite-ai-engine` and `8-max-poker-ai-engine`. The
original random engine is still in `engine/random.ts`.

The rules engine is ported from `8-max-poker-ai-engine`. The table UI is
modelled on the CoinPoker 6-max client: a dark room, an oval green felt, name
and stack plates with action labels, card backs for the bots, chips in front of
each bettor, a pot label above the board, and action buttons at the bottom
right. Click **Deal** to start each hand; bots and street run-outs resolve
without artificial delay.

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
- **Short all-ins**: an all-in raise smaller than a full raise doesn't reopen
  the betting to players who already acted (they may only call or fold), unless
  several short all-ins add up to a full raise.
- **Split pots**: odd chips go to the first winner left of the button.

## Project layout

```
src/
  main.tsx              entry — mounts <App>
  App.tsx               wires the useGameState hook to <GameTable>
  App.css               global reset + fonts
  types.ts              UI-facing GameState/Player/Card/BetAction contract
  hooks/
    useGameState.ts     holds engine state; runs bots (async) + streets to
                        the next human decision
  engine/
    cards.ts            52-card deck, shuffle, card formatting
    evaluator.ts        best-5-of-7 hand ranking + comparison
    game.ts             pure NLHE rules — no React, no side effects
    neural.ts           the AI — async bestMove(state): encode, run the net,
                        sample a move
    net.ts              onnxruntime-web session for public/models/current.onnx
    encode.ts           the network's input encoding (Rust mirror in training/)
    actions.ts          the discrete action space and its raise-to amounts
    random.ts           the original random engine
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
import { bestMove } from "../engine/neural";   // current
// import { bestMove } from "../engine/<other>";
```

```ts
await bestMove(state)   // state: EngineState (see engine/game.ts)
                        // -> an Action { type, amount? } for state.toAct
```

The hook `await`s each move, so a synchronous engine (like `random.ts`) works
too.

Engine actions are `{ type: 'fold' | 'check' | 'call' | 'bet' | 'raise' | 'allin' }`,
where `amount` (for bet/raise) is the *raise-to* total. `legalActions(state)`
returns exactly what's permitted for the player to act, including the legal
bet/raise band — a bot need only choose among those.
