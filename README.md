# 6-Max Poker — AI Engine

A single-table, 6-handed **No-Limit Texas Hold'em** game in the browser
(React 19 + TypeScript 7 + Vite 8) with a swappable AI engine. You take the
bottom seat; the other five are bots.

The bots play a **CFR strategy**: a precomputed counterfactual-regret-
minimization blueprint for 6-max no-limit, trained and served by the
`asus-nvidia` box (see [cfr/README.md](cfr/README.md)). The browser asks the
engine for each bot move (`engine/remote.ts` → `POST /api/act`); there is no
fallback, so if the engine can't be reached the table says so and offers a
retry. The original random bot (`engine/random.ts`) stays in the code but
isn't served.

The engine module exposes a single `bestMove(state)` function, so the engine can
be swapped without touching the UI. This is the same pattern as the sibling
`chess-ai-engine`, `fox-lite-ai-engine` and `8-max-poker-ai-engine`.

The rules engine is ported from `8-max-poker-ai-engine`. The table UI is
modelled on the CoinPoker 6-max client: a dark room, an oval green felt, name
and stack plates with action labels, card backs for the bots, chips in front of
each bettor, a pot label above the board, and action buttons at the bottom
right. Click **Deal** to start each hand; bots and street run-outs resolve
without artificial delay.

- **Bet sizing** with chips: a row of $1/$5/$25/$100 chips that add to the bet
  over a row that take chips off. The bet starts at the minimum raise; the first
  chip that reaches a limit stops the bet there (all-in, or back to the
  minimum), and larger chips are dimmed. While the bots act, the controls stay
  in place, greyed out.
- **Stacks survive a refresh**: the match (stacks, button, hand count) is saved
  in the browser between hands; a hand in progress is dealt again.
- **Debug view** (top-left toggle, remembered): every bot's hole cards face up,
  folded ones dimmed in place, and a sidebar with each bot decision this hand —
  the strategy it used (table size, stack depth), its hand group, the options
  with the chance of each, and the pick. The table
  shrinks to make room for it.

## Playing

The box serves the built app along with the engine:
**http://192.168.0.185:8791**.

## Running locally

```sh
npm install
npm run dev        # or: ./run-local.sh
```

Then open the printed URL. The dev server forwards `/api` to the engine on the
box (`SIXMAX_ENGINE=http://host:port` to point elsewhere). `npm run build`
type-checks and produces a production build in `dist/`; `cfr/deploy_web.sh`
builds and copies it to the box. Desktop browsers only — the table is a fixed-size stage that
scales to fit the window.

## Rules / setup

- **Table**: 6 seats (You, Ada, Boyd, Cleo, Dex, Esme), blinds **1 / 2**,
  everyone starts with **200** (100 bb).
- **Play**: standard NLHE — pre-flop, flop, turn, river, showdown. Fold, check,
  call, bet, raise, all-in; min-raise and side-pot rules enforced.
- **Showdown**: all-in hands are always shown. The others show in turn from the
  river's last bettor or raiser (or the first seat left of the button if the
  river was checked through); a hand beaten by one already shown is mucked
  (MUCK tab, cards face down).
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
    remote.ts           the AI — bestMove(state): the CFR engine on the box
    random.ts           a uniform-random legal move (the original bot, unused)
    saved.ts            the match saved in localStorage between hands
    adapt.ts            maps engine state <-> the UI's GameState / BetAction
  components/
    layout.ts           stage size and every seat/bet/dealer/pot position
    GameTable           scaled stage; composes everything below
    PokerTable          rail + felt
    PlayerSeat          plate, hole cards, action label
    ActionPanel         chip bet sizing + action buttons
    DebugPanel          the debug toggle and the bots' decisions
    Card, CommunityCards, PotDisplay, ChipStack, DealerButton,
    HandResult
cfr/                    the engine: Rust, trained and served on the box
```

## The engine contract

`hooks/useGameState.ts` imports `bestMove` from exactly one engine file. To plug
in a different engine, change that one line:

```ts
import { bestMove } from "../engine/remote";   // current: the CFR engine
// import { bestMove } from "../engine/random";   // the random placeholder
```

```ts
await bestMove(state)   // state: EngineState (see engine/game.ts)
                        // -> an Action { type, amount? } for state.toAct
```

The hook `await`s each move, so both synchronous engines (like `random.ts`) and
async ones (like the remote engine) work; a rejected move stops the table with
the error and a Retry button.

Engine actions are `{ type: 'fold' | 'check' | 'call' | 'bet' | 'raise' | 'allin' }`,
where `amount` (for bet/raise) is the *raise-to* total. `legalActions(state)`
returns exactly what's permitted for the player to act, including the legal
bet/raise band — a bot need only choose among those.
