//! Pure 6-max No-Limit Hold'em core for self-play training: no torch, no I/O.
//!
//! The rules are a faithful port of `src/engine/game.ts` (the browser game is
//! the source of truth); parity is checked against traces dumped from the TS
//! engine (`training/scripts/dump_fixtures.ts`).

pub mod actions;
pub mod allin_ev;
pub mod cards;
pub mod encode;
pub mod equity;
pub mod hand_eval;
pub mod rules;
pub mod table;
