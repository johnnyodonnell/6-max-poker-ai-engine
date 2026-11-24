//! Dealing helpers shared by the card groups, training and evaluation.

use poker_core::cards::Card;
use poker_core::hand_eval::eval7;
use rand::rngs::SmallRng;
use rand::Rng;

/// Deal `out.len()` distinct cards from a fresh deck (partial Fisher–Yates).
#[inline]
pub fn deal_cards(rng: &mut SmallRng, out: &mut [Card]) {
    let mut deck: [Card; 52] = std::array::from_fn(|i| i as Card);
    for i in 0..out.len() {
        let j = rng.gen_range(i..52);
        deck.swap(i, j);
        out[i] = deck[i];
    }
}

/// Showdown result (+1 the first hand wins, -1 the second, 0 tie) for two
/// hole-card pairs on a 5-card board.
#[inline]
pub fn showdown(h0: [Card; 2], h1: [Card; 2], board: &[Card]) -> i8 {
    let mut c = [0u8; 7];
    c[2..].copy_from_slice(board);
    c[0] = h0[0];
    c[1] = h0[1];
    let v0 = eval7(&c);
    c[0] = h1[0];
    c[1] = h1[1];
    let v1 = eval7(&c);
    (v0.cmp(&v1) as i8).signum()
}
