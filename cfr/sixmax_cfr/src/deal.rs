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

/// One deal as the solver sees it: each seat's card group on each street and
/// its showdown strength (higher wins; compared among live seats only).
#[derive(Clone, Copy, Debug, Default)]
pub struct Deal {
    pub bucket: [[u32; 4]; poker_core::rules::SEATS],
    pub rank: [u32; poker_core::rules::SEATS],
}

/// Samples deals for the solver.
pub trait Dealer: Sync {
    fn deal(&self, rng: &mut SmallRng) -> Deal;
}

/// Real cards for `players` seats (6, or 2 for heads-up: seats 0 and 1) and a
/// board, mapped to card groups.
pub struct PokerDealer<'a> {
    pub abs: &'a crate::abs::Abstraction,
    pub players: usize,
}

impl Dealer for PokerDealer<'_> {
    #[inline]
    fn deal(&self, rng: &mut SmallRng) -> Deal {
        let mut c = [0u8; 17];
        let n = 2 * self.players + 5;
        deal_cards(rng, &mut c[..n]);
        let board = &c[2 * self.players..n];
        let mut d = Deal::default();
        let mut seven = [0u8; 7];
        seven[2..].copy_from_slice(board);
        for p in 0..self.players {
            let hole = [c[2 * p], c[2 * p + 1]];
            for s in 0..4 {
                d.bucket[p][s] = self.abs.bucket(s, hole, board);
            }
            seven[0] = hole[0];
            seven[1] = hole[1];
            d.rank[p] = eval7(&seven);
        }
        d
    }
}
