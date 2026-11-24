//! Dense indices for card sets and (hole cards, board) pairs.
//!
//! A k-card set is ranked in colexicographic order (the combination number
//! system: rank = Σ C(c_i, i+1) over its cards sorted ascending), so the
//! C(52, k) sets map onto 0..C(52, k) with no gaps. Tables are laid out
//! board-major: `index(hole, board) = rank(board) * 1326 + rank(hole)`.
//! Entries whose hole cards overlap the board are simply unused (about 20%
//! more space than a gap-free scheme, in exchange for trivial indexing).

use poker_core::cards::Card;

pub const N_HOLES: usize = 1326;

/// C(n, k) for n <= 52, k <= 5.
const BINOM: [[u32; 6]; 53] = {
    let mut t = [[0u32; 6]; 53];
    let mut n = 0;
    while n <= 52 {
        t[n][0] = 1;
        let mut k = 1;
        while k <= 5 {
            t[n][k] = if n == 0 { 0 } else { t[n - 1][k - 1] + t[n - 1][k] };
            k += 1;
        }
        n += 1;
    }
    t
};

#[inline]
pub fn binom(n: usize, k: usize) -> u32 {
    BINOM[n][k]
}

/// Number of k-card sets from 52 cards.
pub fn n_sets(k: usize) -> usize {
    binom(52, k) as usize
}

/// Colex rank of a card set (any order; up to 5 cards).
#[inline]
pub fn rank_set(cards: &[Card]) -> u32 {
    let mut s = [0u8; 5];
    let n = cards.len();
    s[..n].copy_from_slice(cards);
    s[..n].sort_unstable();
    let mut r = 0;
    for (i, &c) in s[..n].iter().enumerate() {
        r += binom(c as usize, i + 1);
    }
    r
}

/// Rank of a two-card hand, 0..1326.
#[inline]
pub fn hole_index(a: Card, b: Card) -> u32 {
    let (lo, hi) = if a < b { (a, b) } else { (b, a) };
    binom(hi as usize, 2) + lo as u32
}

/// Table index of (hole, board).
#[inline]
pub fn index(hole: [Card; 2], board: &[Card]) -> usize {
    rank_set(board) as usize * N_HOLES + hole_index(hole[0], hole[1]) as usize
}

/// Every k-card set, in rank order (position i has rank i).
pub fn all_sets<const K: usize>() -> Vec<[Card; K]> {
    let mut out = Vec::with_capacity(n_sets(K));
    let mut c = [0u8; K];
    fn rec<const K: usize>(i: usize, max: u8, c: &mut [Card; K], out: &mut Vec<[Card; K]>) {
        // Fill position i (counting down from K-1) with values below `max`,
        // largest-card-major so ranks come out in increasing order.
        if i == usize::MAX {
            out.push(*c);
            return;
        }
        for v in (i as u8)..max {
            c[i] = v;
            rec(i.wrapping_sub(1), v, c, out);
        }
    }
    rec(K - 1, 52, &mut c, &mut out);
    out
}

/// The two cards of hand `h` (inverse of `hole_index`).
pub fn hole_cards(h: u32) -> [Card; 2] {
    let mut hi = 1usize;
    while binom(hi + 1, 2) <= h {
        hi += 1;
    }
    [(h - binom(hi, 2)) as Card, hi as Card]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_are_dense_and_ordered() {
        for k in 1..=5 {
            let sets: Vec<Vec<Card>> = match k {
                1 => all_sets::<1>().into_iter().map(|s| s.to_vec()).collect(),
                2 => all_sets::<2>().into_iter().map(|s| s.to_vec()).collect(),
                3 => all_sets::<3>().into_iter().map(|s| s.to_vec()).collect(),
                4 => all_sets::<4>().into_iter().map(|s| s.to_vec()).collect(),
                _ => all_sets::<5>().into_iter().map(|s| s.to_vec()).collect(),
            };
            assert_eq!(sets.len(), n_sets(k));
            for (i, s) in sets.iter().enumerate().step_by(if k == 5 { 997 } else { 1 }) {
                assert_eq!(rank_set(s) as usize, i, "{s:?}");
                let mut r = s.clone();
                r.reverse();
                assert_eq!(rank_set(&r) as usize, i, "order-independent");
            }
        }
    }

    #[test]
    fn hole_index_round_trips() {
        let mut seen = vec![false; N_HOLES];
        for b in 1..52u8 {
            for a in 0..b {
                let h = hole_index(a, b);
                assert_eq!(h, hole_index(b, a));
                assert!(!seen[h as usize]);
                seen[h as usize] = true;
                assert_eq!(hole_cards(h), [a, b]);
            }
        }
        assert!(seen.iter().all(|&x| x));
    }
}
