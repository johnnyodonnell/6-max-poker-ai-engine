//! Fast 7-card hand evaluation.
//!
//! `eval7` returns a `u32` whose numeric order is poker strength: the hand
//! category (0 = high card … 8 = straight flush, as in `evaluator.ts`) in bits
//! 20..24, then up to five tie-break ranks (2..=14) in 4-bit nibbles, high to
//! low. This is exactly the TS score array `[category, ...kickers]` packed into
//! an integer, so comparisons agree with `compareScores` (missing kickers count
//! as 0 in both).

use crate::cards::{rank, suit, Card};

pub const HIGH_CARD: u32 = 0;
pub const PAIR: u32 = 1;
pub const TWO_PAIR: u32 = 2;
pub const TRIPS: u32 = 3;
pub const STRAIGHT: u32 = 4;
pub const FLUSH: u32 = 5;
pub const FULL_HOUSE: u32 = 6;
pub const QUADS: u32 = 7;
pub const STRAIGHT_FLUSH: u32 = 8;

#[inline]
pub fn category(strength: u32) -> u32 {
    strength >> 20
}

#[inline]
fn pack(cat: u32, kickers: &[u8]) -> u32 {
    let mut v = cat << 20;
    for (i, &k) in kickers.iter().take(5).enumerate() {
        v |= (k as u32) << (16 - 4 * i);
    }
    v
}

/// Highest straight in a rank bitmask (bit r-2 set for rank r), as its top
/// rank (5 for the wheel), or 0.
#[inline]
fn straight_high(mask: u16) -> u8 {
    // Ace also plays low: put a copy of the ace bit below the deuce.
    let m = ((mask as u32) << 1) | ((mask as u32 >> 12) & 1);
    // Bit i of m is rank i+1 (bit 0 = low ace).
    for top in (4..=13).rev() {
        let window = 0b1_1111u32 << (top - 4);
        if m & window == window {
            return (top + 1) as u8;
        }
    }
    0
}

/// Up to `n` ranks from a rank bitmask, high to low.
#[inline]
fn top_ranks(mask: u16, n: usize, out: &mut [u8; 5]) -> usize {
    let mut k = 0;
    for r in (0..13).rev() {
        if k == n {
            break;
        }
        if mask & (1 << r) != 0 {
            out[k] = r as u8 + 2;
            k += 1;
        }
    }
    k
}

/// Strength of the best 5-card hand among `cards` (5 to 7 cards).
pub fn eval7(cards: &[Card]) -> u32 {
    let mut suit_mask = [0u16; 4];
    let mut counts = [0u8; 13];
    for &c in cards {
        let r = rank(c) - 2;
        suit_mask[suit(c) as usize] |= 1 << r;
        counts[r as usize] += 1;
    }
    let mut kick = [0u8; 5];

    // Flush / straight flush (two flushes are impossible with <= 7 cards).
    for &m in &suit_mask {
        if m.count_ones() >= 5 {
            let sf = straight_high(m);
            if sf != 0 {
                return pack(STRAIGHT_FLUSH, &[sf]);
            }
            // Quads or a full house beat a flush; check those first below.
            let mut has_quads = false;
            let mut trips = 0;
            let mut pairs = 0;
            for &n in &counts {
                if n == 4 {
                    has_quads = true;
                }
                if n >= 3 {
                    trips += 1;
                } else if n == 2 {
                    pairs += 1;
                }
            }
            if !has_quads && !(trips >= 2 || (trips == 1 && pairs >= 1)) {
                let k = top_ranks(m, 5, &mut kick);
                return pack(FLUSH, &kick[..k]);
            }
        }
    }

    // Rank groups, strongest rank first within each count.
    let mut quad = 0u8;
    let mut trip_ranks = [0u8; 2];
    let mut n_trips = 0;
    let mut pair_ranks = [0u8; 3];
    let mut n_pairs = 0;
    let mut any_mask = 0u16;
    for r in (0..13).rev() {
        let n = counts[r];
        if n > 0 {
            any_mask |= 1 << r;
        }
        let rr = r as u8 + 2;
        match n {
            4 => quad = rr,
            3 => {
                trip_ranks[n_trips] = rr;
                n_trips += 1;
            }
            2 => {
                if n_pairs < 3 {
                    pair_ranks[n_pairs] = rr;
                    n_pairs += 1;
                }
            }
            _ => {}
        }
    }

    if quad != 0 {
        let rest = any_mask & !(1 << (quad - 2));
        top_ranks(rest, 1, &mut kick);
        return pack(QUADS, &[quad, kick[0]]);
    }
    if n_trips >= 2 {
        return pack(FULL_HOUSE, &[trip_ranks[0], trip_ranks[1]]);
    }
    if n_trips == 1 && n_pairs >= 1 {
        return pack(FULL_HOUSE, &[trip_ranks[0], pair_ranks[0]]);
    }
    // (flush handled above)
    let st = straight_high(any_mask);
    if st != 0 {
        return pack(STRAIGHT, &[st]);
    }
    if n_trips == 1 {
        let t = trip_ranks[0];
        let rest = any_mask & !(1 << (t - 2));
        let k = top_ranks(rest, 2, &mut kick);
        let mut v = [t, 0, 0];
        v[1..1 + k].copy_from_slice(&kick[..k]);
        return pack(TRIPS, &v[..1 + k]);
    }
    if n_pairs >= 2 {
        let (hi, lo) = (pair_ranks[0], pair_ranks[1]);
        let rest = any_mask & !(1 << (hi - 2)) & !(1 << (lo - 2));
        top_ranks(rest, 1, &mut kick);
        return pack(TWO_PAIR, &[hi, lo, kick[0]]);
    }
    if n_pairs == 1 {
        let p = pair_ranks[0];
        let rest = any_mask & !(1 << (p - 2));
        let k = top_ranks(rest, 3, &mut kick);
        let mut v = [p, 0, 0, 0];
        v[1..1 + k].copy_from_slice(&kick[..k]);
        return pack(PAIR, &v[..1 + k]);
    }
    let k = top_ranks(any_mask, 5, &mut kick);
    pack(HIGH_CARD, &kick[..k])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse;

    fn ev(ids: &str) -> u32 {
        let cards: Vec<Card> = ids.split_whitespace().map(|s| parse(s).unwrap()).collect();
        eval7(&cards)
    }

    #[test]
    fn categories() {
        assert_eq!(category(ev("Ah Kh Qh Jh 10h 2c 3d")), STRAIGHT_FLUSH);
        assert_eq!(category(ev("Ah 2h 3h 4h 5h 9c 9d")), STRAIGHT_FLUSH);
        assert_eq!(category(ev("9h 9c 9d 9s 2c 3d 4h")), QUADS);
        assert_eq!(category(ev("9h 9c 9d 2s 2c 3d 4h")), FULL_HOUSE);
        assert_eq!(category(ev("9h 9c 9d 2s 2c 2d 4h")), FULL_HOUSE);
        assert_eq!(category(ev("Ah 9h 7h 4h 2h 2c 2d")), FLUSH);
        assert_eq!(category(ev("Ah 2c 3d 4h 5s 9c Kd")), STRAIGHT);
        assert_eq!(category(ev("Ah Ac Ad 4h 5s 9c Kd")), TRIPS);
        assert_eq!(category(ev("Ah Ac 4d 4h 5s 5c Kd")), TWO_PAIR);
        assert_eq!(category(ev("Ah Ac 4d 7h 9s Jc Kd")), PAIR);
        assert_eq!(category(ev("Ah 3c 4d 7h 9s Jc Kd")), HIGH_CARD);
        // wheel < six-high straight
        assert!(ev("Ah 2c 3d 4h 5s Jc Kd") < ev("6h 2c 3d 4h 5s Jc Kd"));
        // third pair can be the two-pair kicker
        assert_eq!(ev("Ah Ac Kd Kh 7s 7c 2d"), pack(TWO_PAIR, &[14, 13, 7]));
    }
}
