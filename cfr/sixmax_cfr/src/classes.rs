//! The 169 strategically distinct preflop hands, laid out like a standard
//! 13×13 chart: row/column 0 = ace … 12 = deuce; pairs on the diagonal,
//! suited hands above it (row = high card), offsuit below (row = low card).

use poker_core::cards::{rank, suit, Card};

pub const N_CLASSES: usize = 169;

/// Chart index of a two-card hand.
#[inline]
pub fn class_of(a: Card, b: Card) -> u32 {
    let (ra, rb) = (rank(a) as u32, rank(b) as u32);
    let (hi, lo) = if ra >= rb { (ra, rb) } else { (rb, ra) };
    let (row, col) = if hi == lo || suit(a) == suit(b) { (14 - hi, 14 - lo) } else { (14 - lo, 14 - hi) };
    row * 13 + col
}

/// e.g. "AA", "AKs", "72o".
pub fn class_name(c: u32) -> String {
    let (row, col) = (c / 13, c % 13);
    let ch = |i: u32| "AKQJT98765432".as_bytes()[i as usize] as char;
    match row.cmp(&col) {
        std::cmp::Ordering::Equal => format!("{}{}", ch(row), ch(col)),
        std::cmp::Ordering::Less => format!("{}{}s", ch(row), ch(col)),
        std::cmp::Ordering::Greater => format!("{}{}o", ch(col), ch(row)),
    }
}

/// Number of two-card combinations in a class (6, 4 or 12).
pub fn combos(c: u32) -> u32 {
    let (row, col) = (c / 13, c % 13);
    match row.cmp(&col) {
        std::cmp::Ordering::Equal => 6,
        std::cmp::Ordering::Less => 4,
        std::cmp::Ordering::Greater => 12,
    }
}

/// Every two-card hand (as (low card, high card) indices) in each class.
pub fn class_hands() -> Vec<Vec<(Card, Card)>> {
    let mut out = vec![Vec::new(); N_CLASSES];
    for a in 0..52u8 {
        for b in (a + 1)..52u8 {
            out[class_of(a, b) as usize].push((a, b));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use poker_core::cards::parse;

    #[test]
    fn layout() {
        let c = |s: &str| {
            let v: Vec<Card> = s.split_whitespace().map(|x| parse(x).unwrap()).collect();
            class_of(v[0], v[1])
        };
        assert_eq!(class_name(c("Ah As")), "AA");
        assert_eq!(class_name(c("Ah Kh")), "AKs");
        assert_eq!(class_name(c("Kd As")), "AKo");
        assert_eq!(class_name(c("7c 2d")), "72o");
        assert_eq!(class_name(c("2c 2d")), "22");
        let hands = class_hands();
        assert_eq!(hands.iter().map(|h| h.len()).sum::<usize>(), 1326);
        for (i, h) in hands.iter().enumerate() {
            assert_eq!(h.len() as u32, combos(i as u32), "{}", class_name(i as u32));
        }
    }
}
