//! Cards as compact indices, matching `src/engine/cards.ts`.
//!
//! A card is a `u8` in 0..52: `suit * 13 + (rank - 2)`, with suits in the TS
//! order c, d, h, s and ranks 2..=14 (11=J … 14=A).

pub type Card = u8;

pub const NUM_CARDS: usize = 52;
pub const SUIT_CHARS: [char; 4] = ['c', 'd', 'h', 's'];

#[inline]
pub fn make(rank: u8, suit: u8) -> Card {
    debug_assert!((2..=14).contains(&rank) && suit < 4);
    suit * 13 + (rank - 2)
}

/// Rank 2..=14.
#[inline]
pub fn rank(c: Card) -> u8 {
    c % 13 + 2
}

/// Suit 0..4 (c, d, h, s).
#[inline]
pub fn suit(c: Card) -> u8 {
    c / 13
}

/// The TS card id, e.g. "Ah", "10s", "2c".
pub fn id(c: Card) -> String {
    let r = match rank(c) {
        11 => "J".to_string(),
        12 => "Q".to_string(),
        13 => "K".to_string(),
        14 => "A".to_string(),
        n => n.to_string(),
    };
    format!("{r}{}", SUIT_CHARS[suit(c) as usize])
}

/// Parse a TS card id ("Ah", "10s", …).
pub fn parse(id: &str) -> Option<Card> {
    let (r, s) = id.split_at(id.len().checked_sub(1)?);
    let suit = SUIT_CHARS.iter().position(|&c| s.starts_with(c))? as u8;
    let rank = match r {
        "J" => 11,
        "Q" => 12,
        "K" => 13,
        "A" => 14,
        n => n.parse::<u8>().ok().filter(|n| (2..=10).contains(n))?,
    };
    Some(make(rank, suit))
}

/// A fresh, ordered 52-card deck (TS `createDeck` order: suit-major c,d,h,s).
pub fn full_deck() -> Vec<Card> {
    (0..NUM_CARDS as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip() {
        for c in 0..52u8 {
            assert_eq!(parse(&id(c)), Some(c));
        }
        assert_eq!(id(make(10, 3)), "10s");
        assert_eq!(parse("Ah"), Some(make(14, 2)));
    }
}
