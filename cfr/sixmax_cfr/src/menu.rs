//! The bet-size menu of the abstract game.
//!
//! A size is a fraction of the pot *after calling*: raise-to = current bet +
//! f × (pot + to-call), rounded to whole chips (half to even), raised to the
//! legal minimum, dropped if it would be all-in (all-in is its own action) and
//! deduplicated. The k-th bet/raise on a street uses list k (the last list
//! repeats). After `cap` bets/raises on a street only fold/call/all-in remain.
//!
//! Pots of three or more players after the flop get a leaner menu (one bet
//! size, pot raises, a lower cap): most of a 6-max tree is multiway, and
//! multiway pots are rare in play.

#[derive(Clone, Debug)]
pub struct Menu {
    pub name: &'static str,
    pub preflop: Vec<Vec<f64>>,
    pub preflop_cap: u32,
    /// Two players in the hand, after the flop.
    pub heads_up: Vec<Vec<f64>>,
    pub heads_up_cap: u32,
    /// Three or more players in the hand, after the flop.
    pub multiway: Vec<Vec<f64>>,
    pub multiway_cap: u32,
}

impl Menu {
    /// v1: two opens (to about 2.5 and 3.5 big blinds from UTG), one 3-bet
    /// and one 4-bet size; after the flop, bets of a third, three quarters
    /// and 1.5× the pot heads-up (half the pot multiway), pot raises.
    pub fn v1() -> Menu {
        Menu {
            name: "v1",
            preflop: vec![vec![0.6, 1.0], vec![1.0], vec![1.0]],
            preflop_cap: 3,
            heads_up: vec![vec![0.33, 0.75, 1.5], vec![1.0]],
            heads_up_cap: 3,
            multiway: vec![vec![0.5], vec![1.0]],
            multiway_cap: 2,
        }
    }

    pub fn by_name(name: &str) -> Option<Menu> {
        match name {
            "v1" => Some(Menu::v1()),
            _ => None,
        }
    }

    /// Sizes for the `raises`-th bet/raise on `street` (0 = preflop) with
    /// `in_hand` players still in the hand.
    pub fn sizes(&self, street: usize, raises: u32, in_hand: usize) -> &[f64] {
        let lists = if street == 0 {
            &self.preflop
        } else if in_hand >= 3 {
            &self.multiway
        } else {
            &self.heads_up
        };
        &lists[(raises as usize).min(lists.len() - 1)]
    }

    pub fn cap(&self, street: usize, in_hand: usize) -> u32 {
        if street == 0 {
            self.preflop_cap
        } else if in_hand >= 3 {
            self.multiway_cap
        } else {
            self.heads_up_cap
        }
    }
}
