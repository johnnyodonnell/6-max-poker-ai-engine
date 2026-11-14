//! Training table setups: every self-play hand starts from a freshly sampled
//! table so the bot sees every situation the app can produce.
//!
//! The app is a freeze-out: 6 × 200 chips to start, then stacks drift and
//! busted seats sit out, but the live players always hold all 1,200 chips.
//! So: 40% of hands use the opening table (6 × 200); the rest seat 2–6 live
//! players (random seats) splitting 1,200 chips unevenly, with a random button.

use rand::seq::SliceRandom;
use rand::Rng;
use rand_distr::{Distribution, Gamma};

use crate::cards::full_deck;
use crate::rules::{State, SEATS, STARTING_STACK};

pub const TOTAL_CHIPS: i32 = SEATS as i32 * STARTING_STACK;
pub const FRESH_TABLE_PROB: f64 = 0.4;

/// Relative weights for 2, 3, 4, 5, 6 live players on a non-fresh table.
const LIVE_WEIGHTS: [f64; 5] = [0.15, 0.15, 0.2, 0.2, 0.3];
/// Dirichlet concentration for splitting the chips (lower = more uneven).
const SPLIT_ALPHA: f64 = 1.5;

/// Stacks for a sampled table (0 = seat is out).
pub fn sample_stacks(rng: &mut impl Rng) -> [i32; SEATS] {
    if rng.gen_bool(FRESH_TABLE_PROB) {
        return [STARTING_STACK; SEATS];
    }
    let mut r = rng.gen::<f64>() * LIVE_WEIGHTS.iter().sum::<f64>();
    let mut live = 2;
    for (k, w) in LIVE_WEIGHTS.iter().enumerate() {
        if r < *w {
            live = k + 2;
            break;
        }
        r -= w;
    }
    let mut seats: Vec<usize> = (0..SEATS).collect();
    seats.shuffle(rng);
    let seats = &seats[..live];

    // Dirichlet split of the chips, every live player keeping at least 1.
    let gamma = Gamma::new(SPLIT_ALPHA, 1.0).unwrap();
    let w: Vec<f64> = (0..live).map(|_| gamma.sample(rng)).collect();
    let wsum: f64 = w.iter().sum();
    let spare = TOTAL_CHIPS - live as i32;
    let mut stacks = [0i32; SEATS];
    let mut given = 0;
    for (k, &seat) in seats.iter().enumerate() {
        let x = (w[k] / wsum * spare as f64).floor() as i32;
        stacks[seat] = 1 + x;
        given += 1 + x;
    }
    // Rounding leftovers go to the first live seat.
    stacks[seats[0]] += TOTAL_CHIPS - given;
    stacks
}

/// A new training hand: sampled stacks, random button, shuffled deck, dealt.
pub fn deal_training_hand(rng: &mut impl Rng) -> State {
    let mut s = State::new_table(sample_stacks(rng), None);
    s.button = rng.gen_range(0..SEATS as i32); // start_hand moves it to the next live seat
    let mut deck = full_deck();
    deck.shuffle(rng);
    s.start_hand(deck);
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn stacks_always_sum_to_1200() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut live_counts = [0usize; 7];
        for _ in 0..20000 {
            let st = sample_stacks(&mut rng);
            assert_eq!(st.iter().sum::<i32>(), TOTAL_CHIPS);
            let live = st.iter().filter(|&&x| x > 0).count();
            assert!((2..=6).contains(&live));
            live_counts[live] += 1;
        }
        assert!(live_counts[2..].iter().all(|&n| n > 500), "{live_counts:?}");
    }
}
