//! Small shared helpers: CLI flags and random test positions.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use poker_core::actions::{sample_trunc_normal, to_action, NUM_TYPES};
use poker_core::encode::encode;
use poker_core::table::deal_training_hand;

/// `--key value` lookup.
pub fn flag(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1)).cloned()
}

pub fn has_flag(args: &[String], key: &str) -> bool {
    args.iter().any(|a| a == key)
}

/// `n` encoded decision points from random play on sampled training tables
/// (uniform legal action type, random raise size) — varied, realistic inputs
/// for parity fixtures and smoke tests.
pub fn random_positions(n: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut s = deal_training_hand(&mut rng);
        while !s.hand_done() && out.len() < n {
            match s.phase {
                poker_core::rules::Phase::Betting if s.to_act >= 0 => {
                    let (x, sp) = encode(&s).expect("encode");
                    out.push(x);
                    let legal: Vec<usize> = (0..NUM_TYPES).filter(|&t| sp.mask[t] > 0.0).collect();
                    let ty = legal[rng.gen_range(0..legal.len())];
                    let sz = sample_trunc_normal(-0.5, 0.8, sp.s_lo, sp.s_hi, &mut rng);
                    s.apply(to_action(&sp, ty, sz));
                }
                _ => s.advance_street(),
            }
        }
    }
    out
}
