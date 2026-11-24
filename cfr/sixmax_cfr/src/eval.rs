//! Fast matches on the abstract game: a strategy (or scripted bot) in one
//! seat against another in every other seat, with every seat taken in turn
//! on the same deal (so card luck mostly cancels). Everyone plays the tree's
//! own actions, so nothing is translated: this measures the strategies, not
//! the play-time engine (`crate::engine`, matches under the real rules).

use std::sync::atomic::{AtomicI64, AtomicU64, Ordering::Relaxed};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::classes::{class_name, N_CLASSES};
use crate::deal::{Dealer, PokerDealer};
use crate::strategy::Strategy;
use crate::tree::{Act, Tree, MAX_ACTIONS, TERMINAL};

/// Scripted opponents, choosing among the tree's actions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Bot {
    /// Uniform over the actions.
    Random,
    /// Always checks or calls.
    Station,
    /// The largest bet/raise short of all-in whenever it can, else calls.
    Maniac,
    /// The smallest bet/raise whenever it can, else calls.
    MinRaiser,
    /// All-in whenever it can, else calls.
    Shover,
    /// Preflop plays only premium hands (raises them, folds the rest), then
    /// checks and calls.
    Nit,
}

pub const BOTS: [(&str, Bot); 6] = [
    ("random", Bot::Random),
    ("station", Bot::Station),
    ("maniac", Bot::Maniac),
    ("minraiser", Bot::MinRaiser),
    ("shover", Bot::Shover),
    ("nit", Bot::Nit),
];

const PREMIUM: [&str; 11] = ["AA", "KK", "QQ", "JJ", "TT", "99", "AKs", "AKo", "AQs", "AQo", "KQs"];

pub enum Player<'a> {
    Table(&'a Strategy),
    Bot(Bot),
}

impl Player<'_> {
    fn choose(&self, t: &Tree, node: u32, bucket: u32, premium: &[bool; N_CLASSES], rng: &mut SmallRng) -> usize {
        let nd = &t.nodes[node as usize];
        let n = nd.n as usize;
        let acts = t.actions(nd);
        let passive = acts.iter().position(|a| matches!(a, Act::Check | Act::Call)).unwrap_or(0);
        let sized: Vec<usize> = (0..n).filter(|&a| matches!(acts[a], Act::Bet { .. } | Act::Raise { .. })).collect();
        let allin = (0..n).find(|&a| matches!(acts[a], Act::AllIn { .. }));
        match self {
            Player::Table(s) => {
                let mut p = [0f64; MAX_ACTIONS];
                s.probs(t, node, bucket, &mut p);
                let mut x: f64 = rng.gen();
                for (a, &w) in p[..n].iter().enumerate() {
                    if x < w {
                        return a;
                    }
                    x -= w;
                }
                n - 1
            }
            Player::Bot(Bot::Random) => rng.gen_range(0..n),
            Player::Bot(Bot::Station) => passive,
            Player::Bot(Bot::Maniac) => sized.last().copied().unwrap_or(passive),
            Player::Bot(Bot::MinRaiser) => sized.first().copied().unwrap_or(passive),
            Player::Bot(Bot::Shover) => allin.unwrap_or(passive),
            Player::Bot(Bot::Nit) => {
                if nd.street == 0 {
                    if premium[bucket as usize] {
                        sized.first().copied().or(allin).unwrap_or(passive)
                    } else if nd.has_fold {
                        0
                    } else {
                        passive
                    }
                } else {
                    passive
                }
            }
        }
    }
}

pub struct MatchResult {
    /// X's mean result, big blinds per hand, and the half-width of its 95% interval.
    pub bb_per_hand: f64,
    pub ci: f64,
    pub deals: u64,
}

/// `x` in one seat, `y` in the others, every seat in turn on each deal.
pub fn play_match(t: &Tree, dealer: &PokerDealer, x: &Player, y: &Player, deals: u64, threads: usize, seed: u64) -> MatchResult {
    let mut premium = [false; N_CLASSES];
    for (c, p) in premium.iter_mut().enumerate() {
        *p = PREMIUM.contains(&class_name(c as u32).as_str());
    }
    let sum = AtomicI64::new(0);
    let sumsq = AtomicU64::new(0);
    std::thread::scope(|sc| {
        for th in 0..threads as u64 {
            let share = deals / threads as u64 + u64::from(th < deals % threads as u64);
            let (sum, sumsq, premium) = (&sum, &sumsq, &premium);
            sc.spawn(move || {
                let mut rng = SmallRng::seed_from_u64(seed ^ (th + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let (mut s1, mut s2) = (0f64, 0f64);
                for _ in 0..share {
                    let d = dealer.deal(&mut rng);
                    let mut res = 0f64;
                    for r in t.seats() {
                        let mut node = 0u32;
                        loop {
                            let nd = &t.nodes[node as usize];
                            if nd.player == TERMINAL {
                                res += t.utility(&t.terms[nd.first as usize], r, &d.rank);
                                break;
                            }
                            let q = nd.player as usize;
                            let who = if q == r { x } else { y };
                            let a = who.choose(t, node, d.bucket[q][nd.street as usize], premium, &mut rng);
                            node = t.children(nd)[a];
                        }
                    }
                    let v = res / t.players as f64 / 2.0; // big blinds per hand
                    s1 += v;
                    s2 += v * v;
                }
                sum.fetch_add((s1 * 1e6) as i64, Relaxed);
                sumsq.fetch_add((s2 * 1e6) as u64, Relaxed);
            });
        }
    });
    let (s1, s2) = (sum.into_inner() as f64 / 1e6, sumsq.into_inner() as f64 / 1e6);
    let mean = s1 / deals as f64;
    let var = (s2 / deals as f64 - mean * mean).max(0.0);
    MatchResult { bb_per_hand: mean, ci: 1.96 * (var / deals as f64).sqrt(), deals }
}
