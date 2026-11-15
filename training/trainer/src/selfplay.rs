//! Search-free self-play: every seat at every table plays the current policy.
//!
//! `n_slots` hands are in flight. Each step:
//!   1. (parallel, CPU) every slot advances to its next decision — finishing
//!      hands and dealing new ones as needed — and encodes it;
//!   2. (GPU) one batched forward over all slots;
//!   3. (parallel, CPU) every slot samples a discrete action from the masked
//!      softmax (temperature 1), records the decision (with the critic's
//!      privileged inputs), and applies it.
//! When a hand ends, each seat's chip result (all-in-EV adjusted when the
//! board was run out with no decisions left) becomes `z` for all of that
//! seat's decisions in the hand.
//!
//! The slots persist across cohorts: a cohort takes the decisions of hands
//! that *finished* during it, and hands still in flight carry over to the
//! next cohort (their early decisions were sampled by the previous weights,
//! which the stored behavior probabilities account for). Throwing them away
//! instead would drop long, big-pot hands preferentially.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

use poker_core::actions::{pot_fraction, to_action, ActionSpace, A_ALLIN, A_FOLD, A_MINRAISE, NUM_ACTIONS};
use poker_core::allin_ev::{expected_stacks, is_runout};
use poker_core::cards::Card;
use poker_core::encode::{canonical_index, encode_into, suit_map, INPUT_SIZE, NO_CARD};
use poker_core::equity::equity;
use poker_core::rules::{Phase, State, Status, SEATS};
use poker_core::table::deal_training_hand;

use crate::hud::Hud;
use crate::net::{PokerNet, PRIV_SIZE};

/// Chips per unit of reward (the 100bb starting stack).
pub const Z_SCALE: f64 = 200.0;
/// Monte Carlo samples for the critic's equity input.
const EQUITY_SAMPLES: usize = 32;

/// One recorded decision.
pub struct Row {
    pub x: Vec<f32>,
    /// Critic-only inputs (see net::PRIV_SIZE).
    pub p: [f32; PRIV_SIZE],
    pub amask: [f32; NUM_ACTIONS],
    /// Behavior policy (masked softmax) the action was sampled from.
    pub probs: [f32; NUM_ACTIONS],
    pub a: u8,
    /// Exact value of folding here: the chips already put in, lost (/ Z_SCALE).
    pub fold_q: f32,
    pub z: f32,
}

/// Masked softmax over the legal actions, then a sample. Returns (action, probs).
pub fn sample_action(sp: &ActionSpace, logits: &[f32], rng: &mut impl Rng) -> (usize, [f32; NUM_ACTIONS]) {
    let mx = (0..NUM_ACTIONS).filter(|&a| sp.amask[a] > 0.0).map(|a| logits[a] as f64).fold(f64::NEG_INFINITY, f64::max);
    let mut w = [0f64; NUM_ACTIONS];
    let mut tot = 0.0;
    for a in 0..NUM_ACTIONS {
        if sp.amask[a] > 0.0 {
            w[a] = (logits[a] as f64 - mx).exp();
            tot += w[a];
        }
    }
    let mut probs = [0f32; NUM_ACTIONS];
    for a in 0..NUM_ACTIONS {
        probs[a] = (w[a] / tot) as f32;
    }
    let r: f64 = rng.gen::<f64>() * tot;
    let mut acc = 0.0;
    let mut choice = (0..NUM_ACTIONS).rev().find(|&a| sp.amask[a] > 0.0).unwrap();
    for a in 0..NUM_ACTIONS {
        if sp.amask[a] > 0.0 {
            acc += w[a];
            if r < acc {
                choice = a;
                break;
            }
        }
    }
    (choice, probs)
}

/// The critic's privileged inputs for the player to act: every live
/// opponent's hole cards (relative seat order, canonical suits extending the
/// mover's mapping), the mover's showdown equity against them, and the live
/// opponent count / 5.
fn privileged(st: &State, rng: &mut impl Rng) -> [f32; PRIV_SIZE] {
    let mover = st.to_act as usize;
    let me = &st.players[mover];
    let mut map = suit_map(&me.hole, &st.community);
    let mut out = [NO_CARD; PRIV_SIZE];
    let mut opps: Vec<&[Card]> = Vec::new();
    let mut dead: Vec<Card> = Vec::new();
    for r in 1..SEATS {
        let p = &st.players[(mover + r) % SEATS];
        if p.hole.len() < 2 {
            continue;
        }
        if matches!(p.status, Status::Active | Status::AllIn) {
            out[2 * (r - 1)] = canonical_index(p.hole[0], &mut map);
            out[2 * (r - 1) + 1] = canonical_index(p.hole[1], &mut map);
            opps.push(&p.hole);
        } else {
            dead.extend_from_slice(&p.hole);
        }
    }
    out[10] = equity(&me.hole, &opps, &st.community, &dead, EQUITY_SAMPLES, rng) as f32;
    out[11] = opps.len() as f32 / 5.0;
    out
}

struct Slot {
    st: State,
    rng: StdRng,
    start: [i32; SEATS],
    pending: Vec<(usize, Row)>,
    done: Vec<Row>,
    sp: Option<ActionSpace>,
    hud: Hud,
}

impl Slot {
    fn new(seed: u64) -> Slot {
        let mut rng = StdRng::seed_from_u64(seed);
        let st = deal_training_hand(&mut rng);
        let mut s = Slot { st, rng, start: [0; SEATS], pending: Vec::new(), done: Vec::new(), sp: None, hud: Hud::default() };
        s.mark_start();
        s
    }

    fn mark_start(&mut self) {
        for (i, p) in self.st.players.iter().enumerate() {
            self.start[i] = p.stack + p.commit; // stacks before the blinds
        }
    }

    fn finish(&mut self, final_stacks: [f64; SEATS]) {
        self.hud.record_hand(&self.st);
        for (seat, mut row) in self.pending.drain(..) {
            row.z = ((final_stacks[seat] - self.start[seat] as f64) / Z_SCALE) as f32;
            self.done.push(row);
        }
        self.st = deal_training_hand(&mut self.rng);
        self.mark_start();
    }

    /// Advance to the next decision and encode it into `x`.
    fn advance_and_encode(&mut self, x: &mut [f32]) {
        loop {
            match self.st.phase {
                Phase::Betting if self.st.to_act >= 0 => {
                    self.sp = Some(encode_into(&self.st, x).expect("decision"));
                    return;
                }
                Phase::Betting | Phase::StreetComplete => {
                    if is_runout(&self.st) {
                        let ev = expected_stacks(&self.st, &mut self.rng);
                        self.finish(ev);
                    } else {
                        self.st.advance_street();
                    }
                }
                Phase::HandOver | Phase::MatchOver => {
                    let mut f = [0f64; SEATS];
                    for i in 0..SEATS {
                        f[i] = self.st.players[i].stack as f64;
                    }
                    self.finish(f);
                }
                Phase::Waiting => unreachable!("training hands are dealt on creation"),
            }
        }
    }

    /// Sample from the policy logits, record the decision, apply it.
    fn act(&mut self, x: &[f32], logits: &[f32]) {
        let sp = self.sp.take().expect("pending decision");
        let (a, probs) = sample_action(&sp, logits, &mut self.rng);
        let sized = (A_MINRAISE..A_ALLIN).contains(&a);
        self.hud.record_decision(a, if sized { pot_fraction(&sp, a) } else { 0.0 });
        let seat = self.st.to_act as usize;
        let p = privileged(&self.st, &mut self.rng);
        let fold_q = ((self.st.players[seat].stack - self.start[seat]) as f64 / Z_SCALE) as f32;
        self.pending.push((seat, Row { x: x.to_vec(), p, amask: sp.amask, probs, a: a as u8, fold_q, z: 0.0 }));
        debug_assert!(a != A_FOLD || sp.amask[A_FOLD] > 0.0);
        self.st.apply(to_action(&sp, a));
    }
}

pub struct Cohort {
    pub rows: Vec<Row>,
    pub hud: Hud,
    pub forwards: usize,
}

/// Self-play tables that live for the whole run.
pub struct SelfPlay {
    slots: Vec<Slot>,
    buf: Vec<f32>,
}

impl SelfPlay {
    pub fn new(n_slots: usize, seed: u64) -> SelfPlay {
        let slots = (0..n_slots).into_par_iter().map(|i| Slot::new(seed.wrapping_mul(1_000_003).wrapping_add(i as u64))).collect();
        SelfPlay { slots, buf: vec![0f32; n_slots * INPUT_SIZE] }
    }

    /// Play until at least `target_rows` decisions from hands finished in this
    /// call are collected; unfinished hands continue next call.
    pub fn run_cohort(&mut self, net: &PokerNet, dev: Device, target_rows: usize) -> Cohort {
        let n = self.slots.len();
        let mut forwards = 0;
        loop {
            self.slots.par_iter_mut().zip(self.buf.par_chunks_mut(INPUT_SIZE)).for_each(|(slot, x)| slot.advance_and_encode(x));
            let done: usize = self.slots.iter().map(|s| s.done.len()).sum();
            if done >= target_rows {
                break;
            }
            let xs = Tensor::from_slice(&self.buf).view([n as i64, INPUT_SIZE as i64]).to_device(dev);
            let out = tch::no_grad(|| net.forward(&xs));
            let logits = Vec::<f32>::try_from(&out.to_kind(Kind::Float).to_device(Device::Cpu).contiguous().view([-1])).unwrap();
            forwards += 1;
            self.slots
                .par_iter_mut()
                .zip(self.buf.par_chunks(INPUT_SIZE))
                .zip(logits.par_chunks(NUM_ACTIONS))
                .for_each(|((slot, x), l)| slot.act(x, l));
        }
        let mut hud = Hud::default();
        let mut rows = Vec::new();
        for s in self.slots.iter_mut() {
            hud.merge(&std::mem::take(&mut s.hud));
            rows.append(&mut s.done);
        }
        Cohort { rows, hud, forwards }
    }
}
