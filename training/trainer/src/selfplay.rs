//! Search-free self-play: every seat at every table plays the current network.
//!
//! `n_slots` hands are in flight. Each step:
//!   1. (parallel, CPU) every slot advances to its next decision — finishing
//!      hands and dealing new ones as needed — and encodes it;
//!   2. (GPU) one batched forward over all slots;
//!   3. (parallel, CPU) every slot samples an action from the policy
//!      (temperature 1: masked softmax over action types, then a truncated
//!      normal for a raise size), records the decision, and applies it.
//! When a hand ends, each seat's chip result (all-in-EV adjusted when the
//! board was run out with no decisions left) becomes `z` for all of that
//! seat's decisions in the hand. Stops once `target_rows` decisions are done;
//! hands still in flight are discarded.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use tch::{Device, Kind, Tensor};

use poker_core::actions::{sample_trunc_normal, to_action, trunc_normal_logpdf, ActionSpace, NUM_TYPES, RAISE};
use poker_core::allin_ev::{expected_stacks, is_runout};
use poker_core::encode::{encode_into, INPUT_SIZE};
use poker_core::rules::{Phase, State, SEATS};
use poker_core::table::deal_training_hand;

use crate::hud::Hud;
use crate::net::PokerNet;

/// Chips per unit of reward (the 100bb starting stack).
pub const Z_SCALE: f64 = 200.0;

/// One recorded decision.
pub struct Row {
    pub x: Vec<f32>,
    pub mask: [f32; NUM_TYPES],
    pub lo: f32,
    pub hi: f32,
    pub ty: u8,
    pub s: f32,
    /// Behavior log-prob of (type, size) under the sampling policy.
    pub logp: f32,
    pub z: f32,
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

    /// Sample from the policy outputs, record the decision, apply it.
    fn act(&mut self, x: &[f32], logits: &[f32], size: &[f32]) {
        let sp = self.sp.take().expect("pending decision");
        let (ty, s, logp) = sample_policy(&sp, logits, size, &mut self.rng);
        self.hud.record_decision(ty, s);
        let seat = self.st.to_act as usize;
        self.pending.push((
            seat,
            Row {
                x: x.to_vec(),
                mask: sp.mask,
                lo: sp.s_lo as f32,
                hi: sp.s_hi as f32,
                ty: ty as u8,
                s: s as f32,
                logp: logp as f32,
                z: 0.0,
            },
        ));
        self.st.apply(to_action(&sp, ty, s));
    }
}

/// Sample (action type, raise size) from the policy outputs for one decision
/// — temperature 1: masked softmax over the types, then a truncated normal
/// for a raise's size. Returns (type, s, log-prob of the sampled action).
pub fn sample_policy(sp: &ActionSpace, logits: &[f32], size: &[f32], rng: &mut impl Rng) -> (usize, f64, f64) {
    let mut lp = [f64::NEG_INFINITY; NUM_TYPES];
    let mx = (0..NUM_TYPES).filter(|&t| sp.mask[t] > 0.0).map(|t| logits[t] as f64).fold(f64::NEG_INFINITY, f64::max);
    let mut z = 0.0;
    for t in 0..NUM_TYPES {
        if sp.mask[t] > 0.0 {
            z += (logits[t] as f64 - mx).exp();
        }
    }
    for t in 0..NUM_TYPES {
        if sp.mask[t] > 0.0 {
            lp[t] = logits[t] as f64 - mx - z.ln();
        }
    }
    let r: f64 = rng.gen();
    let mut acc = 0.0;
    let mut ty = (0..NUM_TYPES).rev().find(|&t| sp.mask[t] > 0.0).unwrap();
    for t in 0..NUM_TYPES {
        if sp.mask[t] > 0.0 {
            acc += lp[t].exp();
            if r < acc {
                ty = t;
                break;
            }
        }
    }
    let mut logp = lp[ty];
    let mut s = sp.s_lo;
    if ty == RAISE && sp.s_hi > sp.s_lo {
        let (mu, sigma) = (size[0] as f64, size[1] as f64);
        s = sample_trunc_normal(mu, sigma, sp.s_lo, sp.s_hi, rng);
        logp += trunc_normal_logpdf(s, mu, sigma, sp.s_lo, sp.s_hi);
    }
    (ty, s, logp)
}

pub struct Cohort {
    pub rows: Vec<Row>,
    pub hud: Hud,
    pub forwards: usize,
}

/// Play self-play hands with `net` until at least `target_rows` decisions
/// from finished hands are collected.
pub fn run_cohort(net: &PokerNet, dev: Device, n_slots: usize, target_rows: usize, seed: u64) -> Cohort {
    let mut slots: Vec<Slot> = (0..n_slots)
        .into_par_iter()
        .map(|i| Slot::new(seed.wrapping_mul(1_000_003).wrapping_add(i as u64)))
        .collect();
    let mut buf = vec![0f32; n_slots * INPUT_SIZE];
    let mut forwards = 0;
    loop {
        slots
            .par_iter_mut()
            .zip(buf.par_chunks_mut(INPUT_SIZE))
            .for_each(|(slot, x)| slot.advance_and_encode(x));
        let done: usize = slots.iter().map(|s| s.done.len()).sum();
        if done >= target_rows {
            break;
        }
        let xs = Tensor::from_slice(&buf).view([n_slots as i64, INPUT_SIZE as i64]).to_device(dev);
        let out = tch::no_grad(|| net.forward(&xs));
        let to_vec = |t: &Tensor| {
            Vec::<f32>::try_from(&t.to_kind(Kind::Float).to_device(Device::Cpu).contiguous().view([-1])).unwrap()
        };
        let logits = to_vec(&out.type_logits);
        let size = to_vec(&out.size_params);
        forwards += 1;
        slots
            .par_iter_mut()
            .zip(buf.par_chunks(INPUT_SIZE))
            .zip(logits.par_chunks(NUM_TYPES).zip(size.par_chunks(2)))
            .for_each(|((slot, x), (l, sz))| slot.act(x, l, sz));
    }
    let mut hud = Hud::default();
    let mut rows = Vec::new();
    for s in slots {
        hud.merge(&s.hud);
        rows.extend(s.done);
    }
    Cohort { rows, hud, forwards }
}
