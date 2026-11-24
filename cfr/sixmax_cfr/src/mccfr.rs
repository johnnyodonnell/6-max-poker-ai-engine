//! External-sampling Monte Carlo CFR over a `Tree`, multithreaded.
//!
//! Each iteration samples one deal and traverses the tree once per player:
//! the traverser explores all of its actions and updates their regrets; the
//! other players sample one action from their current strategy. Folding ends
//! the hand for the traverser, so its value (minus what it has put in) needs
//! no walk.
//!
//! Following Pluribus's blueprint recipe:
//! - Linear-CFR-style discounting early on: every `discount_every`
//!   iterations, until `discount_until`, regrets and average sums are scaled
//!   by k/(k+1) (k = discounts so far + 1).
//! - Negative-regret pruning once `prune_after` iterations have run: on 95%
//!   of iterations, actions whose regret is below `prune_threshold` are not
//!   explored, unless they end the hand or are on the river.
//! - Regrets are 4-byte integers (hundredths of a chip), floored at
//!   `regret_floor` so pruned actions can recover.
//! - The average strategy is updated on a random 1 in `AVG_EVERY` iterations
//!   (with `AVG_EVERY` times the weight: unbiased), as Pluribus does: adding
//!   to the hottest preflop sums from every thread on every iteration halved
//!   the speed.
//! - The average strategy is kept only for the first `avg_streets` streets
//!   (preflop in the big runs: a strategy sum per entry wouldn't fit beside
//!   the regrets). Later streets play the final current strategy, which
//!   played at least as well as the average strategy and Pluribus-style
//!   snapshots in the 25bb pilot.
//!
//! Threads share the tables without locks: regret updates are atomic adds
//! (no lost updates), average sums are compare-and-swap adds of f64 bits,
//! and a traversal may read slightly stale values.

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering::Relaxed};

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

use crate::deal::{Deal, Dealer};
use crate::tree::{Tree, MAX_ACTIONS, TERMINAL};

/// Regrets are stored in hundredths of a chip.
pub const SCALE: f64 = 100.0;

/// The average strategy is updated on one iteration in this many.
pub const AVG_EVERY: u32 = 32;

#[derive(Clone, Debug)]
pub struct Config {
    pub threads: usize,
    pub discount_every: u64,
    pub discount_until: u64,
    pub prune_after: u64,
    /// In regret units (hundredths of a chip).
    pub prune_threshold: i32,
    pub regret_floor: i32,
}

impl Config {
    /// The heads-up engine's schedule for a run of `target` iterations at
    /// `stack` chips: discount every 0.5% for the first 20%, then prune
    /// (threshold -500 x stack chips, floor -515 x stack).
    pub fn schedule(target: u64, stack: i32, threads: usize) -> Config {
        Config {
            threads,
            discount_every: (target / 200).max(1),
            discount_until: target / 5,
            prune_after: target / 5,
            prune_threshold: (-500.0 * stack as f64 * SCALE) as i32,
            regret_floor: (-515.0 * stack as f64 * SCALE) as i32,
        }
    }

    /// No discounting or pruning (plain external-sampling MCCFR).
    pub fn plain(threads: usize) -> Config {
        Config {
            threads,
            discount_every: 0,
            discount_until: 0,
            prune_after: u64::MAX,
            prune_threshold: i32::MIN,
            regret_floor: i32::MIN,
        }
    }
}

/// Work counters: traverser decision nodes walked and regret entries updated.
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub visits: u64,
    pub updates: u64,
}

pub struct Solver {
    pub regret: Box<[AtomicI32]>,
    /// Average-strategy sums (f64 bits) for entries `0..avg_end`.
    pub avg: Box<[AtomicU64]>,
    pub avg_end: u64,
    pub avg_streets: usize,
    pub iterations: u64,
    pub discounts: u64,
    /// The run's planned length (0 if not set).
    pub target: u64,
    pub counts: Counts,
}

fn zeroed<T>(n: usize) -> Box<[T]> {
    // SAFETY: only used for AtomicI32/AtomicU64, whose all-zero bit pattern
    // is 0 (0.0 as f64 bits). alloc_zeroed lets the OS back pages lazily.
    unsafe {
        if n == 0 {
            return Box::from_raw(std::ptr::slice_from_raw_parts_mut(std::ptr::NonNull::<T>::dangling().as_ptr(), 0));
        }
        let layout = std::alloc::Layout::array::<T>(n).expect("table size");
        let p = std::alloc::alloc_zeroed(layout) as *mut T;
        assert!(!p.is_null(), "out of memory allocating {n} entries");
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, n))
    }
}

#[inline]
fn add_f64(a: &AtomicU64, d: f64) {
    let _ = a.fetch_update(Relaxed, Relaxed, |b| Some((f64::from_bits(b) + d).to_bits()));
}

/// Regret matching: positive regrets normalized, uniform if none are positive.
#[inline]
pub fn regret_matching(r: &[AtomicI32], out: &mut [f64]) {
    let mut sum = 0.0;
    for (o, x) in out.iter_mut().zip(r) {
        let v = x.load(Relaxed).max(0) as f64;
        *o = v;
        sum += v;
    }
    if sum > 0.0 {
        for o in out.iter_mut() {
            *o /= sum;
        }
    } else {
        out.fill(1.0 / out.len() as f64);
    }
}

#[inline]
pub fn sample(p: &[f64], rng: &mut SmallRng) -> usize {
    let mut x: f64 = rng.gen();
    for (i, &w) in p.iter().enumerate() {
        if x < w {
            return i;
        }
        x -= w;
    }
    p.iter().rposition(|&w| w > 0.0).unwrap_or(p.len() - 1)
}

struct Ctx<'a> {
    t: &'a Tree,
    regret: &'a [AtomicI32],
    avg: &'a [AtomicU64],
    avg_end: u64,
    /// Add to the average sums on this iteration (with weight AVG_EVERY).
    avg_now: bool,
    prune: bool,
    threshold: i32,
    floor: i32,
}

fn traverse(cx: &Ctx, node: u32, p: usize, d: &Deal, rng: &mut SmallRng, c: &mut Counts) -> f64 {
    let t = cx.t;
    let nd = &t.nodes[node as usize];
    if nd.player == TERMINAL {
        return t.utility(&t.terms[nd.first as usize], p, &d.rank);
    }
    let q = nd.player as usize;
    let n = nd.n as usize;
    let base = (nd.base + d.bucket[q][nd.street as usize] as u64 * n as u64) as usize;
    let r = &cx.regret[base..base + n];
    let mut sigma = [0f64; MAX_ACTIONS];
    regret_matching(r, &mut sigma[..n]);
    let kids = t.children(nd);
    if q != p {
        if cx.avg_now && (base as u64) < cx.avg_end {
            for a in 0..n {
                add_f64(&cx.avg[base + a], sigma[a] * AVG_EVERY as f64);
            }
        }
        let a = sample(&sigma[..n], rng);
        return traverse(cx, kids[a], p, d, rng, c);
    }
    c.visits += 1;
    let mut v = [0f64; MAX_ACTIONS];
    let mut explored = [false; MAX_ACTIONS];
    let mut vn = 0.0;
    let can_prune = cx.prune && nd.street < 3;
    for a in 0..n {
        if a == 0 && nd.has_fold {
            v[0] = -(nd.commit as f64);
        } else {
            if can_prune && r[a].load(Relaxed) < cx.threshold && t.nodes[kids[a] as usize].player != TERMINAL {
                continue;
            }
            v[a] = traverse(cx, kids[a], p, d, rng, c);
        }
        explored[a] = true;
        vn += sigma[a] * v[a];
    }
    for a in 0..n {
        if !explored[a] {
            continue;
        }
        c.updates += 1;
        let delta = ((v[a] - vn) * SCALE).round().clamp(i32::MIN as f64, i32::MAX as f64) as i32;
        if delta == 0 {
            continue;
        }
        let old = r[a].fetch_add(delta, Relaxed);
        let new = old.wrapping_add(delta);
        if delta > 0 && new < old {
            r[a].store(i32::MAX, Relaxed); // saturate instead of wrapping
        } else if new < cx.floor {
            r[a].store(cx.floor, Relaxed);
        }
    }
    vn
}

impl Solver {
    /// A fresh solver; the average strategy is kept for streets
    /// `0..avg_streets` (1 = preflop only, 4 = every street).
    pub fn new(t: &Tree, avg_streets: usize) -> Solver {
        let avg_end = t.street_start[avg_streets.min(4)];
        Solver {
            regret: zeroed(t.n_entries as usize),
            avg: zeroed(avg_end as usize),
            avg_end,
            avg_streets,
            iterations: 0,
            discounts: 0,
            target: 0,
            counts: Counts::default(),
        }
    }

    /// Run `iters` more iterations across `cfg.threads` threads, with
    /// pruning if the schedule says so (no discounting: see `train`).
    pub fn run(&mut self, t: &Tree, dealer: &dyn Dealer, iters: u64, cfg: &Config, seed: u64) -> Counts {
        let threads = cfg.threads.max(1) as u64;
        let prune = self.iterations >= cfg.prune_after;
        let start = self.iterations;
        let (regret, avg, avg_end) = (&self.regret[..], &self.avg[..], self.avg_end);
        let visits = AtomicU64::new(0);
        let updates = AtomicU64::new(0);
        std::thread::scope(|sc| {
            for th in 0..threads {
                let share = iters / threads + u64::from(th < iters % threads);
                if share == 0 {
                    continue;
                }
                let tseed = seed ^ start.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ (th + 1).wrapping_mul(0xD1B5_4A32_D192_ED03);
                let (visits, updates) = (&visits, &updates);
                sc.spawn(move || {
                    let mut rng = SmallRng::seed_from_u64(tseed);
                    let mut c = Counts::default();
                    for _ in 0..share {
                        let deal = dealer.deal(&mut rng);
                        let cx = Ctx {
                            t,
                            regret,
                            avg,
                            avg_end,
                            avg_now: avg_end > 0 && rng.gen_ratio(1, AVG_EVERY),
                            prune: prune && rng.gen::<f64>() < 0.95,
                            threshold: cfg.prune_threshold,
                            floor: cfg.regret_floor,
                        };
                        for p in t.seats() {
                            traverse(&cx, 0, p, &deal, &mut rng, &mut c);
                        }
                    }
                    visits.fetch_add(c.visits, Relaxed);
                    updates.fetch_add(c.updates, Relaxed);
                });
            }
        });
        self.iterations += iters;
        let c = Counts { visits: visits.into_inner(), updates: updates.into_inner() };
        self.counts.visits += c.visits;
        self.counts.updates += c.updates;
        c
    }

    /// Scale every regret and average sum by `f`.
    pub fn discount(&mut self, f: f64) {
        self.regret.par_chunks(1 << 16).for_each(|ch| {
            for x in ch {
                let r = x.load(Relaxed);
                if r != 0 {
                    x.store((r as f64 * f).round() as i32, Relaxed);
                }
            }
        });
        self.avg.par_chunks(1 << 16).for_each(|ch| {
            for x in ch {
                let a = f64::from_bits(x.load(Relaxed));
                if a != 0.0 {
                    x.store((a * f).to_bits(), Relaxed);
                }
            }
        });
    }

    /// Run up to `target` total iterations following the schedule, calling
    /// `report` after each chunk of at most `chunk` iterations with that
    /// chunk's work counts. `report` returns false to stop early.
    #[allow(clippy::too_many_arguments)]
    pub fn train(
        &mut self,
        t: &Tree,
        dealer: &dyn Dealer,
        target: u64,
        chunk: u64,
        cfg: &Config,
        seed: u64,
        mut report: impl FnMut(&Solver, Counts, u64) -> bool,
    ) {
        while self.iterations < target {
            let mut next = (self.iterations + chunk.max(1)).min(target);
            let discounting = cfg.discount_every > 0 && self.iterations < cfg.discount_until;
            if discounting {
                let boundary = (self.iterations / cfg.discount_every + 1) * cfg.discount_every;
                next = next.min(boundary);
            }
            if self.iterations < cfg.prune_after && cfg.prune_after < next {
                next = cfg.prune_after;
            }
            let n = next - self.iterations;
            let c = self.run(t, dealer, n, cfg, seed);
            if discounting && self.iterations % cfg.discount_every == 0 {
                self.discounts += 1;
                let k = self.discounts as f64;
                self.discount(k / (k + 1.0));
            }
            if !report(self, c, n) {
                break;
            }
        }
    }

    /// The current strategy (regret matching) at (node, group).
    pub fn current(&self, t: &Tree, node: u32, bucket: u32, out: &mut [f64]) {
        let nd = &t.nodes[node as usize];
        let n = nd.n as usize;
        let base = (nd.base + bucket as u64 * n as u64) as usize;
        regret_matching(&self.regret[base..base + n], &mut out[..n]);
    }

    /// The strategy the bot plays at (node, group): the average strategy on
    /// the streets that keep one (falling back to the current strategy where
    /// nothing was averaged), the current strategy elsewhere.
    pub fn strategy(&self, t: &Tree, node: u32, bucket: u32, out: &mut [f64]) {
        let nd = &t.nodes[node as usize];
        let n = nd.n as usize;
        let base = (nd.base + bucket as u64 * n as u64) as usize;
        if (base as u64) < self.avg_end {
            let mut sum = 0.0;
            for a in 0..n {
                out[a] = f64::from_bits(self.avg[base + a].load(Relaxed)).max(0.0);
                sum += out[a];
            }
            if sum > 0.0 {
                for o in out[..n].iter_mut() {
                    *o /= sum;
                }
                return;
            }
        }
        regret_matching(&self.regret[base..base + n], &mut out[..n]);
    }

    /// Save a checkpoint (zstd-compressed), atomically replacing `path`.
    pub fn save(&self, path: &Path, threads: usize) -> std::io::Result<()> {
        let tmp = path.with_extension("tmp");
        {
            let f = BufWriter::with_capacity(1 << 24, std::fs::File::create(&tmp)?);
            let mut w = zstd::stream::Encoder::new(f, 1)?;
            w.multithread(threads.max(1) as u32)?;
            w.write_all(b"SIXCFR01")?;
            for x in [self.iterations, self.discounts, self.target, self.regret.len() as u64, self.avg.len() as u64, self.avg_streets as u64] {
                w.write_all(&x.to_le_bytes())?;
            }
            let mut buf = Vec::with_capacity(8 << 20);
            for c in self.regret.chunks(1 << 20) {
                buf.clear();
                for x in c {
                    buf.extend_from_slice(&x.load(Relaxed).to_le_bytes());
                }
                w.write_all(&buf)?;
            }
            for c in self.avg.chunks(1 << 20) {
                buf.clear();
                for x in c {
                    buf.extend_from_slice(&x.load(Relaxed).to_le_bytes());
                }
                w.write_all(&buf)?;
            }
            w.finish()?.flush()?;
        }
        std::fs::rename(&tmp, path)
    }

    pub fn load(t: &Tree, path: &Path) -> std::io::Result<Solver> {
        let f = BufReader::with_capacity(1 << 24, std::fs::File::open(path)?);
        let mut r = zstd::stream::Decoder::new(f)?;
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != b"SIXCFR01" {
            return Err(std::io::Error::other("not a solver checkpoint"));
        }
        let mut hdr = [0u64; 6];
        for h in hdr.iter_mut() {
            let mut u = [0u8; 8];
            r.read_exact(&mut u)?;
            *h = u64::from_le_bytes(u);
        }
        let [iterations, discounts, target, n_regret, n_avg, avg_streets] = hdr;
        if n_regret != t.n_entries {
            return Err(std::io::Error::other(format!("checkpoint has {n_regret} entries, tree has {}", t.n_entries)));
        }
        let mut s = Solver::new(t, avg_streets as usize);
        if n_avg != s.avg.len() as u64 {
            return Err(std::io::Error::other("checkpoint's average table doesn't match the tree"));
        }
        s.iterations = iterations;
        s.discounts = discounts;
        s.target = target;
        let mut buf = vec![0u8; 8 << 20];
        for c in s.regret.chunks_mut(1 << 20) {
            let bytes = &mut buf[..c.len() * 4];
            r.read_exact(bytes)?;
            for (x, b) in c.iter_mut().zip(bytes.chunks_exact(4)) {
                *x.get_mut() = i32::from_le_bytes(b.try_into().unwrap());
            }
        }
        for c in s.avg.chunks_mut(1 << 20) {
            let bytes = &mut buf[..c.len() * 8];
            r.read_exact(bytes)?;
            for (x, b) in c.iter_mut().zip(bytes.chunks_exact(8)) {
                *x.get_mut() = u64::from_le_bytes(b.try_into().unwrap());
            }
        }
        Ok(s)
    }
}
