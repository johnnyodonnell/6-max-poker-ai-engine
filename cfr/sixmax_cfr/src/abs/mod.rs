//! Card abstraction: which bucket a hand falls in on each street.
//!
//! - Preflop: the 169 classes (lossless).
//! - Flop and turn: k-means (earth mover's distance) on the distribution of
//!   the hand's final river equity over all runouts ("distribution-aware",
//!   Johanson et al. 2013): a 32-bin cumulative histogram as u8.
//! - River: k-means on OCHS, the hand's equity against 8 clusters of
//!   opponent preflop hands.
//!
//! Each postflop street is a dense u16 table indexed board-major (see
//! `index`), built by fitting centers on a sample of boards and then
//! assigning every hand on every board. Files: `<dir>/{flop,turn,river}.u16`
//! plus `meta.json`.

pub mod equity;
pub mod index;
pub mod kmeans;

use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use poker_core::cards::Card;

use crate::classes::{class_hands, class_of, combos, N_CLASSES};
use crate::deal::{deal_cards, showdown};
use equity::{river_ehs, river_ochs, Scratch, N_OCHS};
use index::{all_sets, index, n_sets, N_HOLES};

pub const HIST_BINS: usize = 32;
const NO_BUCKET: u16 = u16::MAX;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Meta {
    pub name: String,
    /// Buckets per street: [169, flop, turn, river].
    pub buckets: [u32; 4],
    /// Preflop class -> opponent cluster used by the river OCHS feature.
    pub ochs_cluster: Vec<u8>,
    pub fit_boards: [usize; 3],
    pub seed: u64,
}

pub struct Abstraction {
    pub meta: Meta,
    pub flop: Vec<u16>,
    pub turn: Vec<u16>,
    pub river: Vec<u16>,
}

impl Abstraction {
    pub fn buckets(&self) -> [u32; 4] {
        self.meta.buckets
    }

    /// Bucket of `hole` on `street` (0..4) given the board so far (at least
    /// 3/4/5 cards for flop/turn/river).
    #[inline]
    pub fn bucket(&self, street: usize, hole: [Card; 2], board: &[Card]) -> u32 {
        let b = match street {
            0 => return class_of(hole[0], hole[1]),
            1 => self.flop[index(hole, &board[..3])],
            2 => self.turn[index(hole, &board[..4])],
            _ => self.river[index(hole, &board[..5])],
        };
        debug_assert_ne!(b, NO_BUCKET, "hole cards on the board");
        b as u32
    }

    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        for (name, t) in [("flop", &self.flop), ("turn", &self.turn), ("river", &self.river)] {
            let tmp = dir.join(format!("{name}.u16.tmp"));
            let mut w = BufWriter::with_capacity(1 << 24, std::fs::File::create(&tmp)?);
            for c in t.chunks(1 << 20) {
                let bytes: Vec<u8> = c.iter().flat_map(|x| x.to_le_bytes()).collect();
                w.write_all(&bytes)?;
            }
            w.flush()?;
            drop(w);
            std::fs::rename(&tmp, dir.join(format!("{name}.u16")))?;
        }
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&self.meta).unwrap())
    }

    pub fn load(dir: &Path) -> std::io::Result<Abstraction> {
        let meta: Meta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)
            .map_err(|e| std::io::Error::other(format!("meta.json: {e}")))?;
        let read = |name: &str, n: usize| -> std::io::Result<Vec<u16>> {
            let mut r = BufReader::with_capacity(1 << 24, std::fs::File::open(dir.join(format!("{name}.u16")))?);
            let mut out = vec![0u16; n];
            let mut buf = vec![0u8; 2 << 20];
            for c in out.chunks_mut(1 << 20) {
                let bytes = &mut buf[..c.len() * 2];
                r.read_exact(bytes)?;
                for (x, b) in c.iter_mut().zip(bytes.chunks_exact(2)) {
                    *x = u16::from_le_bytes([b[0], b[1]]);
                }
            }
            Ok(out)
        };
        Ok(Abstraction {
            flop: read("flop", n_sets(3) * N_HOLES)?,
            turn: read("turn", n_sets(4) * N_HOLES)?,
            river: read("river", n_sets(5) * N_HOLES)?,
            meta,
        })
    }
}

// ---------------------------------------------------------------------------
// Building.

/// Cluster the 169 preflop classes into the 8 OCHS opponent groups: sort by
/// equity against a random hand and cut into groups of ~equal combo counts.
fn ochs_clusters(seed: u64) -> Vec<u8> {
    let hands = class_hands();
    let eq: Vec<f64> = (0..N_CLASSES)
        .into_par_iter()
        .map(|c| {
            let mut rng = SmallRng::seed_from_u64(seed ^ (c as u64 + 1) * 0x9E37_79B9);
            let n = 40_000;
            let mut won = 0.0;
            let mut cards = [0u8; 9];
            for _ in 0..n {
                let (a, b) = hands[c][rng.gen_range(0..hands[c].len())];
                // Opponent hand and board from the rest of the deck.
                loop {
                    deal_cards(&mut rng, &mut cards[..7]);
                    if !cards[..7].contains(&a) && !cards[..7].contains(&b) {
                        break;
                    }
                }
                won += match showdown([a, b], [cards[0], cards[1]], &cards[2..7]) {
                    1 => 1.0,
                    0 => 0.5,
                    _ => 0.0,
                };
            }
            won / n as f64
        })
        .collect();
    let mut order: Vec<usize> = (0..N_CLASSES).collect();
    order.sort_by(|&x, &y| eq[x].partial_cmp(&eq[y]).unwrap());
    let mut out = vec![0u8; N_CLASSES];
    let mut acc = 0u32;
    for &c in &order {
        out[c] = ((acc as usize * N_OCHS) / 1326).min(N_OCHS - 1) as u8;
        acc += combos(c as u32);
    }
    out
}

fn cdf(hist: &[u16; HIST_BINS], total: u32) -> [u8; HIST_BINS] {
    let mut out = [0u8; HIST_BINS];
    let mut acc = 0u32;
    for i in 0..HIST_BINS {
        acc += hist[i] as u32;
        out[i] = ((acc as f64 * 255.0) / total as f64).round() as u8;
    }
    out
}

#[inline]
fn bin(e: f32) -> usize {
    ((e * HIST_BINS as f32) as usize).min(HIST_BINS - 1)
}

/// Per-thread buffers for feature computation.
struct Work {
    sc: Scratch,
    ehs: Box<[f32; N_HOLES]>,
    ochs: Box<[[f32; N_OCHS]; N_HOLES]>,
    hist: Vec<[u16; HIST_BINS]>,
}

impl Work {
    fn new() -> Work {
        Work {
            sc: Scratch::default(),
            ehs: Box::new([0.0; N_HOLES]),
            ochs: Box::new([[0.0; N_OCHS]; N_HOLES]),
            hist: vec![[0; HIST_BINS]; N_HOLES],
        }
    }
}

/// The live hands (hole index) given `dead` cards.
fn live_hands(dead: u64) -> impl Iterator<Item = (u16, Card, Card)> {
    (1..52u8).flat_map(move |b| {
        (0..b).filter_map(move |a| {
            if dead & (1 << a) != 0 || dead & (1 << b) != 0 {
                None
            } else {
                Some((index::hole_index(a, b) as u16, a, b))
            }
        })
    })
}

fn mask(cards: &[Card]) -> u64 {
    cards.iter().fold(0, |m, &c| m | 1 << c)
}

/// Distribution-aware features of every hand on a flop or turn board: the
/// cumulative histogram of its river EHS over every runout.
fn runout_features(board: &[Card], w: &mut Work, out: &mut Vec<(u16, [u8; HIST_BINS])>) {
    let dead = mask(board);
    let live: Vec<u16> = live_hands(dead).map(|x| x.0).collect();
    for &h in &live {
        w.hist[h as usize] = [0; HIST_BINS];
    }
    let mut b5 = [0u8; 5];
    b5[..board.len()].copy_from_slice(board);
    {
        let Work { sc, ehs, hist, .. } = &mut *w;
        let mut add = |b5: &[Card; 5]| {
            river_ehs(b5, sc, &mut **ehs);
            // The sweep scored exactly the hands live on this river board.
            for &(_, h, _, _) in sc.hands.iter() {
                hist[h as usize][bin(ehs[h as usize])] += 1;
            }
        };
        if board.len() == 4 {
            for r in 0..52u8 {
                if dead & (1 << r) == 0 {
                    b5[4] = r;
                    add(&b5);
                }
            }
        } else {
            for t in 0..52u8 {
                if dead & (1 << t) != 0 {
                    continue;
                }
                for r in (t + 1)..52u8 {
                    if dead & (1 << r) != 0 {
                        continue;
                    }
                    b5[3] = t;
                    b5[4] = r;
                    add(&b5);
                }
            }
        }
    }
    out.clear();
    for &h in &live {
        let hist = &w.hist[h as usize];
        let total: u32 = hist.iter().map(|&x| x as u32).sum();
        out.push((h, cdf(hist, total)));
    }
}

fn river_features(board: &[Card; 5], cluster: &[u8; 169], w: &mut Work, out: &mut Vec<(u16, [u8; N_OCHS])>) {
    river_ochs(board, cluster, &mut w.sc, &mut *w.ochs);
    out.clear();
    for &(_, h, _, _) in w.sc.hands.iter() {
        let o = &w.ochs[h as usize];
        out.push((h, std::array::from_fn(|k| (o[k] * 255.0).round() as u8)));
    }
}

/// Fit centers for one street on `n_boards` sampled boards, then assign every
/// hand on every board; returns the table (ordered weakest bucket first).
fn build_street<const K: usize, const D: usize>(
    k: usize,
    n_boards: usize,
    seed: u64,
    features: &(dyn Fn(&[Card; K], &mut Work, &mut Vec<(u16, [u8; D])>) + Sync),
    strength: fn(&[u8; D]) -> f64,
    log: &(dyn Fn(String) + Sync),
) -> Vec<u16> {
    let boards = all_sets::<K>();
    let mut rng = SmallRng::seed_from_u64(seed);
    let sample: Vec<usize> = (0..n_boards.min(boards.len())).map(|_| rng.gen_range(0..boards.len())).collect();
    let t = std::time::Instant::now();
    let points: Vec<[u8; D]> = sample
        .par_iter()
        .map_init(
            || (Work::new(), Vec::new()),
            |(w, out), &i| {
                features(&boards[i], w, out);
                out.iter().map(|x| x.1).collect::<Vec<_>>()
            },
        )
        .flatten()
        .collect();
    log(format!("  {} sample points from {} boards in {:.1}s", points.len(), sample.len(), t.elapsed().as_secs_f64()));

    let t = std::time::Instant::now();
    let mut centers = kmeans::fit(&points, k, 30, seed ^ 0xC0FFEE, log);
    // Order buckets weakest first (nicer ids for debugging).
    centers.sort_by(|a, b| strength(a).partial_cmp(&strength(b)).unwrap());
    log(format!("  fitted {k} centers in {:.1}s", t.elapsed().as_secs_f64()));
    drop(points);

    let t = std::time::Instant::now();
    let mut table = vec![NO_BUCKET; boards.len() * N_HOLES];
    table.par_chunks_mut(N_HOLES).zip(boards.par_iter()).for_each_init(
        || (Work::new(), Vec::new()),
        |(w, out), (chunk, board)| {
            features(board, w, out);
            for (h, f) in out.iter() {
                chunk[*h as usize] = kmeans::nearest(&centers, f).0 as u16;
            }
        },
    );
    log(format!("  assigned {} boards in {:.1}s", boards.len(), t.elapsed().as_secs_f64()));
    table
}

/// Strength of a cumulative-histogram feature: less mass in the low bins is
/// stronger.
fn cdf_strength<const D: usize>(f: &[u8; D]) -> f64 {
    -(f.iter().map(|&x| x as f64).sum::<f64>())
}

fn ochs_strength<const D: usize>(f: &[u8; D]) -> f64 {
    f.iter().map(|&x| x as f64).sum::<f64>()
}

/// Build an abstraction with [169, flop, turn, river] buckets, fitting each
/// street's centers on `fit_boards` sampled boards.
pub fn build(name: &str, buckets: [u32; 4], fit_boards: [usize; 3], seed: u64, log: &(dyn Fn(String) + Sync)) -> Abstraction {
    assert_eq!(buckets[0], N_CLASSES as u32, "preflop is always the 169 classes");
    for &b in &buckets[1..] {
        assert!(b >= 1 && b < NO_BUCKET as u32, "bucket count {b}");
    }
    let t = std::time::Instant::now();
    let ochs_cluster = ochs_clusters(seed);
    log(format!("OCHS opponent clusters in {:.1}s", t.elapsed().as_secs_f64()));
    let cl: [u8; 169] = ochs_cluster.clone().try_into().unwrap();

    log(format!("flop: {} buckets", buckets[1]));
    let flop = build_street::<3, HIST_BINS>(
        buckets[1] as usize,
        fit_boards[0],
        seed ^ 1,
        &|b, w, out| runout_features(b, w, out),
        cdf_strength::<HIST_BINS>,
        log,
    );
    log(format!("turn: {} buckets", buckets[2]));
    let turn = build_street::<4, HIST_BINS>(
        buckets[2] as usize,
        fit_boards[1],
        seed ^ 2,
        &|b, w, out| runout_features(b, w, out),
        cdf_strength::<HIST_BINS>,
        log,
    );
    log(format!("river: {} buckets", buckets[3]));
    let river = build_street::<5, N_OCHS>(
        buckets[3] as usize,
        fit_boards[2],
        seed ^ 3,
        &|b, w, out| river_features(b, &cl, w, out),
        ochs_strength::<N_OCHS>,
        log,
    );
    Abstraction {
        meta: Meta { name: name.into(), buckets, ochs_cluster, fit_boards, seed },
        flop,
        turn,
        river,
    }
}
