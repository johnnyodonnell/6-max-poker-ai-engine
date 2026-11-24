//! Exported strategies: what the bot plays, one byte per (information set,
//! action), laid out like the solver's entries. Each information set's bytes
//! sum to 255 (the remainder goes to its likeliest action).
//!
//! Files in a strategy directory: `strategy.u8` and `meta.json`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::mccfr::Solver;
use crate::tree::{Tree, MAX_ACTIONS, TERMINAL};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrategyMeta {
    pub players: usize,
    pub stack: i32,
    pub menu: String,
    pub abstraction: String,
    pub iterations: u64,
    pub entries: u64,
}

pub struct Strategy {
    pub meta: StrategyMeta,
    pub probs: Vec<u8>,
}

/// Probabilities as bytes summing to 255 (the remainder to the likeliest).
pub fn quantize(p: &[f64], out: &mut [u8]) {
    let mut sum = 0u32;
    let mut best = 0;
    for (i, &x) in p.iter().enumerate() {
        out[i] = (x.max(0.0) * 255.0).floor().min(255.0) as u8;
        sum += out[i] as u32;
        if x > p[best] {
            best = i;
        }
    }
    out[best] = (out[best] as u32 + 255u32.saturating_sub(sum)).min(255) as u8;
}

impl Strategy {
    pub fn from_solver(meta: StrategyMeta, t: &Tree, solver: &Solver) -> Strategy {
        let mut probs = vec![0u8; t.n_entries as usize];
        let mut p = [0f64; MAX_ACTIONS];
        let mut q = [0u8; MAX_ACTIONS];
        for (id, nd) in t.nodes.iter().enumerate() {
            if nd.player == TERMINAL {
                continue;
            }
            let n = nd.n as usize;
            for b in 0..t.buckets[nd.street as usize] {
                solver.strategy(t, id as u32, b, &mut p);
                quantize(&p[..n], &mut q[..n]);
                let base = (nd.base + b as u64 * n as u64) as usize;
                probs[base..base + n].copy_from_slice(&q[..n]);
            }
        }
        Strategy { meta, probs }
    }

    /// The action probabilities at (node, group).
    #[inline]
    pub fn probs(&self, t: &Tree, node: u32, bucket: u32, out: &mut [f64]) {
        let nd = &t.nodes[node as usize];
        let n = nd.n as usize;
        let base = (nd.base + bucket as u64 * n as u64) as usize;
        let w = &self.probs[base..base + n];
        let tot: u32 = w.iter().map(|&x| x as u32).sum();
        for a in 0..n {
            out[a] = if tot > 0 { w[a] as f64 / tot as f64 } else { 1.0 / n as f64 };
        }
    }

    pub fn save(&self, dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(dir)?;
        let tmp = dir.join("strategy.u8.tmp");
        std::fs::write(&tmp, &self.probs)?;
        std::fs::rename(&tmp, dir.join("strategy.u8"))?;
        std::fs::write(dir.join("meta.json"), serde_json::to_string_pretty(&self.meta).unwrap())
    }

    pub fn load(dir: &Path) -> std::io::Result<Strategy> {
        let meta: StrategyMeta = serde_json::from_str(&std::fs::read_to_string(dir.join("meta.json"))?)
            .map_err(|e| std::io::Error::other(format!("meta.json: {e}")))?;
        let probs = std::fs::read(dir.join("strategy.u8"))?;
        if probs.len() as u64 != meta.entries {
            return Err(std::io::Error::other("strategy.u8 doesn't match meta.json"));
        }
        Ok(Strategy { meta, probs })
    }
}
