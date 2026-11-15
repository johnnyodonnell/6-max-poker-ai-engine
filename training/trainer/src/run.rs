//! The training daemon: a strict-synchronous loop in one process, with the
//! model resident on the GPU (no cohort files, no worker protocol):
//!
//!   1. self-play a cohort with the current weights (on-policy)
//!   2. one REINFORCE step over it
//!   3. save `latest` (weights + AdamW moments + counters) for resume
//!   4. on the snapshot schedule: save a snapshot (and evaluate it)
//!   5. log one JSON line
//!
//!   poker train --run-dir ~/Workspace/6-max-poker-ai-engine/runs/run1
//!       [--cohort-rows 131072] [--slots 4096] [--micro-batch 8192]
//!       [--lr 3e-4] [--warmup 100] [--weight-decay 1e-4] [--c-value 1.0] [--c-entropy 0.01]
//!       [--c-sigma 0.001] [--grad-clip 1.0] [--snapshot-every DURATION]
//!       [--seed 42] [--max-cohorts 0]
//!
//! Snapshot schedule: every 30 minutes at first. After two snapshots in a row
//! show good improvement — each beats the previous snapshot by a significant
//! margin (bb/100 minus its 95% CI still > 0) — the interval steps up to 1
//! hour, then likewise to 4 hours. `--snapshot-every` fixes the interval
//! instead (for tests).
//!
//! Resumes from <run-dir>/latest.* if present.

use std::path::Path;
use std::time::Instant;

use serde_json::json;
use tch::nn::VarStore;
use tch::Device;

use crate::net::{n_params, PokerNet};
use crate::selfplay::run_cohort;
use crate::train::{reinforce_step, AdamW, Batch, TrainCfg};
use crate::util::flag;

fn parse_duration(s: &str) -> f64 {
    let (num, mult) = match s.chars().last() {
        Some('h') => (&s[..s.len() - 1], 3600.0),
        Some('m') => (&s[..s.len() - 1], 60.0),
        Some('s') => (&s[..s.len() - 1], 1.0),
        _ => (s, 1.0),
    };
    num.parse::<f64>().expect("duration") * mult
}

fn num<T: std::str::FromStr>(args: &[String], key: &str, default: T) -> T
where
    T::Err: std::fmt::Debug,
{
    flag(args, key).map(|v| v.parse().unwrap()).unwrap_or(default)
}

/// Snapshot intervals, in seconds: 30m -> 1h -> 4h.
const SCHEDULE: [f64; 3] = [1800.0, 3600.0, 14400.0];
/// Consecutive significant improvements needed to step up the interval.
const STEP_UP_AFTER: u32 = 2;

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct Meta {
    elapsed_sec: f64,
    cohorts: u64,
    hands: u64,
    rows: u64,
    next_snapshot_at: f64,
    #[serde(default)]
    interval_idx: usize,
    /// Consecutive snapshots that significantly beat their predecessor.
    #[serde(default)]
    improve_streak: u32,
    #[serde(default)]
    last_snapshot: Option<String>,
}

pub fn train(args: &[String]) {
    let run_dir = flag(args, "--run-dir").expect("--run-dir DIR");
    let cohort_rows: usize = num(args, "--cohort-rows", 131_072);
    let slots: usize = num(args, "--slots", 4096);
    let seed: u64 = num(args, "--seed", 42);
    let max_cohorts: u64 = num(args, "--max-cohorts", 0);
    // Linear LR warmup over the first cohorts: Adam's first steps move every
    // weight by ~lr, which can swing the (unbounded) value head wildly.
    let warmup: u64 = num(args, "--warmup", 100);
    let fixed_every = flag(args, "--snapshot-every").map(|s| parse_duration(&s));
    // Shrinks the whole schedule (testing only: e.g. 0.01 -> 18s / 36s / 144s).
    let scale: f64 = num(args, "--schedule-scale", 1.0);
    let interval = |m: &Meta| fixed_every.unwrap_or(SCHEDULE[m.interval_idx.min(SCHEDULE.len() - 1)] * scale);
    let cfg = TrainCfg {
        lr: num(args, "--lr", 3e-4),
        weight_decay: num(args, "--weight-decay", 1e-4),
        c_value: num(args, "--c-value", 1.0),
        c_entropy: num(args, "--c-entropy", 0.01),
        c_sigma: num(args, "--c-sigma", 0.001),
        grad_clip: num(args, "--grad-clip", 1.0),
        micro_batch: num(args, "--micro-batch", 8192),
    };

    let dir = Path::new(&run_dir);
    std::fs::create_dir_all(dir.join("snapshots")).expect("run dir");
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();

    let dev = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    println!("device={dev:?}");
    let mut vs = VarStore::new(dev);
    tch::manual_seed(seed as i64);
    let net = PokerNet::new(&vs.root());
    let mut opt = AdamW::new(&vs);
    let mut meta = Meta { next_snapshot_at: interval(&Meta::default()), ..Default::default() };
    if Path::new(&p("latest.json")).exists() {
        meta = serde_json::from_str(&std::fs::read_to_string(p("latest.json")).unwrap()).expect("latest.json");
        vs.load(p("latest.safetensors")).expect("load latest weights");
        opt.load(&p("latest_opt.safetensors"), dev);
        println!(
            "resumed {} (elapsed={:.2}h cohorts={} hands={})",
            run_dir,
            meta.elapsed_sec / 3600.0,
            meta.cohorts,
            meta.hands
        );
    } else {
        println!("cold start (seed={seed})");
    }
    println!("net: {} params; cohort_rows={cohort_rows} slots={slots} lr={}", n_params(&vs), cfg.lr);

    let start = Instant::now();
    let base_elapsed = meta.elapsed_sec;
    loop {
        let t0 = Instant::now();
        let cohort_seed = seed.wrapping_add(meta.cohorts.wrapping_mul(7919)).wrapping_add(1);
        let cohort = run_cohort(&net, dev, slots, cohort_rows, cohort_seed);
        let sp_sec = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let batch = Batch::from_rows(&cohort.rows, dev);
        let z_mean = cohort.rows.iter().map(|r| r.z as f64).sum::<f64>() / cohort.rows.len() as f64;
        let lr = cfg.lr * ((meta.cohorts + 1) as f64 / warmup.max(1) as f64).min(1.0);
        let stats = reinforce_step(&net, &vs, &mut opt, &batch, &cfg, lr);
        let tr_sec = t1.elapsed().as_secs_f64();
        drop(batch);

        meta.cohorts += 1;
        meta.hands += cohort.hud.hands;
        meta.rows += cohort.rows.len() as u64;
        meta.elapsed_sec = base_elapsed + start.elapsed().as_secs_f64();

        // Atomic-ish latest checkpoint: write temps, then rename.
        vs.save(p("latest.tmp.safetensors")).expect("save weights");
        opt.save(&p("latest_opt.tmp.safetensors"));
        let mut snapshot = None;
        if meta.elapsed_sec >= meta.next_snapshot_at {
            meta.next_snapshot_at = meta.elapsed_sec + interval(&meta);
            let name = format!("snap_h{:05}_c{:07}.safetensors", (meta.elapsed_sec / 3600.0) as u64, meta.cohorts);
            let path = dir.join("snapshots").join(&name);
            vs.save(&path).expect("save snapshot");
            snapshot = Some(path);
        }
        std::fs::write(p("latest.tmp.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        for (tmp, f) in [
            ("latest.tmp.safetensors", "latest.safetensors"),
            ("latest_opt.tmp.safetensors", "latest_opt.safetensors"),
            ("latest.tmp.json", "latest.json"),
        ] {
            std::fs::rename(p(tmp), p(f)).expect("rename checkpoint");
        }

        println!(
            "{}",
            json!({
                "t": (meta.elapsed_sec * 10.0).round() / 10.0,
                "cohorts": meta.cohorts,
                "hands": meta.hands,
                "rows": cohort.rows.len(),
                "lr": lr,
                "forwards": cohort.forwards,
                "sp_sec": (sp_sec * 100.0).round() / 100.0,
                "tr_sec": (tr_sec * 100.0).round() / 100.0,
                "hands_per_s": (cohort.hud.hands as f64 / sp_sec).round(),
                "z_mean": (z_mean * 1e4).round() / 1e4,
                "loss": (stats.loss * 1e4).round() / 1e4,
                "policy": (stats.policy * 1e4).round() / 1e4,
                "value": (stats.value * 1e4).round() / 1e4,
                "entropy": (stats.entropy * 1e4).round() / 1e4,
                "sigma": (stats.sigma * 1e3).round() / 1e3,
                "grad_norm": (stats.grad_norm * 1e3).round() / 1e3,
                "logp_gap": stats.logp_gap,
                "kl": stats.kl,
                "hud": cohort.hud.to_json(),
            })
        );
        if let Some(path) = snapshot {
            println!("[snapshot] {}", path.display());
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            let vs_prev = crate::eval::after_snapshot(&run_dir, &path, meta.last_snapshot.as_deref(), args);
            meta.last_snapshot = Some(name);
            if let Some((bb, ci)) = vs_prev {
                let improved = bb - ci > 0.0;
                meta.improve_streak = if improved { meta.improve_streak + 1 } else { 0 };
                println!(
                    "[schedule] vs previous: {bb:+.1} ± {ci:.1} bb/100 ({}); streak {}",
                    if improved { "improved" } else { "no significant improvement" },
                    meta.improve_streak
                );
                if fixed_every.is_none() && meta.improve_streak >= STEP_UP_AFTER && meta.interval_idx + 1 < SCHEDULE.len() {
                    let snap_at = meta.next_snapshot_at - interval(&meta);
                    meta.interval_idx += 1;
                    meta.improve_streak = 0;
                    meta.next_snapshot_at = snap_at + interval(&meta);
                    println!("[schedule] snapshots now every {:.0} minutes", interval(&meta) / 60.0);
                }
            }
            std::fs::write(p("latest.tmp.json"), serde_json::to_string(&meta).unwrap()).unwrap();
            std::fs::rename(p("latest.tmp.json"), p("latest.json")).expect("rename meta");
        }

        if max_cohorts > 0 && meta.cohorts >= max_cohorts {
            println!("reached --max-cohorts; stopping");
            break;
        }
    }
}
