//! The training daemon: a strict-synchronous loop in one process, with the
//! models resident on the GPU (no cohort files, no worker protocol):
//!
//!   1. self-play a cohort with the current policy (tables persist across
//!      cohorts; see selfplay.rs)
//!   2. a PPO update of the policy and the training-only critic (train.rs)
//!   3. save `latest` (policy, critic, both AdamW states, counters) for resume
//!   4. on the snapshot schedule: save a snapshot, evaluate it, probe it, and
//!      maybe promote it to the anchor
//!   5. log one JSON line
//!
//!   poker train --run-dir ~/Workspace/6-max-poker-ai-engine/runs/run2
//!       [--cohort-rows 131072] [--max-cohort-rows 524288] [--fixed-batch]
//!       [--slots 4096] [--micro-batch 8192] [--minibatch 32768] [--epochs 2]
//!       [--lr 1e-4] [--critic-lr 3e-4] [--warmup 20] [--beta2 0.99]
//!       [--weight-decay 1e-4] [--c-entropy 0.01] [--c-anchor 0.05]
//!       [--clip 0.2] [--target-kl 0.02] [--cv all|fold|none] [--grad-clip 1.0]
//!       [--snapshot-every 4h] [--seed 42] [--max-cohorts 0]
//!
//! Anchor: the policy is pulled (forward KL) toward an anchor policy — at
//! first the initial weights, then the latest snapshot that significantly beat
//! the anchor head to head (the first snapshot replaces the initial anchor
//! once it significantly beats random).
//!
//! Batch size: the gradient noise scale of the policy term is measured every
//! cohort (EMA). Unless --fixed-batch, the cohort grows toward it (in
//! minibatch steps, between --cohort-rows and --max-cohort-rows): updates
//! stay mostly signal as the policy improves and its gradients get noisier.
//!
//! Snapshots: every --snapshot-every of training time (default 4 hours).
//!
//! Resumes from <run-dir>/latest.* if present.

use std::path::Path;
use std::time::Instant;

use serde_json::json;
use tch::nn::VarStore;
use tch::Device;

use crate::net::{n_params, NetCfg, PokerNet};
use crate::selfplay::SelfPlay;
use crate::train::{ppo_update, AdamW, Batch, ControlVariate, TrainCfg};
use crate::util::{flag, has_flag};

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

/// Noise-scale EMA weight of the newest cohort.
const NOISE_EMA: f64 = 0.05;
pub const INIT_ANCHOR: &str = "init";

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct Meta {
    elapsed_sec: f64,
    cohorts: u64,
    hands: u64,
    rows: u64,
    next_snapshot_at: f64,
    last_snapshot: Option<String>,
    /// INIT_ANCHOR or a snapshot name.
    anchor: String,
    cohort_rows: usize,
    noise_tr: f64,
    noise_g2: f64,
}

fn significant(r: Option<&(f64, f64)>) -> bool {
    r.is_some_and(|(bb, ci)| bb - ci > 0.0)
}

pub fn train(args: &[String]) {
    let run_dir = flag(args, "--run-dir").expect("--run-dir DIR");
    let min_rows: usize = num(args, "--cohort-rows", 131_072);
    let max_rows: usize = num(args, "--max-cohort-rows", 524_288);
    let adaptive = !has_flag(args, "--fixed-batch");
    let slots: usize = num(args, "--slots", 4096);
    let seed: u64 = num(args, "--seed", 42);
    let max_cohorts: u64 = num(args, "--max-cohorts", 0);
    let warmup: u64 = num(args, "--warmup", 20);
    let every = parse_duration(&flag(args, "--snapshot-every").unwrap_or("4h".into()));
    let cv = match flag(args, "--cv").as_deref().unwrap_or("all") {
        "all" => ControlVariate::All,
        "fold" => ControlVariate::Fold,
        "none" => ControlVariate::None,
        other => panic!("--cv all|fold|none, got {other}"),
    };
    let cfg = TrainCfg {
        lr: num(args, "--lr", 1e-4),
        critic_lr: num(args, "--critic-lr", 3e-4),
        weight_decay: num(args, "--weight-decay", 1e-4),
        beta2: num(args, "--beta2", 0.99),
        c_entropy: num(args, "--c-entropy", 0.01),
        c_anchor: num(args, "--c-anchor", 0.05),
        clip: num(args, "--clip", 0.2),
        target_kl: num(args, "--target-kl", 0.02),
        epochs: num(args, "--epochs", 2),
        minibatch: num(args, "--minibatch", 32_768),
        micro_batch: num(args, "--micro-batch", 8192),
        grad_clip: num(args, "--grad-clip", 1.0),
        cv,
    };

    let dir = Path::new(&run_dir);
    std::fs::create_dir_all(dir.join("snapshots")).expect("run dir");
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();

    let dev = if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu };
    println!("device={dev:?}");
    tch::manual_seed(seed as i64);
    let mut pvs = VarStore::new(dev);
    let policy = PokerNet::new(&pvs.root(), NetCfg::POLICY);
    let mut cvs = VarStore::new(dev);
    let critic = PokerNet::new(&cvs.root(), NetCfg::CRITIC);
    let mut avs = VarStore::new(dev);
    let anchor = PokerNet::new(&avs.root(), NetCfg::POLICY);
    let mut popt = AdamW::new(&pvs, 0.9, cfg.beta2);
    let mut copt = AdamW::new(&cvs, 0.9, cfg.beta2);
    let mut meta = Meta {
        next_snapshot_at: every,
        anchor: INIT_ANCHOR.into(),
        cohort_rows: min_rows,
        ..Default::default()
    };
    if Path::new(&p("latest.json")).exists() {
        meta = serde_json::from_str(&std::fs::read_to_string(p("latest.json")).unwrap()).expect("latest.json");
        pvs.load(p("latest.safetensors")).expect("load latest policy");
        cvs.load(p("latest_critic.safetensors")).expect("load latest critic");
        popt.load(&p("latest_opt.safetensors"), dev);
        copt.load(&p("latest_critic_opt.safetensors"), dev);
        println!(
            "resumed {} (elapsed={:.2}h cohorts={} hands={} anchor={})",
            run_dir,
            meta.elapsed_sec / 3600.0,
            meta.cohorts,
            meta.hands,
            meta.anchor
        );
    } else {
        pvs.save(p("anchor.safetensors")).expect("save initial anchor");
        println!("cold start (seed={seed})");
    }
    avs.load(p("anchor.safetensors")).expect("load anchor");
    println!(
        "policy: {} params; critic: {} params; cohort_rows={} slots={slots} lr={} cv={:?}",
        n_params(&pvs),
        n_params(&cvs),
        meta.cohort_rows,
        cfg.lr,
        cfg.cv
    );

    let mut sp = SelfPlay::new(slots, seed.wrapping_add(meta.cohorts.wrapping_mul(7919)));
    let start = Instant::now();
    let base_elapsed = meta.elapsed_sec;
    loop {
        let t0 = Instant::now();
        let cohort = sp.run_cohort(&policy, dev, meta.cohort_rows);
        let sp_sec = t0.elapsed().as_secs_f64();

        let t1 = Instant::now();
        let batch = Batch::from_rows(&cohort.rows, dev);
        let z_mean = cohort.rows.iter().map(|r| r.z as f64).sum::<f64>() / cohort.rows.len() as f64;
        let lr = cfg.lr * ((meta.cohorts + 1) as f64 / warmup.max(1) as f64).min(1.0);
        let st = ppo_update(&policy, &pvs, &mut popt, &critic, &cvs, &mut copt, &anchor, &batch, &cfg, lr, (seed + meta.cohorts) as i64, meta.cohorts % 4 == 0);
        let tr_sec = t1.elapsed().as_secs_f64();
        drop(batch);

        // Gradient noise scale (EMA of numerator and denominator separately).
        if st.noise_tr != 0.0 && st.noise_g2.is_finite() && st.noise_tr.is_finite() {
            if meta.noise_tr == 0.0 {
                meta.noise_tr = st.noise_tr;
                meta.noise_g2 = st.noise_g2;
            } else {
                meta.noise_tr += NOISE_EMA * (st.noise_tr - meta.noise_tr);
                meta.noise_g2 += NOISE_EMA * (st.noise_g2 - meta.noise_g2);
            }
        }
        let b_noise = if meta.noise_g2 > 0.0 { meta.noise_tr / meta.noise_g2 } else { f64::INFINITY };
        if adaptive && meta.cohorts >= 20 {
            let target = b_noise.min(max_rows as f64).max(min_rows as f64);
            let steps = (target / cfg.minibatch as f64).ceil() as usize;
            meta.cohort_rows = (steps * cfg.minibatch).clamp(min_rows, max_rows);
        }

        meta.cohorts += 1;
        meta.hands += cohort.hud.hands;
        meta.rows += cohort.rows.len() as u64;
        meta.elapsed_sec = base_elapsed + start.elapsed().as_secs_f64();

        // Atomic-ish latest checkpoint: write temps, then rename.
        pvs.save(p("latest.tmp.safetensors")).expect("save policy");
        cvs.save(p("latest_critic.tmp.safetensors")).expect("save critic");
        popt.save(&p("latest_opt.tmp.safetensors"));
        copt.save(&p("latest_critic_opt.tmp.safetensors"));
        let mut snapshot = None;
        if meta.elapsed_sec >= meta.next_snapshot_at {
            meta.next_snapshot_at = meta.elapsed_sec + every;
            let name = format!("snap_h{:05}_c{:07}.safetensors", (meta.elapsed_sec / 3600.0) as u64, meta.cohorts);
            let path = dir.join("snapshots").join(&name);
            pvs.save(&path).expect("save snapshot");
            snapshot = Some(path);
        }
        std::fs::write(p("latest.tmp.json"), serde_json::to_string(&meta).unwrap()).unwrap();
        for (tmp, f) in [
            ("latest.tmp.safetensors", "latest.safetensors"),
            ("latest_critic.tmp.safetensors", "latest_critic.safetensors"),
            ("latest_opt.tmp.safetensors", "latest_opt.safetensors"),
            ("latest_critic_opt.tmp.safetensors", "latest_critic_opt.safetensors"),
            ("latest.tmp.json", "latest.json"),
        ] {
            std::fs::rename(p(tmp), p(f)).expect("rename checkpoint");
        }

        let r4 = |x: f64| (x * 1e4).round() / 1e4;
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
                "z_mean": r4(z_mean),
                "policy": r4(st.policy),
                "entropy": r4(st.entropy),
                "anchor_kl": r4(st.anchor_kl),
                "kl": r4(st.kl),
                "clip_frac": r4(st.clip_frac),
                "epochs": st.epochs,
                "steps": st.steps,
                "grad_norm": (st.grad_norm * 1e3).round() / 1e3,
                "critic_loss": r4(st.critic_loss),
                "critic_ev": r4(st.critic_ev),
                "b_noise": if b_noise.is_finite() { b_noise.round() } else { -1.0 },
                "next_rows": meta.cohort_rows,
                "hud": cohort.hud.to_json(),
            })
        );
        if let Some(path) = snapshot {
            println!("[snapshot] {}", path.display());
            let name = path.file_stem().unwrap().to_string_lossy().into_owned();
            println!("[probe] {name} {}", crate::probe::run(&policy, dev));
            let mut must: Vec<String> = meta.last_snapshot.iter().cloned().collect();
            if meta.anchor != INIT_ANCHOR && !must.contains(&meta.anchor) {
                must.push(meta.anchor.clone());
            }
            let res = crate::eval::after_snapshot(&run_dir, &path, &must, args);
            if !res.is_empty() {
                // Anchor: replace it when the candidate significantly beats it.
                let beats = if meta.anchor == INIT_ANCHOR {
                    significant(res.get(crate::eval::RANDOM))
                } else {
                    significant(res.get(&meta.anchor))
                };
                if beats {
                    std::fs::copy(&path, p("anchor.safetensors")).expect("copy anchor");
                    avs.load(p("anchor.safetensors")).expect("load anchor");
                    println!("[anchor] {} -> {name}", meta.anchor);
                    meta.anchor = name.clone();
                } else {
                    println!("[anchor] stays {}", meta.anchor);
                }
            }
            meta.last_snapshot = Some(name);
            std::fs::write(p("latest.tmp.json"), serde_json::to_string(&meta).unwrap()).unwrap();
            std::fs::rename(p("latest.tmp.json"), p("latest.json")).expect("rename meta");
        }

        if max_cohorts > 0 && meta.cohorts >= max_cohorts {
            println!("reached --max-cohorts; stopping");
            break;
        }
    }
}
