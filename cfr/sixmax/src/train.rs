//! `sixmax train` / `sixmax export`: train one table size and stack depth
//! with MCCFR (resumable), then export its strategy for the engine.
//!
//!   sixmax train --run NAME --players 6|2 --stack CHIPS --abs ABS
//!                [--menu v1] [--budget 6000] [--calibrate 120] [--iters N]
//!                [--minutes M] [--seed 1] [--avg-streets 1]
//!       runs/NAME/<6p|2p>-s<stack>/solver.ckpt   (every 2 h + at the end)
//!       strategies/NAME/<6p|2p>-s<stack>/        (exported at the end)
//!
//! The run's length: `--iters`, or else enough iterations for `--budget`
//! regret updates per entry (the heads-up v1c level is about 6,000),
//! counting the updates per traversal measured over the last part of a
//! `--calibrate`-second warm-up (which is part of the run; it stops at 1%
//! of the run on small trees). `--minutes`
//! stops (and exports) early, for quick checks.
//!
//!   sixmax export --run NAME --players 6|2 --stack CHIPS --abs ABS [--menu v1]
//!       export from the latest checkpoint (e.g. while training continues)

use std::path::PathBuf;
use std::time::Instant;

use sixmax_cfr::abs::Abstraction;
use sixmax_cfr::deal::PokerDealer;
use sixmax_cfr::mccfr::{Config, Solver};
use sixmax_cfr::menu::Menu;
use sixmax_cfr::strategy::{Strategy, StrategyMeta};
use sixmax_cfr::tree::{self, Tree};

use crate::util::{flag, log, num, rss_gib, sixmax_home, threads};

pub struct Setup {
    pub run: String,
    pub abs_name: String,
    pub players: usize,
    pub stack: i32,
    pub menu: Menu,
    pub run_dir: PathBuf,
    pub out_dir: PathBuf,
}

pub fn depth_dir(players: usize, stack: i32) -> String {
    format!("{players}p-s{stack}")
}

pub fn setup(args: &[String]) -> Setup {
    let run = flag(args, "--run").expect("--run NAME");
    let abs_name = flag(args, "--abs").unwrap_or("g200".into());
    let players: usize = num(args, "--players", 6);
    let stack: i32 = num(args, "--stack", 200);
    let menu = Menu::by_name(&flag(args, "--menu").unwrap_or("v1".into())).expect("unknown menu");
    let d = depth_dir(players, stack);
    let run_dir = sixmax_home().join("runs").join(&run).join(&d);
    let out_dir = sixmax_home().join("strategies").join(&run).join(&d);
    Setup { run, abs_name, players, stack, menu, run_dir, out_dir }
}

fn export(st: &Setup, t: &Tree, solver: &Solver) {
    let t0 = Instant::now();
    let meta = StrategyMeta {
        players: st.players,
        stack: st.stack,
        menu: st.menu.name.into(),
        abstraction: st.abs_name.clone(),
        iterations: solver.iterations,
        entries: t.n_entries,
    };
    let s = Strategy::from_solver(meta, t, solver);
    s.save(&st.out_dir).expect("save strategy");
    log(format!("exported {} ({} entries) in {:.1}s", st.out_dir.display(), s.probs.len(), t0.elapsed().as_secs_f64()));
}

pub fn train_cmd(args: &[String]) {
    let st = setup(args);
    let seed: u64 = num(args, "--seed", 1);
    let budget: f64 = num(args, "--budget", 6000.0);
    let calibrate: f64 = num(args, "--calibrate", 120.0);
    let minutes: f64 = num(args, "--minutes", f64::INFINITY);
    let avg_streets: usize = num(args, "--avg-streets", 1);
    let nthreads = threads();
    std::fs::create_dir_all(&st.run_dir).expect("run dir");
    let abs = Abstraction::load(&sixmax_home().join("data").join(&st.abs_name)).expect("load abstraction");
    let t0 = Instant::now();
    let t = tree::build(st.players, st.stack, &st.menu, abs.buckets());
    log(format!(
        "run {} {} players {}bb, card groups {} {:?}, menu {}: decision nodes {:?}, {} entries ({:.2} GiB of regrets), tree built in {:.1}s, {nthreads} threads",
        st.run,
        st.players,
        st.stack / 2,
        st.abs_name,
        abs.buckets(),
        st.menu.name,
        t.street_counts(),
        t.n_entries,
        t.n_entries as f64 * 4.0 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    ));
    let dealer = PokerDealer { abs: &abs, players: st.players };
    let ckpt = st.run_dir.join("solver.ckpt");
    let mut solver = if ckpt.exists() {
        let s = Solver::load(&t, &ckpt).expect("load checkpoint");
        log(format!("resumed {} at {} of {} iterations", ckpt.display(), s.iterations, s.target));
        s
    } else {
        Solver::new(&t, avg_streets)
    };
    let started = Instant::now();
    if solver.target == 0 {
        if let Some(n) = flag(args, "--iters") {
            solver.target = n.parse::<f64>().expect("--iters") as u64;
        } else {
            // Warm up, measuring updates per traversal over the last quarter.
            let plain = Config::plain(nthreads);
            let step = (nthreads as u64 * 20_000).max(1);
            let mut last = (0u64, 0u64);
            let tc = Instant::now();
            while tc.elapsed().as_secs_f64() < calibrate {
                let c = solver.run(&t, &dealer, step, &plain, seed);
                if tc.elapsed().as_secs_f64() > 0.75 * calibrate {
                    last.0 += step;
                    last.1 += c.updates;
                }
                // Small trees: stop at 1% of the run, so the schedule
                // (discounting over the first 20%) isn't eaten by the warm-up.
                let upt = solver.counts.updates as f64 / (solver.iterations * st.players as u64) as f64;
                let provisional = budget * t.n_entries as f64 / (st.players as f64 * upt);
                if tc.elapsed().as_secs_f64() > 5.0 && solver.iterations as f64 > 0.01 * provisional {
                    break;
                }
            }
            if last.0 == 0 {
                last = (solver.iterations, solver.counts.updates);
            }
            let upt = last.1 as f64 / (last.0 * st.players as u64) as f64;
            let target = (budget * t.n_entries as f64 / (st.players as f64 * upt)).ceil() as u64;
            solver.target = target.max(solver.iterations);
            log(format!(
                "warm-up: {} iterations in {:.0}s, {:.1} updates per traversal -> {} iterations for {budget} updates per entry",
                solver.iterations,
                tc.elapsed().as_secs_f64(),
                upt,
                solver.target
            ));
        }
    }
    let target = solver.target;
    let cfg = Config::schedule(target, st.stack, nthreads);
    let mut last_ckpt = Instant::now();
    let mut stopped = false;
    let start_iters = solver.iterations;
    solver.train(&t, &dealer, target, (target / 100).max(1), &cfg, seed, |s, c, n| {
        let secs = started.elapsed().as_secs_f64();
        let rate = (s.iterations - start_iters) as f64 / secs.max(1e-9);
        let eta = (target - s.iterations) as f64 / rate.max(1.0);
        log(format!(
            "{:>5.1}%  {} iterations  {:.0}/s  {:.1} updates/traversal  elapsed {:.1} h  eta {:.1} h  discounts {}{}  rss {:.1} GiB",
            s.iterations as f64 / target as f64 * 100.0,
            s.iterations,
            rate,
            c.updates as f64 / (n * st.players as u64).max(1) as f64,
            secs / 3600.0,
            eta / 3600.0,
            s.discounts,
            if s.iterations >= cfg.prune_after { "  pruning" } else { "" },
            rss_gib()
        ));
        if last_ckpt.elapsed().as_secs() > 2 * 3600 {
            s.save(&ckpt, nthreads).expect("save checkpoint");
            log(format!("checkpoint {}", ckpt.display()));
            last_ckpt = Instant::now();
        }
        if secs > minutes * 60.0 {
            stopped = true;
            return false;
        }
        true
    });
    solver.save(&ckpt, nthreads).expect("save checkpoint");
    log(format!(
        "{} at {} of {} iterations after {:.2} h; checkpoint saved",
        if stopped { "stopped" } else { "trained" },
        solver.iterations,
        target,
        started.elapsed().as_secs_f64() / 3600.0
    ));
    export(&st, &t, &solver);
}

pub fn export_cmd(args: &[String]) {
    let st = setup(args);
    let abs = Abstraction::load(&sixmax_home().join("data").join(&st.abs_name)).expect("load abstraction");
    let t = tree::build(st.players, st.stack, &st.menu, abs.buckets());
    let solver = Solver::load(&t, &st.run_dir.join("solver.ckpt")).expect("load checkpoint");
    export(&st, &t, &solver);
}
