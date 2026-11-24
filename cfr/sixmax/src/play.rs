//! `sixmax eval` and `sixmax probe`: quick checks of exported strategies on
//! the abstract game.
//!
//!   sixmax eval --players 6|2 --stack CHIPS [--abs g200] [--deals 2e6]
//!               --strat NAME=RUN ... (--match X,Y ... | --vs-bots NAME)
//!       X in one seat against Y in the others, every seat in turn on each
//!       deal. X and Y are --strat names or bots (random, station, maniac,
//!       minraiser, shover, nit).
//!   sixmax probe --run RUN --players 6|2 --stack CHIPS [--abs g200]
//!       opening charts: raise % per starting hand

use sixmax_cfr::abs::Abstraction;
use sixmax_cfr::classes::class_name;
use sixmax_cfr::deal::PokerDealer;
use sixmax_cfr::eval::{play_match, Player, BOTS};
use sixmax_cfr::menu::Menu;
use sixmax_cfr::strategy::Strategy;
use sixmax_cfr::tree::{self, position_name, Act, Tree, MAX_ACTIONS};

use crate::train::depth_dir;
use crate::util::{flag, log, num, sixmax_home, threads};

fn values(args: &[String], name: &str) -> Vec<String> {
    args.windows(2).filter(|w| w[0] == name).map(|w| w[1].clone()).collect()
}

fn load(run: &str, players: usize, stack: i32) -> Strategy {
    Strategy::load(&sixmax_home().join("strategies").join(run).join(depth_dir(players, stack))).expect("load strategy")
}

pub fn eval_cmd(args: &[String]) {
    let players: usize = num(args, "--players", 6);
    let stack: i32 = num(args, "--stack", 200);
    let deals: u64 = num(args, "--deals", 2_000_000u64);
    let abs = Abstraction::load(&sixmax_home().join("data").join(flag(args, "--abs").unwrap_or("g200".into()))).expect("load abstraction");
    let t = tree::build(players, stack, &Menu::v1(), abs.buckets());
    let mut names = Vec::new();
    let mut strats = Vec::new();
    for spec in values(args, "--strat") {
        let (name, run) = spec.split_once('=').expect("--strat NAME=RUN");
        let s = load(run, players, stack);
        assert_eq!(s.probs.len() as u64, t.n_entries, "{run}: strategy doesn't match the tree");
        names.push(name.to_string());
        strats.push(s);
    }
    let player = |name: &str| -> Player {
        if let Some(i) = names.iter().position(|n| n == name) {
            Player::Table(&strats[i])
        } else {
            Player::Bot(BOTS.iter().find(|b| b.0 == name).unwrap_or_else(|| panic!("unknown player {name}")).1)
        }
    };
    let mut matches: Vec<(String, String)> = values(args, "--match")
        .iter()
        .map(|m| {
            let (x, y) = m.split_once(',').expect("--match X,Y");
            (x.to_string(), y.to_string())
        })
        .collect();
    if let Some(x) = flag(args, "--vs-bots") {
        for (b, _) in BOTS {
            matches.push((x.clone(), b.to_string()));
        }
    }
    let dealer = PokerDealer { abs: &abs, players };
    for (x, y) in matches {
        let t0 = std::time::Instant::now();
        let r = play_match(&t, &dealer, &player(&x), &player(&y), deals, threads(), 0xE7A1);
        log(format!(
            "{players} players {}bb: {x} (1 seat) vs {y} ({} seats): {:+.1} ± {:.1} bb/100 over {} deals x {players} seats ({:.0}s)",
            stack / 2,
            players - 1,
            r.bb_per_hand * 100.0,
            r.ci * 100.0,
            r.deals,
            t0.elapsed().as_secs_f64()
        ));
    }
}

/// Raise % per starting hand at `node`, as a 13x13 chart (rows = high card,
/// suited above the diagonal).
fn chart(t: &Tree, s: &Strategy, node: u32) -> String {
    let nd = &t.nodes[node as usize];
    let n = nd.n as usize;
    let acts = t.actions(nd);
    let mut out = String::new();
    for row in 0..13u32 {
        for col in 0..13u32 {
            let mut p = [0f64; MAX_ACTIONS];
            s.probs(t, node, row * 13 + col, &mut p);
            let raise: f64 = (0..n).filter(|&a| acts[a].is_aggressive()).map(|a| p[a]).sum();
            out.push_str(&format!("{:>4}", (raise * 100.0).round() as i32));
        }
        out.push_str(&format!("   {}\n", class_name(row * 14)));
    }
    out
}

pub fn probe_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run RUN");
    let players: usize = num(args, "--players", 6);
    let stack: i32 = num(args, "--stack", 200);
    let abs = Abstraction::load(&sixmax_home().join("data").join(flag(args, "--abs").unwrap_or("g200".into()))).expect("load abstraction");
    let t = tree::build(players, stack, &Menu::v1(), abs.buckets());
    let s = load(&run, players, stack);
    // Opening spots: everyone before folds.
    let folds = if players == 6 { vec![0, 3, 4] } else { vec![0] };
    for f in folds {
        let mut node = 0u32;
        for _ in 0..f {
            let nd = &t.nodes[node as usize];
            node = t.children(nd)[t.actions(nd).iter().position(|a| *a == Act::Fold).unwrap()];
        }
        let nd = &t.nodes[node as usize];
        println!(
            "{} raise % first in ({}bb, {} players), rows = high card, suited above the diagonal:\n{}",
            position_name(players, nd.player as usize),
            stack / 2,
            players,
            chart(&t, &s, node)
        );
    }
}
