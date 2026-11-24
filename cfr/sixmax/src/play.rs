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

// ---------------------------------------------------------------------------
// `sixmax match`: the play-time engine under the real rules.
//
//   sixmax match --run RUN [--abs g200] [--vs random|station|maniac|engine]
//                [--hands 20000] [--stacks 200|random]
//       --vs BOT: the engine in one seat (every seat in turn) against five
//       scripted bots; its result in bb/100. --vs engine: six engines. Both
//       report how often the engine fell back to its equity heuristic and
//       check that every action it chose was legal. With --stacks random,
//       each hand deals every seat 6-150bb and 0-3 empty seats.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::Mutex;

use poker_core::rules::{Action, ActionType, Phase, State, Status, SEATS};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use sixmax_cfr::engine::{Engine, HistAction, View};

#[derive(Clone, Copy, PartialEq)]
enum Seat {
    Engine,
    Random,
    Station,
    Maniac,
}

fn bot_action(kind: Seat, s: &State, rng: &mut SmallRng) -> Action {
    let legal = s.legal().unwrap();
    let passive = Action::new(if legal.can_check { ActionType::Check } else { ActionType::Call });
    match kind {
        Seat::Station => passive,
        Seat::Maniac => {
            if legal.can_raise {
                let pot = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>() + legal.to_call;
                let to = (s.current_bet + pot).clamp(legal.min_raise_to, legal.max_raise_to);
                if to >= legal.max_raise_to {
                    Action::new(ActionType::AllIn)
                } else {
                    Action::to(if legal.opening { ActionType::Bet } else { ActionType::Raise }, to)
                }
            } else {
                passive
            }
        }
        _ => {
            // As the app's random bot: a uniform legal action type, a uniform raise size.
            let mut kinds = vec![];
            if legal.can_fold {
                kinds.push(ActionType::Fold);
            }
            kinds.push(if legal.can_check { ActionType::Check } else { ActionType::Call });
            if legal.can_raise {
                kinds.push(if legal.opening { ActionType::Bet } else { ActionType::Raise });
                kinds.push(ActionType::AllIn);
            }
            match kinds[rng.gen_range(0..kinds.len())] {
                k @ (ActionType::Bet | ActionType::Raise) => {
                    let to = if legal.max_raise_to > legal.min_raise_to { rng.gen_range(legal.min_raise_to..legal.max_raise_to) } else { legal.min_raise_to };
                    Action::to(k, to)
                }
                k => Action::new(k),
            }
        }
    }
}

pub fn match_cmd(args: &[String]) {
    let run = flag(args, "--run").expect("--run RUN");
    let abs_dir = sixmax_home().join("data").join(flag(args, "--abs").unwrap_or("g200".into()));
    let engine = Engine::load(&abs_dir, &sixmax_home().join("strategies").join(&run)).expect("load engine");
    log(format!(
        "engine {run}: 6-max depths {:?}bb, heads-up depths {:?}bb",
        engine.six.iter().map(|d| d.strategy.meta.stack / 2).collect::<Vec<_>>(),
        engine.hu.iter().map(|d| d.strategy.meta.stack / 2).collect::<Vec<_>>()
    ));
    let vs = flag(args, "--vs").unwrap_or("random".into());
    let others = match vs.as_str() {
        "random" => Seat::Random,
        "station" => Seat::Station,
        "maniac" => Seat::Maniac,
        "engine" => Seat::Engine,
        o => panic!("unknown opponent {o}"),
    };
    let hands: u64 = num(args, "--hands", 20_000u64);
    let random_stacks = flag(args, "--stacks").as_deref() == Some("random");
    let fixed: i32 = flag(args, "--stacks").and_then(|s| s.parse().ok()).unwrap_or(200);
    let (decisions, fallbacks) = (AtomicU64::new(0), AtomicU64::new(0));
    let results = Mutex::new(Vec::<f64>::new());
    let reasons = Mutex::new(std::collections::BTreeMap::<String, u64>::new());
    let nthreads = threads() as u64;
    let t0 = std::time::Instant::now();
    std::thread::scope(|sc| {
        for th in 0..nthreads {
            let (engine, decisions, fallbacks, results, reasons) = (&engine, &decisions, &fallbacks, &results, &reasons);
            sc.spawn(move || {
                let mut rng = SmallRng::seed_from_u64(0xBEEF ^ (th + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut mine = Vec::new();
                let mut h = th;
                while h < hands {
                    let hero = (h % SEATS as u64) as usize;
                    let mut stacks = [fixed; SEATS];
                    if random_stacks {
                        for st in stacks.iter_mut() {
                            *st = rng.gen_range(12..=300);
                        }
                        for _ in 0..rng.gen_range(0..=3) {
                            let k = rng.gen_range(0..SEATS);
                            if k != hero {
                                stacks[k] = 0;
                            }
                        }
                    }
                    let button = loop {
                        let b = rng.gen_range(0..SEATS);
                        if stacks[b] > 0 {
                            break b;
                        }
                    };
                    let mut s = State::new_table(stacks, None);
                    s.button = ((button + SEATS - 1) % SEATS) as i32;
                    let mut deck: Vec<u8> = (0..52).collect();
                    for i in (1..52).rev() {
                        deck.swap(i, rng.gen_range(0..=i));
                    }
                    s.start_hand(deck);
                    let seed: u64 = rng.gen();
                    while !s.hand_done() {
                        if s.phase == Phase::StreetComplete {
                            s.advance_street();
                            continue;
                        }
                        let seat = s.to_act as usize;
                        let kind = if seat == hero || others == Seat::Engine { Seat::Engine } else { others };
                        let action = if kind == Seat::Engine {
                            let v = View {
                                bot: seat,
                                button: s.button as usize,
                                stacks,
                                hole: [s.players[seat].hole[0], s.players[seat].hole[1]],
                                board: s.community.clone(),
                                actions: s.actions.iter().map(|a| HistAction { seat: a.seat, kind: a.kind, to: a.to }).collect(),
                                seed,
                            };
                            let d = engine.decide(&v, &mut rng).unwrap_or_else(|e| panic!("engine error: {e}"));
                            decisions.fetch_add(1, Relaxed);
                            if let Some(why) = &d.debug.fallback {
                                fallbacks.fetch_add(1, Relaxed);
                                *reasons.lock().unwrap().entry(why.clone()).or_default() += 1;
                            }
                            // Every engine action must be legal as chosen.
                            let legal = s.legal().unwrap();
                            match d.action.kind {
                                ActionType::Fold => assert!(legal.can_fold, "illegal fold"),
                                ActionType::Check => assert!(legal.can_check, "illegal check"),
                                ActionType::Call => assert!(legal.can_call, "illegal call"),
                                ActionType::Bet | ActionType::Raise => {
                                    let to = d.action.amount.unwrap();
                                    assert!(legal.can_raise && to >= legal.min_raise_to && to <= legal.max_raise_to, "illegal raise to {to}");
                                }
                                ActionType::AllIn => {}
                            }
                            d.action
                        } else {
                            bot_action(kind, &s, &mut rng)
                        };
                        s.apply(action);
                    }
                    let start = stacks[hero];
                    let end = s.players[hero].stack;
                    mine.push((end - start) as f64 / 2.0);
                    let _ = Status::Out;
                    h += nthreads;
                }
                results.lock().unwrap().extend(mine);
            });
        }
    });
    let r = results.into_inner().unwrap();
    let n = r.len() as f64;
    let mean = r.iter().sum::<f64>() / n;
    let var = r.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let (d, f) = (decisions.into_inner(), fallbacks.into_inner());
    log(format!(
        "{} hands vs {vs} ({} stacks): engine {:+.1} ± {:.1} bb/100; {} engine decisions, all legal; fallback {:.2}% ({:.0}s)",
        r.len(),
        if random_stacks { "random".to_string() } else { format!("{}bb", fixed / 2) },
        mean * 100.0,
        1.96 * (var / n).sqrt() * 100.0,
        d,
        f as f64 / d.max(1) as f64 * 100.0,
        t0.elapsed().as_secs_f64()
    ));
    for (why, k) in reasons.into_inner().unwrap() {
        log(format!("  fallback: {why}: {k}"));
    }
}
