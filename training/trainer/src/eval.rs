//! Snapshot evaluation: bb/100 against the random bot and earlier snapshots,
//! with a rating ladder in <run-dir>/pool.json.
//!
//! A match is the candidate in one seat against 5 copies of an opponent, on
//! fresh 6 × 100bb tables. Each deal is played 6 times with the candidate
//! rotated through every seat (duplicate dealing: same cards to the same
//! seats), and results are all-in-EV adjusted, which cuts card luck a lot.
//! Neural agents sample from the policy, as the browser bot does.
//!
//! The ladder fits ratings to the snapshot-vs-snapshot bb/100 results by least
//! squares (rating(a) - rating(b) ~ bb/100 of a vs b), with the first snapshot
//! pinned at 0 — the multiplayer stand-in for fox-lite's Bradley–Terry Elo.
//! bb/100 vs random is reported separately (`vs_random`). Each new snapshot
//! plays random, the top `--n-top` rated snapshots, and `--n-anchors`
//! snapshots spread across the rating range.
//!
//!   poker eval --run-dir DIR --candidate snap.safetensors [--deals 2000]
//!   poker match --a snap.safetensors|random --b snap.safetensors|random [--deals 2000]

use std::path::Path;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{json, Value};
use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};

use poker_core::actions::{to_action, NUM_ACTIONS};
use poker_core::allin_ev::{expected_stacks, is_runout};
use poker_core::cards::full_deck;
use poker_core::encode::{encode_into, INPUT_SIZE};
use poker_core::rules::{Action, ActionType, Phase, State, SEATS, STARTING_STACK};

use crate::net::{NetCfg, PokerNet};
use crate::selfplay::sample_action;
use crate::util::{flag, has_flag};

pub const RANDOM: &str = "random";

pub enum Agent {
    Random,
    // The VarStore owns the weights the net uses.
    Net(#[allow(dead_code)] VarStore, PokerNet),
}

pub fn load_agent(spec: &str, dev: Device) -> Agent {
    if spec == RANDOM {
        return Agent::Random;
    }
    let mut vs = VarStore::new(dev);
    let net = PokerNet::new(&vs.root(), NetCfg::POLICY);
    vs.load(spec).unwrap_or_else(|e| panic!("load {spec}: {e}"));
    Agent::Net(vs, net)
}

/// Port of src/engine/random.ts: a uniform legal action type, and a uniform
/// raise-to between the min-raise and all-in.
fn random_action(st: &State, rng: &mut impl Rng) -> Action {
    let l = st.legal().expect("legal");
    let mut kinds = Vec::with_capacity(3);
    if l.can_fold {
        kinds.push(ActionType::Fold);
    }
    kinds.push(if l.can_check { ActionType::Check } else { ActionType::Call });
    if l.can_raise {
        kinds.push(if l.opening { ActionType::Bet } else { ActionType::Raise });
    }
    let k = kinds[rng.gen_range(0..kinds.len())];
    match k {
        ActionType::Bet | ActionType::Raise => Action::to(k, rng.gen_range(l.min_raise_to..=l.max_raise_to)),
        _ => Action::new(k),
    }
}

struct Game {
    st: State,
    cand: usize,
    result: Option<f64>, // candidate's chip result (EV-adjusted)
}

impl Game {
    /// Advance to the next decision (true) or the end of the hand (false).
    fn advance(&mut self, rng: &mut StdRng) -> bool {
        loop {
            match self.st.phase {
                Phase::Betting if self.st.to_act >= 0 => return true,
                Phase::HandOver | Phase::MatchOver => {
                    self.result = Some((self.st.players[self.cand].stack - STARTING_STACK) as f64);
                    return false;
                }
                _ if is_runout(&self.st) => {
                    let ev = expected_stacks(&self.st, rng);
                    self.result = Some(ev[self.cand] - STARTING_STACK as f64);
                    return false;
                }
                _ => self.st.advance_street(),
            }
        }
    }
}

/// Candidate `a` in one seat vs 5 copies of `b`, `deals` deals × 6 seat
/// rotations. Returns (bb/100 for a, 95% CI half-width).
pub fn play_match(a: &Agent, b: &Agent, deals: usize, seed: u64, dev: Device) -> (f64, f64) {
    const CHUNK: usize = 4096;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut deal_sums = vec![0f64; deals];
    let total = deals * SEATS;
    let mut next = 0;
    while next < total {
        let n = CHUNK.min(total - next);
        let mut games: Vec<Game> = (next..next + n)
            .map(|h| {
                let (d, rot) = (h / SEATS, h % SEATS);
                let mut drng = StdRng::seed_from_u64(seed ^ (d as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                let mut deck = full_deck();
                deck.shuffle(&mut drng);
                let mut st = State::new_table([STARTING_STACK; SEATS], None);
                st.button = (d % SEATS) as i32; // rotate the button across deals too
                st.start_hand(deck);
                Game { st, cand: rot, result: None }
            })
            .collect();
        let mut x = vec![0f32; INPUT_SIZE];
        loop {
            let live: Vec<usize> = (0..n).filter(|&i| games[i].result.is_none() && games[i].advance(&mut rng)).collect();
            if live.is_empty() {
                break;
            }
            for (agent, is_cand) in [(a, true), (b, false)] {
                let mine: Vec<usize> =
                    live.iter().copied().filter(|&i| (games[i].st.to_act as usize == games[i].cand) == is_cand).collect();
                if mine.is_empty() {
                    continue;
                }
                match agent {
                    Agent::Random => {
                        for &i in &mine {
                            let act = random_action(&games[i].st, &mut rng);
                            games[i].st.apply(act);
                        }
                    }
                    Agent::Net(_, net) => {
                        let mut buf = Vec::with_capacity(mine.len() * INPUT_SIZE);
                        let mut spaces = Vec::with_capacity(mine.len());
                        for &i in &mine {
                            spaces.push(encode_into(&games[i].st, &mut x).expect("decision"));
                            buf.extend_from_slice(&x);
                        }
                        let xs = Tensor::from_slice(&buf).view([mine.len() as i64, INPUT_SIZE as i64]).to_device(dev);
                        let out = tch::no_grad(|| net.forward(&xs));
                        let logits =
                            Vec::<f32>::try_from(&out.to_kind(Kind::Float).to_device(Device::Cpu).contiguous().view([-1])).unwrap();
                        for (k, &i) in mine.iter().enumerate() {
                            let sp = &spaces[k];
                            let (act, _) = sample_action(sp, &logits[k * NUM_ACTIONS..(k + 1) * NUM_ACTIONS], &mut rng);
                            games[i].st.apply(to_action(sp, act));
                        }
                    }
                }
            }
        }
        for (k, g) in games.iter().enumerate() {
            deal_sums[(next + k) / SEATS] += g.result.unwrap();
        }
        next += n;
    }
    // Per-deal mean result in bb, then bb/100.
    let per: Vec<f64> = deal_sums.iter().map(|s| s / SEATS as f64 / 2.0).collect();
    let mean = per.iter().sum::<f64>() / deals as f64;
    let var = per.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (deals.max(2) - 1) as f64;
    (mean * 100.0, 1.96 * (var / deals as f64).sqrt() * 100.0)
}

// ---- pool / ladder ---------------------------------------------------------------

fn load_pool(path: &Path) -> Value {
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok())
        .unwrap_or_else(|| json!({"models": {RANDOM: {}}, "results": []}))
}

/// Least-squares ratings from snapshot-vs-snapshot bb/100 (Gauss-Seidel), with
/// the earliest snapshot pinned at 0. Results against random are left out of
/// the fit: beating random maniacs is a different skill (a tight, sound
/// policy wins *less* against them than a loose one), so mixing them in can
/// rank a weaker snapshot first. Each model's bb/100 vs random is kept as
/// `vs_random` instead.
fn refit(pool: &mut Value) {
    let mut names: Vec<String> =
        pool["models"].as_object().unwrap().keys().filter(|k| k.as_str() != RANDOM).cloned().collect();
    names.sort(); // snap_hHHHHH_cCCCCCC: chronological
    let all: Vec<(String, String, f64)> = pool["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| (r["a"].as_str().unwrap().to_string(), r["b"].as_str().unwrap().to_string(), r["bb100"].as_f64().unwrap()))
        .collect();
    let mut results: Vec<(String, String, f64)> = Vec::new();
    for (a, b, y) in all {
        if b == RANDOM {
            pool["models"][&a]["vs_random"] = json!(y);
        } else if a != RANDOM {
            results.push((a, b, y));
        }
    }
    let mut rating: std::collections::HashMap<String, f64> = names.iter().map(|n| (n.clone(), 0.0)).collect();
    for _ in 0..500 {
        for n in names.iter().skip(1) {
            let (mut num, mut den) = (0.0, 0.0);
            for (a, b, y) in &results {
                if a == n {
                    num += rating[b] + y;
                    den += 1.0;
                } else if b == n {
                    num += rating[a] - y;
                    den += 1.0;
                }
            }
            if den > 0.0 {
                rating.insert(n.clone(), num / den);
            }
        }
    }
    for n in &names {
        pool["models"][n]["rating"] = json!((rating[n] * 10.0).round() / 10.0);
    }
    if let Some(m) = pool["models"].get_mut(RANDOM) {
        m.as_object_mut().unwrap().remove("rating");
    }
}

/// Evaluate a new snapshot against the pool and update pool.json. The
/// snapshots named in `must` (e.g. the previous one and the training anchor)
/// are always among the opponents. Returns the candidate's (bb/100, 95% CI)
/// against every opponent played.
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    run_dir: &str,
    candidate: &Path,
    must: &[String],
    deals: usize,
    n_top: usize,
    n_anchors: usize,
    seed: u64,
    dev: Device,
) -> std::collections::HashMap<String, (f64, f64)> {
    let pool_path = Path::new(run_dir).join("pool.json");
    let mut pool = load_pool(&pool_path);
    let name = candidate.file_stem().unwrap().to_string_lossy().into_owned();

    // Opponents: random, the top-rated snapshots, and anchors spread over the range.
    let mut rated: Vec<(String, f64)> = pool["models"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.as_str() != RANDOM && **k != name)
        .map(|(k, v)| (k.clone(), v["rating"].as_f64().unwrap_or(0.0)))
        .collect();
    rated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let mut opps: Vec<String> = vec![RANDOM.to_string()];
    for p in must {
        if pool["models"].get(p).is_some() && !opps.contains(p) {
            opps.push(p.clone());
        }
    }
    for r in rated.iter().take(n_top) {
        if !opps.contains(&r.0) {
            opps.push(r.0.clone());
        }
    }
    if let Some(top) = rated.first().map(|r| r.1) {
        for k in 1..=n_anchors {
            let target = top * k as f64 / (n_anchors + 1) as f64;
            if let Some(best) = rated
                .iter()
                .filter(|r| !opps.contains(&r.0))
                .min_by(|a, b| (a.1 - target).abs().partial_cmp(&(b.1 - target).abs()).unwrap())
            {
                opps.push(best.0.clone());
            }
        }
    }

    let mut results = std::collections::HashMap::new();
    let cand = load_agent(&candidate.to_string_lossy(), dev);
    pool["models"][&name] = json!({"rating": 0.0, "path": candidate.to_string_lossy()});
    for o in &opps {
        let spec = if o == RANDOM { RANDOM.to_string() } else { pool["models"][o]["path"].as_str().unwrap().to_string() };
        let opp = load_agent(&spec, dev);
        let (bb, ci) = play_match(&cand, &opp, deals, seed, dev);
        println!("[eval] {name} vs {o}: {bb:+.1} ± {ci:.1} bb/100 ({} hands)", deals * SEATS);
        results.insert(o.clone(), (bb, ci));
        pool["results"].as_array_mut().unwrap().push(json!({
            "a": name, "b": o, "bb100": (bb * 10.0).round() / 10.0, "ci": (ci * 10.0).round() / 10.0, "deals": deals,
        }));
    }
    refit(&mut pool);
    let mut ladder: Vec<(String, f64)> = pool["models"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.as_str() != RANDOM)
        .map(|(k, v)| (k.clone(), v["rating"].as_f64().unwrap()))
        .collect();
    ladder.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top: Vec<String> = ladder
        .iter()
        .take(5)
        .map(|(k, r)| format!("{k}={r:+.1} (vs random {:+.0})", pool["models"][k]["vs_random"].as_f64().unwrap_or(0.0)))
        .collect();
    println!("[eval] ladder: {}", top.join("  "));
    std::fs::write(&pool_path, serde_json::to_string_pretty(&pool).unwrap()).expect("write pool.json");
    results
}

fn usize_flag(args: &[String], key: &str, default: usize) -> usize {
    flag(args, key).map(|v| v.parse().unwrap()).unwrap_or(default)
}

fn device() -> Device {
    if tch::Cuda::is_available() { Device::Cuda(0) } else { Device::Cpu }
}

/// Called by the training daemon after each snapshot (see `evaluate`).
pub fn after_snapshot(
    run_dir: &str,
    snapshot: &Path,
    must: &[String],
    args: &[String],
) -> std::collections::HashMap<String, (f64, f64)> {
    if has_flag(args, "--no-eval") {
        return Default::default();
    }
    let seed = snapshot.to_string_lossy().bytes().fold(7u64, |h, b| h.wrapping_mul(31).wrapping_add(b as u64));
    evaluate(
        run_dir,
        snapshot,
        must,
        usize_flag(args, "--eval-deals", 2000),
        usize_flag(args, "--n-top", 2),
        usize_flag(args, "--n-anchors", 3),
        seed,
        device(),
    )
}

/// `poker eval`
pub fn eval_cmd(args: &[String]) {
    let run_dir = flag(args, "--run-dir").expect("--run-dir");
    let cand = flag(args, "--candidate").expect("--candidate snap.safetensors");
    evaluate(
        &run_dir,
        Path::new(&cand),
        &[],
        usize_flag(args, "--deals", 2000),
        usize_flag(args, "--n-top", 2),
        usize_flag(args, "--n-anchors", 3),
        usize_flag(args, "--seed", 1) as u64,
        device(),
    );
}

/// `poker match`
pub fn match_cmd(args: &[String]) {
    let dev = device();
    let a = flag(args, "--a").expect("--a");
    let b = flag(args, "--b").expect("--b");
    let deals = usize_flag(args, "--deals", 2000);
    let (bb, ci) = play_match(&load_agent(&a, dev), &load_agent(&b, dev), deals, usize_flag(args, "--seed", 1) as u64, dev);
    println!("{a} vs 5x {b}: {bb:+.1} ± {ci:.1} bb/100 ({} hands)", deals * SEATS);
}

/// `poker best --run-dir DIR`: print the path of the top-rated snapshot.
pub fn best_cmd(args: &[String]) {
    let run_dir = flag(args, "--run-dir").expect("--run-dir");
    let mut pool = load_pool(&Path::new(&run_dir).join("pool.json"));
    refit(&mut pool);
    let best = pool["models"]
        .as_object()
        .unwrap()
        .iter()
        .filter(|(k, _)| k.as_str() != RANDOM)
        .max_by(|a, b| a.1["rating"].as_f64().partial_cmp(&b.1["rating"].as_f64()).unwrap());
    match best {
        Some((_, v)) => println!("{}", v["path"].as_str().unwrap()),
        None => {
            eprintln!("no rated snapshots in {run_dir}/pool.json yet");
            std::process::exit(1);
        }
    }
}
