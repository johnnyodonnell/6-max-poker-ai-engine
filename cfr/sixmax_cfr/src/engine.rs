//! Play-time decisions from trained strategies (the heads-up engine's
//! `Engine::decide`, generalized to six seats).
//!
//! For each decision the engine:
//! 1. replays the hand so far through the rules port (the exact real state,
//!    and a parity check on every recorded action);
//! 2. picks the tree: the heads-up ladder with two players dealt in, else the
//!    6-handed ladder. Real seats map to positions by their distance from the
//!    button; a 5-, 4- or 3-handed table enters the 6-handed tree with the
//!    missing early positions (UTG, then MP, then CO) folded;
//! 3. picks the stack depth nearest (in ratio) the effective stack: the bot's
//!    stack at the start of the hand against the largest stack of another
//!    player still in the hand;
//! 4. walks the tree along the real history, translating each real bet or
//!    raise to an abstract size with the pseudo-harmonic mapping (Ganzfried &
//!    Sandholm 2013), seeded per hand so earlier actions translate the same
//!    way at every decision. A player who is all-in for real but not in the
//!    tree (a short stack) checks or calls wherever the tree asks them to act;
//! 5. looks up the bot's card group and the strategy there, turns each
//!    abstract action into a legal real one (pot fractions of the *real*
//!    pot), and samples one.
//!
//! Where the tree can't follow the real hand (a raise past the cap, a deep
//! stack's bet translated to all-in while the real hand goes on, ...) an
//! equity heuristic plays instead (`fallback_action`), and the debug output
//! says why.

use std::path::Path;

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use serde::Serialize;

use poker_core::cards::Card;
use poker_core::hand_eval::eval7;
use poker_core::rules::{Action, ActionType, HandAction, Legal, Phase, State, Status, Street, SEATS};

use crate::abs::Abstraction;
use crate::menu::Menu;
use crate::strategy::Strategy;
use crate::tree::{self, position_name, Act, Tree, MAX_ACTIONS, TERMINAL};

/// One betting action of the hand so far.
#[derive(Clone, Copy, Debug)]
pub struct HistAction {
    pub seat: usize,
    /// fold / check / call / bet / raise (an all-in is the call, bet or
    /// raise it amounts to).
    pub kind: ActionType,
    /// The actor's street bet after the action (the raise-to for bet/raise).
    pub to: i32,
}

/// What the bot knows when it must act.
#[derive(Clone, Debug)]
pub struct View {
    pub bot: usize,
    pub button: usize,
    /// Every seat's stack at the start of the hand, before the blinds
    /// (0 = sitting out).
    pub stacks: [i32; SEATS],
    pub hole: [Card; 2],
    pub board: Vec<Card>,
    pub actions: Vec<HistAction>,
    /// Per-hand seed: keeps the translation of earlier actions stable across
    /// the bot's decisions in one hand.
    pub seed: u64,
}

#[derive(Clone, Debug, Serialize)]
pub struct Opt {
    /// The abstract action.
    pub label: String,
    /// The real action it becomes.
    pub action: String,
    pub prob: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct Debug {
    /// "6-max" (with the table size) or "heads-up".
    pub table: String,
    pub players: usize,
    pub effective_bb: f64,
    pub strategy_bb: f64,
    pub street: String,
    pub bucket: Option<u32>,
    /// The bot's hand group, readable: its preflop class, or "group n of N".
    pub bucket_label: Option<String>,
    pub node: Option<u32>,
    /// The abstract betting history the real one was translated to.
    pub path: Vec<String>,
    pub options: Vec<Opt>,
    pub chosen: String,
    /// Why the equity heuristic was used, if it was.
    pub fallback: Option<String>,
}

pub struct Decision {
    pub action: Action,
    pub debug: Debug,
}

/// One trained table size and stack depth.
pub struct Depth {
    pub tree: Tree,
    pub strategy: Strategy,
}

pub struct Engine {
    pub abs: Abstraction,
    /// 6-handed depths, sorted by stack.
    pub six: Vec<Depth>,
    /// Heads-up depths, sorted by stack.
    pub hu: Vec<Depth>,
}

fn street_index(s: Street) -> usize {
    match s {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    }
}

fn street_name(s: Street) -> &'static str {
    ["preflop", "flop", "turn", "river"][street_index(s)]
}

pub fn action_label(a: &Action) -> String {
    match a.kind {
        ActionType::Fold => "fold".into(),
        ActionType::Check => "check".into(),
        ActionType::Call => "call".into(),
        ActionType::Bet => format!("bet {}", a.amount.unwrap_or(0)),
        ActionType::Raise => format!("raise to {}", a.amount.unwrap_or(0)),
        ActionType::AllIn => "all-in".into(),
    }
}

fn hand_seed(seed: u64, i: usize) -> u64 {
    seed ^ (i as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15)
}

/// A full deck ordered so `start_hand` deals the bot its cards and the
/// streets deal the known board (the others' cards and the rest are filler).
fn deck_for(v: &View) -> Result<Vec<Card>, String> {
    let mut used = [false; 52];
    for &c in v.hole.iter().chain(&v.board) {
        if c >= 52 || used[c as usize] {
            return Err(format!("bad or repeated card {c}"));
        }
        used[c as usize] = true;
    }
    let mut spare = (0..52u8).filter(|&c| !used[c as usize]);
    let mut order: Vec<Card> = Vec::with_capacity(52);
    for seat in 0..SEATS {
        if v.stacks[seat] <= 0 {
            continue;
        }
        if seat == v.bot {
            order.extend_from_slice(&v.hole);
        } else {
            order.push(spare.next().unwrap());
            order.push(spare.next().unwrap());
        }
    }
    order.extend_from_slice(&v.board);
    order.extend(spare);
    order.reverse(); // the rules pop cards off the end
    Ok(order)
}

/// A fresh hand dealt per the view: the button where the app had it.
fn start(v: &View) -> Result<State, String> {
    if v.bot >= SEATS || v.button >= SEATS {
        return Err("bad seat".into());
    }
    if v.stacks.iter().any(|&s| s < 0) || v.stacks[v.bot] <= 0 || v.stacks[v.button] <= 0 {
        return Err(format!("bad stacks {:?}", v.stacks));
    }
    let mut s = State::new_table(v.stacks, None);
    s.button = ((v.button + SEATS - 1) % SEATS) as i32; // start_hand moves it on one live seat
    s.start_hand(deck_for(v)?);
    if s.phase != Phase::Betting || s.button as usize != v.button {
        return Err("couldn't deal the hand as the app did".into());
    }
    Ok(s)
}

/// Apply real action `a` (the `i`-th of the hand), checking it replays as recorded.
fn step(s: &mut State, a: &HistAction, i: usize) -> Result<(), String> {
    while s.phase == Phase::StreetComplete {
        s.advance_street();
    }
    if s.phase != Phase::Betting || s.to_act < 0 || s.to_act as usize != a.seat {
        return Err(format!("action {i}: seat {} is not to act", a.seat));
    }
    let act = match a.kind {
        ActionType::Bet | ActionType::Raise => Action::to(a.kind, a.to),
        k => Action::new(k),
    };
    s.apply(act);
    let rec = s.actions.last().unwrap();
    if rec.kind != a.kind || rec.to != a.to {
        return Err(format!("action {i}: replayed as {:?} to {} (sent {:?} to {})", rec.kind, rec.to, a.kind, a.to));
    }
    Ok(())
}

/// The real state at the bot's decision.
pub fn replay(v: &View) -> Result<State, String> {
    let mut s = start(v)?;
    for (i, a) in v.actions.iter().enumerate() {
        step(&mut s, a, i)?;
    }
    while s.phase == Phase::StreetComplete && s.community.len() < v.board.len() {
        s.advance_street();
    }
    if s.phase != Phase::Betting || s.to_act < 0 || s.to_act as usize != v.bot {
        return Err("it is not the bot's turn".into());
    }
    if s.community != v.board {
        return Err("board mismatch".into());
    }
    Ok(s)
}

/// The abstract action a real one is translated to (index into `acts`).
fn translate(acts: &[Act], ha: &HandAction, rng: &mut SmallRng) -> usize {
    let find = |pred: &dyn Fn(&Act) -> bool| acts.iter().position(|a| pred(a));
    let passive = || find(&|a| matches!(a, Act::Check | Act::Call)).unwrap_or(0);
    match ha.kind {
        ActionType::Fold => find(&|a| matches!(a, Act::Fold)).unwrap_or_else(passive),
        ActionType::Check | ActionType::Call => passive(),
        _ => {
            let allin = find(&|a| matches!(a, Act::AllIn { .. }));
            if ha.allin {
                if let Some(k) = allin {
                    // A short stack's all-in reads as the size it is, not as
                    // the tree's (deeper) all-in.
                    let bet_before = ha.to - ha.added;
                    let pac = ha.pot_before + (ha.current_bet_before - bet_before);
                    let f = (ha.to - ha.current_bet_before) as f64 / pac.max(1) as f64;
                    if f >= acts[k].frac().unwrap() as f64 * 0.75 {
                        return k;
                    }
                }
            }
            let bet_before = ha.to - ha.added;
            let pac = ha.pot_before + (ha.current_bet_before - bet_before);
            let f = (ha.to - ha.current_bet_before) as f64 / pac.max(1) as f64;
            let mut sizes: Vec<(usize, f64)> = acts
                .iter()
                .enumerate()
                .filter(|(_, a)| matches!(a, Act::Bet { .. } | Act::Raise { .. }))
                .map(|(i, a)| (i, a.frac().unwrap() as f64))
                .collect();
            // All-in is the upper neighbour for oversized bets.
            if let Some(k) = allin {
                sizes.push((k, acts[k].frac().unwrap() as f64));
            }
            if sizes.is_empty() {
                return allin.unwrap_or_else(passive);
            }
            sizes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            if f <= sizes[0].1 {
                return sizes[0].0;
            }
            if f >= sizes[sizes.len() - 1].1 {
                return sizes[sizes.len() - 1].0;
            }
            let j = sizes.iter().position(|s| s.1 >= f).unwrap();
            let ((ia, a), (ib, b)) = (sizes[j - 1], sizes[j]);
            // Pseudo-harmonic mapping: probability of mapping down to A.
            let p_a = (b - f) * (1.0 + a) / ((b - a) * (1.0 + f));
            if rng.gen::<f64>() < p_a {
                ia
            } else {
                ib
            }
        }
    }
}

/// The real action an abstract one becomes at the current real state.
fn to_real(a: &Act, legal: &Legal, current_bet: i32, pac: i32) -> Action {
    let passive = || Action::new(if legal.can_check { ActionType::Check } else { ActionType::Call });
    match *a {
        Act::Fold => {
            if legal.can_fold {
                Action::new(ActionType::Fold)
            } else {
                passive()
            }
        }
        Act::Check | Act::Call => passive(),
        Act::Bet { frac, .. } | Act::Raise { frac, .. } => {
            if !legal.can_raise {
                return passive();
            }
            let to = (current_bet as f64 + frac as f64 * pac as f64).round() as i32;
            let to = to.clamp(legal.min_raise_to, legal.max_raise_to);
            if to >= legal.max_raise_to {
                Action::new(ActionType::AllIn)
            } else {
                Action::to(if legal.opening { ActionType::Bet } else { ActionType::Raise }, to)
            }
        }
        Act::AllIn { .. } => {
            if legal.can_raise {
                Action::new(ActionType::AllIn)
            } else {
                passive()
            }
        }
    }
}

/// Equity against `opps` random hands (Monte Carlo, seeded: deterministic).
pub fn equity_vs_random(hole: [Card; 2], board: &[Card], opps: usize, seed: u64) -> f64 {
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut dead = 0u64;
    for &c in hole.iter().chain(board) {
        dead |= 1 << c;
    }
    let mut cards = [0u8; 7];
    let n = 1500;
    let mut won = 0.0;
    for _ in 0..n {
        let mut used = dead;
        let mut draw = || loop {
            let c = rng.gen_range(0..52u8);
            if used & (1 << c) == 0 {
                used |= 1 << c;
                return c;
            }
        };
        let mut opp = [[0u8; 2]; 5];
        for o in opp.iter_mut().take(opps) {
            *o = [draw(), draw()];
        }
        let mut full = [0u8; 5];
        full[..board.len()].copy_from_slice(board);
        for slot in full[board.len()..].iter_mut() {
            *slot = draw();
        }
        cards[2..].copy_from_slice(&full);
        cards[0] = hole[0];
        cards[1] = hole[1];
        let me = eval7(&cards);
        let (mut best, mut ties) = (true, 1.0);
        for o in &opp[..opps] {
            cards[0] = o[0];
            cards[1] = o[1];
            let them = eval7(&cards);
            if them > me {
                best = false;
                break;
            }
            if them == me {
                ties += 1.0;
            }
        }
        if best {
            won += 1.0 / ties;
        }
    }
    won / n as f64
}

/// The equity heuristic: with `eq` against the live opponents, call when it
/// beats the pot odds by a margin, jam the strongest hands, bet good hands
/// when checked to, otherwise check or fold.
fn fallback_action(eq: f64, opps: usize, legal: &Legal, s: &State) -> Action {
    let pot_total = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
    let to_call = legal.to_call;
    // What a strong hand is depends on how many opponents there are.
    let strong = 1.0 - 0.2 / (opps.max(1) as f64).sqrt();
    if to_call > 0 {
        let odds = to_call as f64 / (pot_total + to_call) as f64;
        if eq > strong && legal.can_raise {
            Action::new(ActionType::AllIn)
        } else if eq >= odds + 0.05 {
            Action::new(ActionType::Call)
        } else {
            Action::new(ActionType::Fold)
        }
    } else if eq > strong - 0.05 && legal.can_raise {
        let me = &s.players[s.to_act as usize];
        let pac = pot_total + (s.current_bet - me.bet);
        to_real(&Act::Bet { to: 0, frac: 0.75 }, legal, s.current_bet, pac)
    } else {
        Action::new(ActionType::Check)
    }
}

/// Where the real seats sit in the tree: the tree's seat for each real seat
/// (None if not dealt in) and how many early positions start folded.
fn seat_map(s: &State) -> ([Option<usize>; SEATS], usize, usize) {
    let mut order = Vec::with_capacity(SEATS);
    let b = s.button as usize;
    for k in 0..SEATS {
        let i = (b + k) % SEATS;
        if s.players[i].status != Status::Out {
            order.push(i);
        }
    }
    let n = order.len();
    let mut map = [None; SEATS];
    if n == 2 {
        map[order[0]] = Some(0);
        map[order[1]] = Some(1);
        return (map, 2, 0);
    }
    let missing = SEATS - n;
    for (k, &i) in order.iter().enumerate() {
        map[i] = Some(if k < 3 { k } else { k + missing });
    }
    (map, n, missing)
}

/// The tree's passive action at `node` (check or call).
fn passive_index(t: &Tree, node: u32) -> usize {
    let nd = &t.nodes[node as usize];
    t.actions(nd).iter().position(|a| matches!(a, Act::Check | Act::Call)).unwrap_or(0)
}

/// While the tree asks a player who is all-in for real to act, take the
/// tree's check/call for them.
fn skip_real_allins(t: &Tree, node: &mut u32, s: &State, back: &[usize; SEATS], path: &mut Vec<String>) {
    loop {
        let nd = &t.nodes[*node as usize];
        if nd.player == TERMINAL {
            return;
        }
        let real = back[nd.player as usize];
        if real == usize::MAX || s.players[real].status != Status::AllIn {
            return;
        }
        let k = passive_index(t, *node);
        path.push(format!("{}: {} (all-in for real)", position_name(t.players, nd.player as usize), t.actions(nd)[k].label()));
        *node = t.children(nd)[k];
    }
}

impl Engine {
    /// Load every exported depth under `run_dir` (`6p-s<chips>/`,
    /// `2p-s<chips>/`) with the card groups in `abs_dir`.
    pub fn load(abs_dir: &Path, run_dir: &Path) -> std::io::Result<Engine> {
        let abs = Abstraction::load(abs_dir)?;
        let mut six = Vec::new();
        let mut hu = Vec::new();
        let mut dirs: Vec<_> = std::fs::read_dir(run_dir)?.filter_map(|e| e.ok()).map(|e| e.path()).collect();
        dirs.sort();
        for dir in dirs {
            if !dir.join("strategy.u8").exists() {
                continue;
            }
            let strategy = Strategy::open(&dir)?;
            let menu = Menu::by_name(&strategy.meta.menu).ok_or_else(|| std::io::Error::other("unknown menu"))?;
            let tree = tree::build(strategy.meta.players, strategy.meta.stack, &menu, abs.buckets());
            if tree.n_entries != strategy.meta.entries {
                return Err(std::io::Error::other(format!("{}: strategy doesn't match its tree", dir.display())));
            }
            let d = Depth { tree, strategy };
            if d.strategy.meta.players == 2 {
                hu.push(d)
            } else {
                six.push(d)
            }
        }
        six.sort_by_key(|d| d.strategy.meta.stack);
        hu.sort_by_key(|d| d.strategy.meta.stack);
        Ok(Engine { abs, six, hu })
    }

    /// The trained depth nearest `eff` chips in ratio.
    fn pick(list: &[Depth], eff: i32) -> Option<&Depth> {
        let e = eff.max(1) as f64;
        list.iter().min_by(|a, b| {
            let da = (a.strategy.meta.stack as f64 / e).ln().abs();
            let db = (b.strategy.meta.stack as f64 / e).ln().abs();
            da.partial_cmp(&db).unwrap()
        })
    }

    /// The bot's decision.
    pub fn decide(&self, v: &View, rng: &mut SmallRng) -> Result<Decision, String> {
        let s = replay(v)?;
        let legal = s.legal().ok_or("no legal actions")?;
        let (map, players, missing) = seat_map(&s);
        let mut back = [usize::MAX; SEATS];
        for (real, t) in map.iter().enumerate() {
            if let Some(t) = t {
                back[*t] = real;
            }
        }
        // The effective stack: the bot's against the deepest other player
        // still in the hand (stacks at the start of the hand).
        let deepest = (0..SEATS)
            .filter(|&i| i != v.bot && matches!(s.players[i].status, Status::Active | Status::AllIn))
            .map(|i| v.stacks[i])
            .max()
            .unwrap_or(v.stacks[v.bot]);
        let eff = v.stacks[v.bot].min(deepest);
        let list = if players == 2 { &self.hu } else { &self.six };
        let depth = Engine::pick(list, eff).ok_or("no strategies for this table size")?;
        let t = &depth.tree;
        let mut dbg = Debug {
            table: if players == 2 { "heads-up".into() } else { format!("6-max ({players} players)") },
            players,
            effective_bb: eff as f64 / 2.0,
            strategy_bb: depth.strategy.meta.stack as f64 / 2.0,
            street: street_name(s.street).into(),
            ..Debug::default()
        };
        let node = self.walk(v, t, &map, &back, missing, &mut dbg.path);
        let opps = (0..SEATS).filter(|&i| i != v.bot && matches!(s.players[i].status, Status::Active | Status::AllIn)).count();
        let node = match node {
            Ok(n) => n,
            Err(why) => {
                dbg.fallback = Some(why);
                let eq = equity_vs_random(v.hole, &s.community, opps.min(5), hand_seed(v.seed, 1000 + s.actions.len()));
                let action = fallback_action(eq, opps, &legal, &s);
                dbg.chosen = format!("{} (equity {:.0}% against {opps} random hand{})", action_label(&action), eq * 100.0, if opps == 1 { "" } else { "s" });
                return Ok(Decision { action, debug: dbg });
            }
        };
        let nd = &t.nodes[node as usize];
        let me = &s.players[v.bot];
        let pot_total = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
        let pac = pot_total + (s.current_bet - me.bet);
        let street = street_index(s.street);
        let bucket = self.abs.bucket(street, v.hole, &s.community);
        let n = nd.n as usize;
        let mut p = [0f64; MAX_ACTIONS];
        depth.strategy.probs(t, node, bucket, &mut p);
        let acts = t.actions(nd);
        let mut dist: Vec<(Action, f64)> = Vec::new();
        for a in 0..n {
            let real = to_real(&acts[a], &legal, s.current_bet, pac);
            dbg.options.push(Opt { label: acts[a].label(), action: action_label(&real), prob: p[a] });
            match dist.iter_mut().find(|m| m.0 == real) {
                Some(m) => m.1 += p[a],
                None => dist.push((real, p[a])),
            }
        }
        let total: f64 = dist.iter().map(|m| m.1).sum();
        let mut x = rng.gen::<f64>() * total;
        let mut action = dist.last().unwrap().0;
        for (a, w) in &dist {
            if x < *w {
                action = *a;
                break;
            }
            x -= w;
        }
        dbg.bucket = Some(bucket);
        dbg.bucket_label = Some(if street == 0 {
            crate::classes::class_name(bucket)
        } else {
            format!("hand group {} of {}", bucket, self.abs.buckets()[street])
        });
        dbg.node = Some(node);
        dbg.chosen = action_label(&action);
        Ok(Decision { action, debug: dbg })
    }

    /// The tree's node at the bot's decision, walking the real history; or
    /// why the tree can't follow it.
    fn walk(&self, v: &View, t: &Tree, map: &[Option<usize>; SEATS], back: &[usize; SEATS], missing: usize, path: &mut Vec<String>) -> Result<u32, String> {
        let mut node = 0u32;
        for k in 0..missing {
            let nd = &t.nodes[node as usize];
            debug_assert_eq!(nd.player as usize, 3 + k);
            let f = t.actions(nd).iter().position(|a| *a == Act::Fold).ok_or("no fold for an empty seat")?;
            node = t.children(nd)[f];
        }
        let mut s = start(v)?;
        for (i, a) in v.actions.iter().enumerate() {
            while s.phase == Phase::StreetComplete {
                s.advance_street();
            }
            skip_real_allins(t, &mut node, &s, back, path);
            let nd = &t.nodes[node as usize];
            if nd.player == TERMINAL {
                return Err("the abstract hand ended before the real one".into());
            }
            if Some(nd.player as usize) != map[a.seat] || nd.street as usize != street_index(s.street) {
                return Err("the abstract and real betting diverged".into());
            }
            step(&mut s, a, i)?;
            let ha = s.actions.last().unwrap();
            let acts = t.actions(nd);
            let mut trng = SmallRng::seed_from_u64(hand_seed(v.seed, i));
            let k = translate(acts, ha, &mut trng);
            path.push(format!("{}: {}", position_name(t.players, nd.player as usize), acts[k].label()));
            node = t.children(nd)[k];
        }
        while s.phase == Phase::StreetComplete {
            s.advance_street();
        }
        skip_real_allins(t, &mut node, &s, back, path);
        let nd = &t.nodes[node as usize];
        if nd.player == TERMINAL {
            return Err("the abstract hand already ended".into());
        }
        if Some(nd.player as usize) != map[v.bot] || nd.street as usize != street_index(s.street) {
            return Err("the abstract and real turns diverged".into());
        }
        Ok(node)
    }
}
