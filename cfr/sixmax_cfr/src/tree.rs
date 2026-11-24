//! The abstract betting tree for one table size and stack depth.
//!
//! Built by driving the real rules (`poker_core::rules`, the port of the
//! app's game.ts) from a fresh hand with every player at `stack` chips: at
//! every decision the legal actions come from the rules engine and the menu's
//! sizes are turned into legal raise-to amounts (see `menu`). So the tree
//! follows the app's betting rules exactly: blinds 1/2, min-raise by
//! increment, short all-ins that don't reopen the betting.
//!
//! Seats are the rules engine's: the first hand puts the button on seat 0.
//! - 6 players: 0 = BTN, 1 = SB, 2 = BB, 3 = UTG, 4 = MP, 5 = CO. A 5-, 4- or
//!   3-handed table is this tree with the early positions folded (UTG, then
//!   MP, then CO): same blinds, same order, same stacks.
//! - 2 players (heads-up): 0 = BTN/SB, 1 = BB; seats 2-5 sit out.
//!
//! A player's information set at a decision node is (node, their card group
//! on the node's street); regrets live in one flat array indexed by
//! `node.base + group * n + action`.

use poker_core::rules::{Action, ActionType, Phase, State, Status, Street, SEATS};

use crate::menu::Menu;

pub const MAX_ACTIONS: usize = 8;
/// `Node::player` at a terminal.
pub const TERMINAL: u8 = u8::MAX;

/// An abstract action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Act {
    Fold,
    Check,
    Call,
    /// Opening bet to `to` chips; `frac` = bet / pot.
    Bet { to: u16, frac: f32 },
    /// Raise to `to` chips; `frac` = (to - current bet) / pot after calling.
    Raise { to: u16, frac: f32 },
    /// All-in for `to` chips (a bet, raise, or a call for less).
    AllIn { to: u16, frac: f32 },
}

impl Act {
    pub fn label(&self) -> String {
        match *self {
            Act::Fold => "fold".into(),
            Act::Check => "check".into(),
            Act::Call => "call".into(),
            Act::Bet { to, frac } => format!("bet {to} ({:.0}% pot)", frac * 100.0),
            Act::Raise { to, frac } => format!("raise to {to} ({:.0}% pot)", frac * 100.0),
            Act::AllIn { to, .. } => format!("all-in {to}"),
        }
    }

    /// Pot fraction of a bet/raise/all-in (None for fold/check/call).
    pub fn frac(&self) -> Option<f32> {
        match *self {
            Act::Bet { frac, .. } | Act::Raise { frac, .. } | Act::AllIn { frac, .. } => Some(frac),
            _ => None,
        }
    }

    pub fn is_aggressive(&self) -> bool {
        self.frac().is_some()
    }

    /// The rules engine's action for this abstract action.
    pub fn to_rules(&self) -> Action {
        match *self {
            Act::Fold => Action::new(ActionType::Fold),
            Act::Check => Action::new(ActionType::Check),
            Act::Call => Action::new(ActionType::Call),
            Act::Bet { to, .. } => Action::to(ActionType::Bet, to as i32),
            Act::Raise { to, .. } => Action::to(ActionType::Raise, to as i32),
            Act::AllIn { .. } => Action::new(ActionType::AllIn),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Node {
    /// Offset of (group 0, action 0) in the regret/strategy arrays.
    pub base: u64,
    /// First child in `Tree::child`/`Tree::acts` (decisions), or the index
    /// in `Tree::terms` (terminals).
    pub first: u32,
    /// The actor's chips in the pot before acting (what folding loses).
    pub commit: u16,
    /// Seat to act, or TERMINAL.
    pub player: u8,
    pub street: u8,
    pub n: u8,
    /// Action 0 is fold.
    pub has_fold: bool,
}

#[derive(Clone, Copy, Debug)]
pub struct Term {
    /// Each seat's chips put in over the hand.
    pub contrib: [u16; SEATS],
    /// Bit i: seat i is still in the hand (not folded, not sitting out).
    pub live: u8,
}

pub struct Tree {
    /// 6 or 2.
    pub players: usize,
    pub stack: i32,
    pub menu: Menu,
    pub nodes: Vec<Node>,
    pub child: Vec<u32>,
    pub acts: Vec<Act>,
    pub terms: Vec<Term>,
    /// Card groups per street (information-set granularity).
    pub buckets: [u32; 4],
    /// Total (information set, action) entries.
    pub n_entries: u64,
}

struct Builder<'a> {
    t: Tree,
    menu: &'a Menu,
}

fn street_index(s: Street) -> usize {
    match s {
        Street::Preflop => 0,
        Street::Flop => 1,
        Street::Turn => 2,
        Street::River => 3,
    }
}

/// A fresh hand with `players` seats (6, or 2 for heads-up) at `stack` chips.
pub fn start_state(players: usize, stack: i32) -> State {
    assert!(players == 6 || players == 2, "tables of 6 or 2 players");
    let mut stacks = [0; SEATS];
    for s in stacks.iter_mut().take(players) {
        *s = stack;
    }
    let mut s = State::new_table(stacks, None);
    s.start_hand((0..52).collect());
    assert_eq!(s.phase, Phase::Betting);
    s
}

pub fn build(players: usize, stack: i32, menu: &Menu, buckets: [u32; 4]) -> Tree {
    assert!(stack > 2, "stack {stack} too short for a betting tree");
    let t = Tree {
        players,
        stack,
        menu: menu.clone(),
        nodes: Vec::new(),
        child: Vec::new(),
        acts: Vec::new(),
        terms: Vec::new(),
        buckets,
        n_entries: 0,
    };
    let mut b = Builder { t, menu };
    let s = start_state(players, stack);
    let root = b.decision(s, 0);
    assert_eq!(root, 0);
    let mut t = b.t;
    let mut off = 0u64;
    for nd in t.nodes.iter_mut() {
        if nd.player != TERMINAL {
            nd.base = off;
            off += t.buckets[nd.street as usize] as u64 * nd.n as u64;
        }
    }
    t.n_entries = off;
    t
}

impl Builder<'_> {
    fn decision(&mut self, s: State, raises: u32) -> u32 {
        let legal = s.legal().expect("a decision");
        let i = s.to_act as usize;
        let street = street_index(s.street);
        let in_hand = s.players.iter().filter(|p| matches!(p.status, Status::Active | Status::AllIn)).count();
        let others_can_act = s.players.iter().enumerate().any(|(j, p)| j != i && p.status == Status::Active);
        let mut opts = [Act::Fold; MAX_ACTIONS];
        let mut n = 0;
        let mut push = |a: Act| {
            opts[n] = a;
            n += 1;
        };
        if legal.can_fold {
            push(Act::Fold);
        }
        push(if legal.can_check { Act::Check } else { Act::Call });
        if legal.can_raise && others_can_act {
            let pot_total = s.pot + s.players.iter().map(|p| p.bet).sum::<i32>();
            let pac = pot_total + legal.to_call;
            let frac = |to: i32| (to - s.current_bet) as f32 / pac as f32;
            if raises < self.menu.cap(street, in_hand) {
                let mut seen = [0i32; MAX_ACTIONS];
                let mut ns = 0;
                for &f in self.menu.sizes(street, raises, in_hand) {
                    let to = ((s.current_bet as f64 + f * pac as f64).round_ties_even() as i32).max(legal.min_raise_to);
                    if to >= legal.max_raise_to || seen[..ns].contains(&to) {
                        continue;
                    }
                    seen[ns] = to;
                    ns += 1;
                    push(if legal.opening {
                        Act::Bet { to: to as u16, frac: frac(to) }
                    } else {
                        Act::Raise { to: to as u16, frac: frac(to) }
                    });
                }
            }
            let to = legal.max_raise_to;
            push(Act::AllIn { to: to as u16, frac: frac(to) });
        }
        let id = self.t.nodes.len() as u32;
        self.t.nodes.push(Node {
            base: 0,
            first: 0,
            commit: s.players[i].commit as u16,
            player: i as u8,
            street: street as u8,
            n: n as u8,
            has_fold: legal.can_fold,
        });
        let mut kids = [0u32; MAX_ACTIONS];
        for k in 0..n {
            let mut s2 = s.clone();
            s2.apply(opts[k].to_rules());
            let r2 = if opts[k].is_aggressive() { raises + 1 } else { raises };
            kids[k] = self.after(s2, r2);
        }
        let first = self.t.child.len() as u32;
        self.t.child.extend_from_slice(&kids[..n]);
        self.t.acts.extend_from_slice(&opts[..n]);
        self.t.nodes[id as usize].first = first;
        id
    }

    /// The node reached after an action: the next decision (possibly on a
    /// later street) or a terminal.
    fn after(&mut self, mut s: State, mut raises: u32) -> u32 {
        loop {
            match s.phase {
                Phase::Betting => return self.decision(s, raises),
                Phase::StreetComplete => {
                    s.advance_street();
                    raises = 0;
                    if s.hand_done() {
                        return self.terminal(&s);
                    }
                }
                other => unreachable!("unexpected phase {other:?}"),
            }
        }
    }

    fn terminal(&mut self, s: &State) -> u32 {
        let mut contrib = [0u16; SEATS];
        let mut live = 0u8;
        for (i, p) in s.players.iter().enumerate() {
            contrib[i] = p.commit as u16;
            if matches!(p.status, Status::Active | Status::AllIn) {
                live |= 1 << i;
            }
        }
        self.t.terms.push(Term { contrib, live });
        let street = street_index(s.street) as u8;
        self.t.nodes.push(Node {
            base: 0,
            first: (self.t.terms.len() - 1) as u32,
            commit: 0,
            player: TERMINAL,
            street,
            n: 0,
            has_fold: false,
        });
        (self.t.nodes.len() - 1) as u32
    }
}

impl Tree {
    #[inline]
    pub fn children(&self, nd: &Node) -> &[u32] {
        &self.child[nd.first as usize..nd.first as usize + nd.n as usize]
    }

    #[inline]
    pub fn actions(&self, nd: &Node) -> &[Act] {
        &self.acts[nd.first as usize..nd.first as usize + nd.n as usize]
    }

    /// Chips won by seat `p` at a terminal, given each seat's hand strength
    /// (higher is better; only compared among live seats). Pots are layered
    /// by contribution (side pots); a split layer is shared equally (the
    /// odd-chip rule is ignored: an exact-expectation simplification).
    #[inline]
    pub fn utility(&self, t: &Term, p: usize, rank: &[u32; SEATS]) -> f64 {
        let live = t.live;
        if live.count_ones() == 1 {
            let w = live.trailing_zeros() as usize;
            let total: i32 = t.contrib.iter().map(|&x| x as i32).sum();
            return if p == w { (total - t.contrib[p] as i32) as f64 } else { -(t.contrib[p] as f64) };
        }
        let mut levels = [0i32; SEATS];
        let mut nl = 0;
        for i in 0..SEATS {
            if live & (1 << i) != 0 {
                levels[nl] = t.contrib[i] as i32;
                nl += 1;
            }
        }
        levels[..nl].sort_unstable();
        let (mut prev, mut win) = (0i32, 0.0f64);
        for k in 0..nl {
            let l = levels[k];
            if l == prev {
                continue;
            }
            // Every contribution above the top live level belongs to the
            // last layer (a folded player can't have put in more, but be safe).
            let top = k == nl - 1;
            let mut layer = 0;
            for i in 0..SEATS {
                let c = t.contrib[i] as i32;
                layer += if top { c - c.min(prev) } else { c.min(l) - c.min(prev) };
            }
            let (mut best, mut nw, mut p_in) = (0u32, 0u32, false);
            for i in 0..SEATS {
                if live & (1 << i) != 0 && t.contrib[i] as i32 >= l {
                    let r = rank[i];
                    if nw == 0 || r > best {
                        best = r;
                        nw = 1;
                        p_in = i == p;
                    } else if r == best {
                        nw += 1;
                        p_in |= i == p;
                    }
                }
            }
            if p_in {
                win += layer as f64 / nw as f64;
            }
            prev = l;
        }
        win - t.contrib[p] as f64
    }

    /// Decision nodes per street.
    pub fn street_counts(&self) -> [u64; 4] {
        let mut c = [0u64; 4];
        for nd in &self.nodes {
            if nd.player != TERMINAL {
                c[nd.street as usize] += 1;
            }
        }
        c
    }

    /// Actions summed over decision nodes, per street (entries per card group).
    pub fn street_actions(&self) -> [u64; 4] {
        let mut c = [0u64; 4];
        for nd in &self.nodes {
            if nd.player != TERMINAL {
                c[nd.street as usize] += nd.n as u64;
            }
        }
        c
    }

    /// The seats that play in this tree.
    pub fn seats(&self) -> std::ops::Range<usize> {
        0..self.players
    }
}

/// The name of a seat's position.
pub fn position_name(players: usize, seat: usize) -> &'static str {
    match (players, seat) {
        (2, 0) => "BTN/SB",
        (2, 1) => "BB",
        (_, 0) => "BTN",
        (_, 1) => "SB",
        (_, 2) => "BB",
        (_, 3) => "UTG",
        (_, 4) => "MP",
        (_, 5) => "CO",
        _ => "?",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn tree(players: usize, stack: i32) -> Tree {
        build(players, stack, &Menu::v1(), [169, 10, 10, 10])
    }

    /// Walk random paths through the tree and the rules engine side by side:
    /// every node's actor is the rules engine's, every abstract action is
    /// legal there and lands where the tree says, and every terminal's
    /// contributions are the rules engine's.
    #[test]
    fn tree_follows_the_rules_engine() {
        let mut rng = SmallRng::seed_from_u64(1);
        for (players, stack) in [(6, 200), (6, 50), (6, 12), (2, 200), (2, 24)] {
            let t = tree(players, stack);
            for _ in 0..3000 {
                let mut s = start_state(players, stack);
                let mut node = 0u32;
                loop {
                    let nd = t.nodes[node as usize];
                    if nd.player == TERMINAL {
                        while !s.hand_done() {
                            assert_eq!(s.phase, Phase::StreetComplete);
                            s.advance_street();
                        }
                        let term = t.terms[nd.first as usize];
                        for (i, p) in s.players.iter().enumerate() {
                            assert_eq!(term.contrib[i] as i32, p.commit);
                        }
                        break;
                    }
                    while s.phase == Phase::StreetComplete {
                        s.advance_street();
                    }
                    assert_eq!(s.phase, Phase::Betting);
                    assert_eq!(s.to_act, nd.player as i32);
                    assert_eq!(s.players[s.to_act as usize].commit, nd.commit as i32);
                    let legal = s.legal().unwrap();
                    let k = rng.gen_range(0..nd.n as usize);
                    let a = t.actions(&nd)[k];
                    match a {
                        Act::Fold => assert!(legal.can_fold),
                        Act::Check => assert!(legal.can_check),
                        Act::Call => assert!(legal.can_call),
                        Act::Bet { to, .. } | Act::Raise { to, .. } => {
                            assert!(legal.can_raise);
                            assert!((to as i32) >= legal.min_raise_to && (to as i32) < legal.max_raise_to);
                        }
                        Act::AllIn { to, .. } => assert_eq!(to as i32, legal.max_raise_to),
                    }
                    let i = s.to_act as usize;
                    s.apply(a.to_rules());
                    if let Act::Bet { to, .. } | Act::Raise { to, .. } | Act::AllIn { to, .. } = a {
                        assert_eq!(s.players[i].bet, to as i32);
                    }
                    node = t.children(&nd)[k];
                }
            }
        }
    }

    #[test]
    fn payouts_are_zero_sum() {
        let mut rng = SmallRng::seed_from_u64(2);
        for (players, stack) in [(6, 200), (6, 24), (2, 200)] {
            let t = tree(players, stack);
            for _ in 0..200_000 {
                let term = &t.terms[rng.gen_range(0..t.terms.len())];
                let mut rank = [0u32; SEATS];
                for r in rank.iter_mut() {
                    *r = rng.gen_range(0..4); // plenty of ties
                }
                let sum: f64 = (0..SEATS).map(|p| t.utility(term, p, &rank)).sum();
                assert!(sum.abs() < 1e-9, "not zero-sum: {sum}");
                for p in 0..SEATS {
                    let u = t.utility(term, p, &rank);
                    assert!(u >= -(stack as f64) - 1e-9);
                }
            }
        }
    }

    /// 5-, 4- and 3-handed play are the 6-handed tree after UTG, MP and CO fold.
    #[test]
    fn short_tables_are_subtrees() {
        let t = tree(6, 200);
        let mut node = 0u32;
        for expect in [3u8, 4, 5, 0, 1] {
            let nd = t.nodes[node as usize];
            assert_eq!(nd.player, expect);
            assert!(nd.has_fold);
            node = t.children(&nd)[0];
        }
        let hu = tree(2, 200);
        assert_eq!(hu.nodes[0].player, 0, "heads-up: the button (small blind) acts first preflop");
    }

    #[test]
    fn opens_from_utg() {
        let t = tree(6, 200);
        let labels: Vec<String> = t.actions(&t.nodes[0]).iter().map(|a| a.label()).collect();
        assert_eq!(labels, ["fold", "call", "raise to 5 (60% pot)", "raise to 7 (100% pot)", "all-in 200"]);
    }
}
