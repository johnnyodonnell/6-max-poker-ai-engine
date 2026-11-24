//! Validation games for the solver: Kuhn poker (2 and 3 players) and Leduc
//! hold'em, built as ordinary `Tree`s (limit betting, antes) with every deal
//! enumerated, plus exact best responses for measuring how close a strategy
//! profile is to equilibrium.

use rand::rngs::SmallRng;
use rand::Rng;

use crate::deal::{Deal, Dealer};
use crate::menu::Menu;
use crate::tree::{Act, Node, Term, Tree, MAX_ACTIONS, TERMINAL};

/// A limit-betting game: `rounds` rounds with fixed bet sizes, at most
/// `cap` bets/raises per round, everyone antes `ante`.
struct Spec {
    players: usize,
    rounds: usize,
    bet: [i32; 2],
    cap: u32,
    ante: i32,
}

#[derive(Clone)]
struct St {
    street: usize,
    contrib: [i32; 6],
    round: [i32; 6],
    folded: [bool; 6],
    acted: [bool; 6],
    level: i32,
    raises: u32,
}

struct Builder<'a> {
    t: Tree,
    spec: &'a Spec,
}

impl Builder<'_> {
    fn decision(&mut self, s: St, i: usize) -> u32 {
        let to_call = s.level - s.round[i];
        let size = self.spec.bet[s.street];
        let mut opts: Vec<Act> = Vec::new();
        if to_call > 0 {
            opts.push(Act::Fold);
            opts.push(Act::Call);
            if s.raises < self.spec.cap {
                opts.push(Act::Raise { to: (s.level + size) as u16, frac: 0.0 });
            }
        } else {
            opts.push(Act::Check);
            if s.raises < self.spec.cap {
                opts.push(Act::Bet { to: size as u16, frac: 0.0 });
            }
        }
        let id = self.t.nodes.len() as u32;
        self.t.nodes.push(Node {
            base: 0,
            first: 0,
            commit: s.contrib[i] as u16,
            player: i as u8,
            street: s.street as u8,
            n: opts.len() as u8,
            has_fold: to_call > 0,
        });
        let mut kids = Vec::new();
        for a in &opts {
            let mut s2 = s.clone();
            s2.acted[i] = true;
            match *a {
                Act::Fold => s2.folded[i] = true,
                Act::Call => {
                    s2.contrib[i] += to_call;
                    s2.round[i] += to_call;
                }
                Act::Bet { to, .. } | Act::Raise { to, .. } => {
                    let add = to as i32 - s2.round[i];
                    s2.contrib[i] += add;
                    s2.round[i] = to as i32;
                    s2.level = to as i32;
                    s2.raises += 1;
                }
                _ => {}
            }
            kids.push(self.after(s2, i));
        }
        let first = self.t.child.len() as u32;
        self.t.child.extend_from_slice(&kids);
        self.t.acts.extend_from_slice(&opts);
        self.t.nodes[id as usize].first = first;
        id
    }

    fn after(&mut self, mut s: St, last: usize) -> u32 {
        let np = self.spec.players;
        let live = (0..np).filter(|&j| !s.folded[j]).count();
        if live == 1 {
            return self.terminal(&s);
        }
        for k in 1..=np {
            let j = (last + k) % np;
            if !s.folded[j] && (!s.acted[j] || s.round[j] < s.level) {
                return self.decision(s, j);
            }
        }
        if s.street + 1 == self.spec.rounds {
            return self.terminal(&s);
        }
        s.street += 1;
        s.round = [0; 6];
        s.acted = [false; 6];
        s.level = 0;
        s.raises = 0;
        let first = (0..np).find(|&j| !s.folded[j]).unwrap();
        self.decision(s, first)
    }

    fn terminal(&mut self, s: &St) -> u32 {
        let mut contrib = [0u16; 6];
        let mut live = 0u8;
        for j in 0..self.spec.players {
            contrib[j] = s.contrib[j] as u16;
            if !s.folded[j] {
                live |= 1 << j;
            }
        }
        self.t.terms.push(Term { contrib, live });
        self.t.nodes.push(Node {
            base: 0,
            first: (self.t.terms.len() - 1) as u32,
            commit: 0,
            player: TERMINAL,
            street: s.street as u8,
            n: 0,
            has_fold: false,
        });
        (self.t.nodes.len() - 1) as u32
    }
}

fn build(spec: &Spec, buckets: [u32; 4]) -> Tree {
    let t = Tree {
        players: spec.players,
        stack: 0,
        menu: Menu::v1(),
        nodes: Vec::new(),
        child: Vec::new(),
        acts: Vec::new(),
        terms: Vec::new(),
        buckets,
        n_entries: 0,
        street_start: [0; 5],
    };
    let mut b = Builder { t, spec };
    let mut contrib = [0; 6];
    for c in contrib.iter_mut().take(spec.players) {
        *c = spec.ante;
    }
    let s = St { street: 0, contrib, round: [0; 6], folded: [false; 6], acted: [false; 6], level: 0, raises: 0 };
    let root = b.decision(s, 0);
    assert_eq!(root, 0);
    let mut t = b.t;
    t.finalize();
    t
}

/// Every deal of a toy game with its probability.
pub type Deals = Vec<(Deal, f64)>;

/// Kuhn poker for `players` (2 or 3): a deck of players + 1 cards, one card
/// each, ante 1, one bet of 1, no raises.
pub fn kuhn(players: usize) -> (Tree, Deals) {
    let n = players + 1;
    let t = build(&Spec { players, rounds: 1, bet: [1, 1], cap: 1, ante: 1 }, [n as u32, 1, 1, 1]);
    let mut deals = Vec::new();
    let mut cards = vec![0usize; players];
    fn rec(k: usize, players: usize, n: usize, cards: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if k == players {
            out.push(cards.clone());
            return;
        }
        for c in 0..n {
            if !cards[..k].contains(&c) {
                cards[k] = c;
                rec(k + 1, players, n, cards, out);
            }
        }
    }
    let mut all = Vec::new();
    rec(0, players, n, &mut cards, &mut all);
    let w = 1.0 / all.len() as f64;
    for c in all {
        let mut d = Deal::default();
        for p in 0..players {
            d.bucket[p][0] = c[p] as u32;
            d.rank[p] = c[p] as u32;
        }
        deals.push((d, w));
    }
    (t, deals)
}

/// Leduc hold'em: six cards (three ranks, two suits), one private card each,
/// one public card dealt after the first round; ante 1, bets of 2 then 4,
/// at most two bets/raises per round. A pair with the public card beats any
/// high card.
pub fn leduc() -> (Tree, Deals) {
    let t = build(&Spec { players: 2, rounds: 2, bet: [2, 4], cap: 2, ante: 1 }, [3, 9, 1, 1]);
    let mut deals = Vec::new();
    for a in 0..6usize {
        for b in 0..6 {
            for pc in 0..6 {
                if a == b || a == pc || b == pc {
                    continue;
                }
                let (ra, rb, rp) = (a / 2, b / 2, pc / 2);
                let mut d = Deal::default();
                for (p, r) in [(0, ra), (1, rb)] {
                    d.bucket[p][0] = r as u32;
                    d.bucket[p][1] = (r * 3 + rp) as u32;
                    d.rank[p] = if r == rp { 10 + r as u32 } else { r as u32 };
                }
                deals.push((d, 1.0 / 120.0));
            }
        }
    }
    (t, deals)
}

/// Samples a toy game's deals uniformly (they are equally likely).
pub struct ToyDealer<'a>(pub &'a Deals);

impl Dealer for ToyDealer<'_> {
    fn deal(&self, rng: &mut SmallRng) -> Deal {
        self.0[rng.gen_range(0..self.0.len())].0
    }
}

/// A strategy profile: action probabilities at (node, group).
pub type Profile<'a> = dyn Fn(u32, u32, &mut [f64]) + 'a;

fn walk(t: &Tree, deals: &Deals, br: Option<usize>, p: usize, sigma: &Profile, node: u32, reach: &[f64]) -> Vec<f64> {
    let nd = &t.nodes[node as usize];
    if nd.player == TERMINAL {
        let term = &t.terms[nd.first as usize];
        return deals.iter().map(|(d, _)| t.utility(term, p, &d.rank)).collect();
    }
    let q = nd.player as usize;
    let n = nd.n as usize;
    let kids = t.children(nd);
    let mut probs = vec![[0f64; MAX_ACTIONS]; deals.len()];
    for (i, (d, _)) in deals.iter().enumerate() {
        sigma(node, d.bucket[q][nd.street as usize], &mut probs[i][..n]);
    }
    if br == Some(q) {
        let vals: Vec<Vec<f64>> = (0..n).map(|a| walk(t, deals, br, p, sigma, kids[a], reach)).collect();
        let groups = t.buckets[nd.street as usize] as usize;
        let mut score = vec![[0f64; MAX_ACTIONS]; groups];
        for (i, (d, _)) in deals.iter().enumerate() {
            let g = d.bucket[q][nd.street as usize] as usize;
            for a in 0..n {
                score[g][a] += reach[i] * vals[a][i];
            }
        }
        let best: Vec<usize> = score
            .iter()
            .map(|s| (0..n).max_by(|&x, &y| s[x].partial_cmp(&s[y]).unwrap()).unwrap())
            .collect();
        return deals
            .iter()
            .enumerate()
            .map(|(i, (d, _))| vals[best[d.bucket[q][nd.street as usize] as usize]][i])
            .collect();
    }
    let mut out = vec![0f64; deals.len()];
    for a in 0..n {
        let r2: Vec<f64> = reach.iter().zip(&probs).map(|(r, pr)| r * pr[a]).collect();
        let v = walk(t, deals, br, p, sigma, kids[a], &r2);
        for i in 0..deals.len() {
            out[i] += probs[i][a] * v[i];
        }
    }
    out
}

/// Each player's expected chips when everyone plays `sigma`.
pub fn values(t: &Tree, deals: &Deals, sigma: &Profile) -> Vec<f64> {
    let reach: Vec<f64> = deals.iter().map(|d| d.1).collect();
    (0..t.players)
        .map(|p| walk(t, deals, None, p, sigma, 0, &reach).iter().zip(&reach).map(|(v, r)| v * r).sum())
        .collect()
}

/// How much each player could gain by switching to a best response while
/// the others keep playing `sigma` (all zero exactly at an equilibrium).
pub fn gains(t: &Tree, deals: &Deals, sigma: &Profile) -> Vec<f64> {
    let reach: Vec<f64> = deals.iter().map(|d| d.1).collect();
    let v = values(t, deals, sigma);
    (0..t.players)
        .map(|p| {
            let br: f64 = walk(t, deals, Some(p), p, sigma, 0, &reach).iter().zip(&reach).map(|(v, r)| v * r).sum();
            br - v[p]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mccfr::{Config, Solver};

    fn solve(t: &Tree, deals: &Deals, iters: u64, threads: usize) -> (Solver, Vec<f64>, Vec<f64>) {
        let mut s = Solver::new(t, 4);
        let mut cfg = Config::plain(threads);
        cfg.discount_every = iters / 100;
        cfg.discount_until = iters / 2;
        s.train(t, &ToyDealer(deals), iters, iters / 20, &cfg, 7, |_, _, _| true);
        let sigma = |node: u32, b: u32, out: &mut [f64]| s.strategy(t, node, b, out);
        let g = gains(t, deals, &sigma);
        let v = values(t, deals, &sigma);
        (s, g, v)
    }

    #[test]
    fn uniform_kuhn_is_exploitable() {
        let (t, deals) = kuhn(2);
        let uniform = |node: u32, _b: u32, out: &mut [f64]| {
            let n = t.nodes[node as usize].n as usize;
            out[..n].fill(1.0 / n as f64);
        };
        let g = gains(&t, &deals, &uniform);
        assert!(g.iter().sum::<f64>() / 2.0 > 0.2, "{g:?}");
    }

    #[test]
    fn mccfr_solves_kuhn() {
        let (t, deals) = kuhn(2);
        let (_, g, v) = solve(&t, &deals, 16_000_000, 8);
        let expl = g.iter().sum::<f64>() / 2.0;
        println!("kuhn: exploitability {expl:.5}, value {:.5} (equilibrium -1/18 = {:.5})", v[0], -1.0 / 18.0);
        assert!(expl < 0.003, "kuhn exploitability {expl}");
        assert!((v[0] + 1.0 / 18.0).abs() < 0.01, "kuhn value {}", v[0]);
    }

    #[test]
    fn mccfr_solves_leduc() {
        let (t, deals) = leduc();
        let (_, g, v) = solve(&t, &deals, 8_000_000, 8);
        let expl = g.iter().sum::<f64>() / 2.0;
        println!("leduc: exploitability {expl:.5}, value {:.5} (equilibrium -0.0856)", v[0]);
        assert!(expl < 0.02, "leduc exploitability {expl}");
        assert!((v[0] + 0.0856).abs() < 0.02, "leduc value {}", v[0]);
    }

    /// Three players: no single player can gain much by deviating.
    #[test]
    fn mccfr_three_player_kuhn_approaches_equilibrium() {
        let (t, deals) = kuhn(3);
        let (_, g1, _) = solve(&t, &deals, 1_000_000, 8);
        let (_, g, v) = solve(&t, &deals, 16_000_000, 8);
        println!("3-player kuhn: gains {g1:.4?} at 1M -> {g:.4?} at 16M, values {v:.4?}");
        assert!(g.iter().all(|&x| x < 0.01), "3-player kuhn gains {g:?}");
        assert!((v.iter().sum::<f64>()).abs() < 1e-9);
    }

    #[test]
    fn checkpoint_round_trip() {
        let (t, deals) = leduc();
        let (s, _, _) = solve(&t, &deals, 200_000, 2);
        let dir = std::env::temp_dir().join(format!("sixmax-ckpt-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("solver.ckpt");
        s.save(&path, 2).unwrap();
        let r = Solver::load(&t, &path).unwrap();
        assert_eq!(r.iterations, s.iterations);
        assert_eq!(r.discounts, s.discounts);
        for (a, b) in s.regret.iter().zip(r.regret.iter()) {
            assert_eq!(a.load(std::sync::atomic::Ordering::Relaxed), b.load(std::sync::atomic::Ordering::Relaxed));
        }
        for (a, b) in s.avg.iter().zip(r.avg.iter()) {
            assert_eq!(a.load(std::sync::atomic::Ordering::Relaxed), b.load(std::sync::atomic::Ordering::Relaxed));
        }
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
