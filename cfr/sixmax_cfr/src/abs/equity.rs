//! Exact river equities for every hand on a 5-card board at once.
//!
//! Evaluate the 1,081 possible hands once, sort by strength, and sweep: for a
//! hand (a, b), the opponent hands it beats are the hands below it minus those
//! holding a or b (card removal), and likewise for ties within its strength
//! group. No hand other than (a, b) itself holds both a and b, which makes the
//! inclusion–exclusion exact.
//!
//! - `river_ehs`: equity against a uniformly random opponent hand (EHS).
//! - `river_ochs`: equity against each of 8 opponent clusters (OCHS, Johanson
//!   et al. 2013), the river feature for the river hand groups.

use poker_core::cards::Card;
use poker_core::hand_eval::eval7;

use super::index::{hole_index, N_HOLES};

pub const N_OCHS: usize = 8;

/// Per-thread scratch for the sweeps. After a sweep, `hands` lists the
/// hands that were scored: (strength, hole index, low card, high card).
#[derive(Default)]
pub struct Scratch {
    pub hands: Vec<(u32, u16, u8, u8)>,
}

fn evaluate(board: &[Card; 5], sc: &mut Scratch) {
    let mut dead = 0u64;
    for &c in board {
        dead |= 1 << c;
    }
    sc.hands.clear();
    let mut cards = [0u8; 7];
    cards[2..].copy_from_slice(board);
    for b in 1..52u8 {
        if dead & (1 << b) != 0 {
            continue;
        }
        cards[1] = b;
        for a in 0..b {
            if dead & (1 << a) != 0 {
                continue;
            }
            cards[0] = a;
            sc.hands.push((eval7(&cards), hole_index(a, b) as u16, a, b));
        }
    }
    sc.hands.sort_unstable_by_key(|h| h.0);
}

/// EHS of every hand on `board`, written to `out[hole_index]` (entries for
/// hands that touch the board are left untouched).
pub fn river_ehs(board: &[Card; 5], sc: &mut Scratch, out: &mut [f32; N_HOLES]) {
    evaluate(board, sc);
    // 52 - 5 board - 2 own cards = 45 cards left: C(45, 2) opponent hands.
    const OPP: f32 = 990.0;
    let hands = &sc.hands;
    let n = hands.len();
    let mut below = 0u32;
    let mut below_card = [0u32; 52];
    let mut group_card = [0u32; 52];
    let mut i = 0;
    while i < n {
        let s = hands[i].0;
        let mut j = i;
        while j < n && hands[j].0 == s {
            group_card[hands[j].2 as usize] += 1;
            group_card[hands[j].3 as usize] += 1;
            j += 1;
        }
        let g = (j - i) as u32;
        for &(_, h, a, b) in &hands[i..j] {
            let (a, b) = (a as usize, b as usize);
            let wins = below - below_card[a] - below_card[b];
            let ties = g + 1 - group_card[a] - group_card[b];
            out[h as usize] = (wins as f32 + 0.5 * ties as f32) / OPP;
        }
        for &(_, _, a, b) in &hands[i..j] {
            let (a, b) = (a as usize, b as usize);
            group_card[a] = 0;
            group_card[b] = 0;
            below_card[a] += 1;
            below_card[b] += 1;
        }
        below += g;
        i = j;
    }
}

/// OCHS of every hand on `board`: equity against each opponent cluster
/// (`cluster[class]` gives a preflop class's cluster), written to
/// `out[hole_index]`. A cluster with no live hands scores 0.5.
pub fn river_ochs(board: &[Card; 5], cluster: &[u8; 169], sc: &mut Scratch, out: &mut [[f32; N_OCHS]; N_HOLES]) {
    evaluate(board, sc);
    let hands = &sc.hands;
    let n = hands.len();
    let cl = |a: u8, b: u8| cluster[crate::classes::class_of(a, b) as usize] as usize;

    let mut total = [0u32; N_OCHS];
    let mut total_card = [[0u32; N_OCHS]; 52];
    for &(_, _, a, b) in hands {
        let k = cl(a, b);
        total[k] += 1;
        total_card[a as usize][k] += 1;
        total_card[b as usize][k] += 1;
    }
    let mut below = [0u32; N_OCHS];
    let mut below_card = [[0u32; N_OCHS]; 52];
    let mut group = [0u32; N_OCHS];
    let mut group_card = [[0u32; N_OCHS]; 52];
    let mut i = 0;
    while i < n {
        let s = hands[i].0;
        let mut j = i;
        while j < n && hands[j].0 == s {
            let (a, b) = (hands[j].2, hands[j].3);
            let k = cl(a, b);
            group[k] += 1;
            group_card[a as usize][k] += 1;
            group_card[b as usize][k] += 1;
            j += 1;
        }
        for &(_, h, a, b) in &hands[i..j] {
            let own = cl(a, b);
            let (a, b) = (a as usize, b as usize);
            for k in 0..N_OCHS {
                let both = u32::from(k == own); // the hand itself holds a and b
                let opp = total[k] + both - total_card[a][k] - total_card[b][k];
                let wins = below[k] - below_card[a][k] - below_card[b][k];
                let ties = group[k] + both - group_card[a][k] - group_card[b][k];
                out[h as usize][k] = if opp == 0 { 0.5 } else { (wins as f32 + 0.5 * ties as f32) / opp as f32 };
            }
        }
        for &(_, _, a, b) in &hands[i..j] {
            let k = cl(a, b);
            let (a, b) = (a as usize, b as usize);
            group[k] = 0;
            group_card[a][k] = 0;
            group_card[b][k] = 0;
            below[k] += 1;
            below_card[a][k] += 1;
            below_card[b][k] += 1;
        }
        i = j;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abs::index::hole_cards;
    use rand::rngs::SmallRng;
    use rand::{Rng, SeedableRng};

    fn brute(board: &[Card; 5], h: [Card; 2], keep: impl Fn(Card, Card) -> bool) -> f32 {
        let mut cards = [0u8; 7];
        cards[2..].copy_from_slice(board);
        cards[0] = h[0];
        cards[1] = h[1];
        let me = eval7(&cards);
        let (mut won, mut n) = (0.0, 0.0);
        for b in 1..52u8 {
            for a in 0..b {
                let used = board.contains(&a) || board.contains(&b) || h.contains(&a) || h.contains(&b);
                if used || !keep(a, b) {
                    continue;
                }
                cards[0] = a;
                cards[1] = b;
                let v = eval7(&cards);
                n += 1.0;
                won += if me > v { 1.0 } else if me == v { 0.5 } else { 0.0 };
            }
        }
        if n == 0.0 { 0.5 } else { won / n }
    }

    #[test]
    fn sweeps_match_brute_force() {
        let mut rng = SmallRng::seed_from_u64(3);
        let mut cluster = [0u8; 169];
        for (i, c) in cluster.iter_mut().enumerate() {
            *c = (i * 7 % N_OCHS) as u8;
        }
        let mut sc = Scratch::default();
        let mut ehs = Box::new([0f32; N_HOLES]);
        let mut ochs = Box::new([[0f32; N_OCHS]; N_HOLES]);
        for _ in 0..4 {
            let mut board = [0u8; 5];
            let mut used = 0u64;
            for slot in board.iter_mut() {
                loop {
                    let c = rng.gen_range(0..52u8);
                    if used & (1 << c) == 0 {
                        used |= 1 << c;
                        *slot = c;
                        break;
                    }
                }
            }
            river_ehs(&board, &mut sc, &mut ehs);
            river_ochs(&board, &cluster, &mut sc, &mut ochs);
            for h in (0..N_HOLES as u32).step_by(13) {
                let hc = hole_cards(h);
                if board.contains(&hc[0]) || board.contains(&hc[1]) {
                    continue;
                }
                let e = brute(&board, hc, |_, _| true);
                assert!((ehs[h as usize] - e).abs() < 1e-5, "ehs {} vs {e}", ehs[h as usize]);
                for k in 0..N_OCHS {
                    let ek = brute(&board, hc, |a, b| cluster[crate::classes::class_of(a, b) as usize] as usize == k);
                    assert!((ochs[h as usize][k] - ek).abs() < 1e-5, "ochs[{k}] {} vs {ek}", ochs[h as usize][k]);
                }
            }
        }
    }
}
