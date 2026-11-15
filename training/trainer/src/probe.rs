//! Poker-sense probes of a policy (run on every snapshot, and by `poker
//! probe --weights W`): does it know which hands are strong?
//!
//!   preflop  first-in decisions (UTG and button, 6 x 100bb) for all 169 hand
//!            classes: rank correlation of fold probability with equity, how
//!            much more often pocket pairs are folded than equal-equity
//!            unpaired hands, and how often AA raises rather than limps.
//!   flop     the button facing a ~half-pot lead after open / call, over
//!            random flops and hands: fold rates for sets, top pair and air,
//!            and the rank correlation of fold probability with equity.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde_json::{json, Value};
use tch::{Device, Kind, Tensor};

use poker_core::actions::{A_ALLIN, A_CALL, A_FOLD, NUM_ACTIONS};
use poker_core::cards::{full_deck, make, rank, Card};
use poker_core::encode::{encode_into, INPUT_SIZE};
use poker_core::equity::equity;
use poker_core::rules::{Action, ActionType, State, SEATS};

use crate::net::PokerNet;

const MASK_NEG: f64 = 1e9;

fn policy(net: &PokerNet, xs: &[f32], masks: &[f32], dev: Device) -> Vec<f32> {
    let n = (xs.len() / INPUT_SIZE) as i64;
    let x = Tensor::from_slice(xs).view([n, INPUT_SIZE as i64]).to_device(dev);
    let m = Tensor::from_slice(masks).view([n, NUM_ACTIONS as i64]).to_device(dev);
    let p = tch::no_grad(|| (net.forward(&x) + (m - 1.0) * MASK_NEG).softmax(1, Kind::Float));
    Vec::<f32>::try_from(&p.to_device(Device::Cpu).contiguous().view([-1])).unwrap()
}

fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let rank = |v: &[f64]| {
        let mut idx: Vec<usize> = (0..v.len()).collect();
        idx.sort_by(|&i, &j| v[i].partial_cmp(&v[j]).unwrap());
        let mut r = vec![0f64; v.len()];
        for (k, &i) in idx.iter().enumerate() {
            r[i] = k as f64;
        }
        r
    };
    let (ra, rb) = (rank(a), rank(b));
    let m = (a.len() - 1) as f64 / 2.0;
    let cov: f64 = ra.iter().zip(&rb).map(|(x, y)| (x - m) * (y - m)).sum();
    let var: f64 = ra.iter().map(|x| (x - m).powi(2)).sum();
    cov / var
}

fn fresh_table(rng: &mut StdRng) -> State {
    let mut st = State::new_table([200; SEATS], None);
    st.button = 5;
    let mut deck = full_deck();
    deck.shuffle(rng);
    st.start_hand(deck);
    st
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

pub fn run(net: &PokerNet, dev: Device) -> Value {
    let mut rng = StdRng::seed_from_u64(11);
    let mut classes = Vec::new();
    for hi in (2..=14u8).rev() {
        for lo in (2..=hi).rev() {
            classes.push((hi, lo, false));
            if hi != lo {
                classes.push((hi, lo, true));
            }
        }
    }
    let mut out = json!({});
    for (pos, nfold, nopp) in [("utg", 0usize, 5usize), ("btn", 3, 2)] {
        let mut st = fresh_table(&mut rng);
        for _ in 0..nfold {
            st.apply(Action::new(ActionType::Fold));
        }
        let seat = st.to_act as usize;
        let (mut xs, mut masks, mut eqs) = (Vec::new(), Vec::new(), Vec::new());
        for &(h, l, suited) in &classes {
            let hole = [make(h, 0), make(l, if suited { 0 } else { 1 })];
            let mut s2 = st.clone();
            s2.players[seat].hole = hole.to_vec();
            let mut x = vec![0f32; INPUT_SIZE];
            let sp = encode_into(&s2, &mut x).expect("decision");
            xs.extend_from_slice(&x);
            masks.extend_from_slice(&sp.amask);
            // Equity vs nopp random hands.
            let mut rest: Vec<Card> = (0..52u8).filter(|c| !hole.contains(c)).collect();
            let trials = 2000;
            let mut tot = 0.0;
            for _ in 0..trials {
                let (pick, _) = rest.partial_shuffle(&mut rng, 2 * nopp);
                let opps: Vec<&[Card]> = pick.chunks(2).collect();
                let opps: Vec<Vec<Card>> = opps.iter().map(|o| o.to_vec()).collect();
                let refs: Vec<&[Card]> = opps.iter().map(|o| o.as_slice()).collect();
                tot += equity(&hole, &refs, &[], &[], 1, &mut rng);
            }
            eqs.push(tot / trials as f64);
        }
        let p = policy(net, &xs, &masks, dev);
        let fold: Vec<f64> = (0..classes.len()).map(|i| p[i * NUM_ACTIONS + A_FOLD] as f64).collect();
        let combos = |c: &(u8, u8, bool)| if c.0 == c.1 { 6.0 } else if c.2 { 4.0 } else { 12.0 };
        let w: f64 = classes.iter().map(combos).sum();
        let play: f64 = classes.iter().enumerate().map(|(i, c)| combos(c) * (1.0 - fold[i])).sum::<f64>() / w;
        // Pocket pairs vs unpaired hands within +-0.01 equity.
        let mut diffs = Vec::new();
        for (i, c) in classes.iter().enumerate().filter(|(_, c)| c.0 == c.1) {
            let near: Vec<f64> =
                classes.iter().enumerate().filter(|(j, d)| d.0 != d.1 && (eqs[*j] - eqs[i]).abs() <= 0.01).map(|(j, _)| fold[j]).collect();
            if !near.is_empty() && c.0 < 11 {
                diffs.push(fold[i] - near.iter().sum::<f64>() / near.len() as f64);
            }
        }
        let aa = classes.iter().position(|c| *c == (14, 14, false)).unwrap();
        let aa_raise = 1.0 - p[aa * NUM_ACTIONS + A_FOLD] - p[aa * NUM_ACTIONS + A_CALL];
        out[pos] = json!({
            "play": round2(play),
            "rho_fold_equity": round2(spearman(&fold, &eqs)),
            "pair_extra_fold": round2(diffs.iter().sum::<f64>() / diffs.len().max(1) as f64),
            "aa_raise": round2(aa_raise as f64),
            "aa_allin": round2(p[aa * NUM_ACTIONS + A_ALLIN] as f64),
        });
    }

    // Flop: BTN opens to 5, BB calls and leads 6 into 11.
    let mut st = fresh_table(&mut rng);
    for _ in 0..3 {
        st.apply(Action::new(ActionType::Fold));
    }
    st.apply(Action::to(ActionType::Raise, 5));
    st.apply(Action::new(ActionType::Fold));
    st.apply(Action::new(ActionType::Call));
    st.advance_street();
    st.apply(Action::to(ActionType::Bet, 6));
    let seat = st.to_act as usize;
    let (mut xs, mut masks, mut eqs, mut kind) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for _ in 0..300 {
        let mut d = full_deck();
        d.shuffle(&mut rng);
        let flop = [d[0], d[1], d[2]];
        for h in 0..24 {
            let hole = [d[3 + 2 * h], d[4 + 2 * h]];
            let mut s2 = st.clone();
            s2.community = flop.to_vec();
            s2.players[seat].hole = hole.to_vec();
            let mut x = vec![0f32; INPUT_SIZE];
            let sp = encode_into(&s2, &mut x).expect("decision");
            xs.extend_from_slice(&x);
            masks.extend_from_slice(&sp.amask);
            // Equity vs one random hand.
            let mut rest: Vec<Card> = (0..52u8).filter(|c| !hole.contains(c) && !flop.contains(c)).collect();
            let mut tot = 0.0;
            for _ in 0..100 {
                let (pick, _) = rest.partial_shuffle(&mut rng, 2);
                let opp = [pick[0], pick[1]];
                tot += equity(&hole, &[&opp], &flop, &[], 1, &mut rng);
            }
            eqs.push(tot / 100.0);
            let br: Vec<u8> = flop.iter().map(|&c| rank(c)).collect();
            let top = *br.iter().max().unwrap();
            let pocket = rank(hole[0]) == rank(hole[1]);
            let paired = hole.iter().any(|&c| br.contains(&rank(c)));
            kind.push(if pocket && br.contains(&rank(hole[0])) {
                0 // set
            } else if !pocket && hole.iter().any(|&c| rank(c) == top) {
                1 // top pair
            } else if !pocket && !paired {
                2 // unpaired
            } else {
                3
            });
        }
    }
    let p = policy(net, &xs, &masks, dev);
    let fold: Vec<f64> = (0..eqs.len()).map(|i| p[i * NUM_ACTIONS + A_FOLD] as f64).collect();
    let rate = |sel: &dyn Fn(usize) -> bool| {
        let v: Vec<f64> = (0..fold.len()).filter(|&i| sel(i)).map(|i| fold[i]).collect();
        round2(v.iter().sum::<f64>() / v.len().max(1) as f64)
    };
    out["flop"] = json!({
        "rho_fold_equity": round2(spearman(&fold, &eqs)),
        "fold_set": rate(&|i| kind[i] == 0),
        "fold_top_pair": rate(&|i| kind[i] == 1),
        "fold_air": rate(&|i| kind[i] == 2 && eqs[i] < 0.35),
        "fold_unpaired_strong": rate(&|i| kind[i] == 2 && eqs[i] > 0.6),
    });
    out
}
