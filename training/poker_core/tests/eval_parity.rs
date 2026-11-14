//! Hand-evaluator parity: `hand_eval::eval7` must order random 7-card hands
//! exactly as `evaluator.ts` (`evaluate7` + `compareScores`) does.

mod common;

use std::cmp::Ordering;

use poker_core::cards::parse;
use poker_core::hand_eval::{category, eval7};

fn compare_scores(a: &[i64], b: &[i64]) -> Ordering {
    let n = a.len().max(b.len());
    for i in 0..n {
        let x = a.get(i).copied().unwrap_or(0);
        let y = b.get(i).copied().unwrap_or(0);
        if x != y {
            return x.cmp(&y);
        }
    }
    Ordering::Equal
}

#[test]
fn eval7_orders_like_typescript() {
    let fx = common::fixture("eval_hands.json");
    let hands: Vec<(u32, Vec<i64>)> = fx["hands"]
        .as_array()
        .unwrap()
        .iter()
        .map(|h| {
            let cards: Vec<u8> =
                h["cards"].as_array().unwrap().iter().map(|c| parse(c.as_str().unwrap()).unwrap()).collect();
            let score: Vec<i64> = h["score"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
            (eval7(&cards), score)
        })
        .collect();
    assert!(hands.len() > 1000);

    for (strength, score) in &hands {
        assert_eq!(category(*strength) as i64, score[0], "category mismatch for score {score:?}");
    }
    // Ordering on every adjacent pair plus a strided set of far pairs.
    let n = hands.len();
    for i in 0..n {
        for j in [i + 1, i * 7 + 3] {
            let j = j % n;
            let (sa, ta) = &hands[i];
            let (sb, tb) = &hands[j];
            assert_eq!(sa.cmp(sb), compare_scores(ta, tb), "order mismatch: {ta:?} vs {tb:?}");
        }
    }
}
