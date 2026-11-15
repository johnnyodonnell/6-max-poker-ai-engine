//! Encoder + action-space parity: replay the TS traces and, at every decision
//! that carries an `enc` snapshot, require the Rust encoding, action space and
//! size->raise-to mapping to match `encode.ts` / `actions.ts`.

mod common;

use poker_core::actions::NUM_ACTIONS;
use poker_core::cards::parse;
use poker_core::encode::{encode, INPUT_SIZE};
use poker_core::rules::{Action, ActionType, State};
use serde_json::Value;

fn parse_kind(s: &str) -> ActionType {
    match s {
        "fold" => ActionType::Fold,
        "check" => ActionType::Check,
        "call" => ActionType::Call,
        "bet" => ActionType::Bet,
        "raise" => ActionType::Raise,
        "allin" => ActionType::AllIn,
        other => panic!("bad action type {other}"),
    }
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

#[test]
fn encodings_match_typescript() {
    let fx = common::fixture("rules_traces.json");
    let mut checked = 0usize;
    for (mi, m) in fx["matches"].as_array().unwrap().iter().enumerate() {
        let mut s = State::new_table([200; 6], Some(0));
        for (ei, ev) in m["events"].as_array().unwrap().iter().enumerate() {
            match ev["kind"].as_str().unwrap() {
                "start" => {
                    let deck: Vec<u8> = ev["deck"]
                        .as_array()
                        .map(|d| d.iter().map(|c| parse(c.as_str().unwrap()).unwrap()).collect())
                        .unwrap_or_default();
                    s.start_hand(deck);
                }
                "advance" => s.advance_street(),
                "act" => {
                    if let Some(e) = ev.get("enc") {
                        let at = format!("match {mi} event {ei}");
                        let (x, sp) = encode(&s).expect("encode");
                        let mut want = vec![0f32; INPUT_SIZE];
                        for pair in e["enc"].as_array().unwrap() {
                            want[pair[0].as_u64().unwrap() as usize] = f(&pair[1]) as f32;
                        }
                        for i in 0..INPUT_SIZE {
                            assert!(
                                (x[i] - want[i]).abs() <= 1e-5 * (1.0 + want[i].abs()),
                                "{at}: x[{i}] = {} vs TS {}",
                                x[i],
                                want[i]
                            );
                        }
                        let w = &e["space"];
                        let mask: Vec<f32> = w["mask"].as_array().unwrap().iter().map(|v| f(v) as f32).collect();
                        assert_eq!(sp.mask.to_vec(), mask, "{at}: mask");
                        assert_eq!(sp.pot_after_call as f64, f(&w["potAfterCall"]), "{at}");
                        assert_eq!(sp.min_raise_to as f64, f(&w["minRaiseTo"]), "{at}");
                        assert_eq!(sp.max_raise_to as f64, f(&w["maxRaiseTo"]), "{at}");
                        assert!((sp.s_lo - f(&w["sLo"])).abs() < 1e-9, "{at}: sLo");
                        assert!((sp.s_hi - f(&w["sHi"])).abs() < 1e-9, "{at}: sHi");
                        let amask: Vec<f32> = w["amask"].as_array().unwrap().iter().map(|v| f(v) as f32).collect();
                        assert_eq!(sp.amask.to_vec(), amask, "{at}: amask");
                        let amounts: Vec<i32> = w["amounts"].as_array().unwrap().iter().map(|v| f(v) as i32).collect();
                        assert_eq!(sp.amounts.to_vec(), amounts, "{at}: amounts");
                        assert_eq!(amounts.len(), NUM_ACTIONS);
                        checked += 1;
                    }
                    let a = &ev["action"];
                    let kind = parse_kind(a["type"].as_str().unwrap());
                    let amount = a.get("amount").and_then(Value::as_i64).map(|v| v as i32);
                    s.apply(Action { kind, amount });
                }
                _ => {}
            }
        }
    }
    println!("checked {checked} encodings");
    assert!(checked > 1000, "only {checked} encodings checked");
}
