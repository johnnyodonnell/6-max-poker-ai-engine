//! Rules parity: replay full random matches recorded from the TS engine
//! (`dump_fixtures.ts` -> rules_traces.json) and require the Rust port to
//! reproduce every intermediate state, every legal-action set, every pot
//! result, and every hand's action history exactly.

mod common;

use poker_core::cards::{id, parse};
use poker_core::rules::{Action, ActionType, Phase, State, Status, Street};
use serde_json::{json, Value};

fn phase_str(p: Phase) -> &'static str {
    match p {
        Phase::Waiting => "waiting",
        Phase::Betting => "betting",
        Phase::StreetComplete => "street-complete",
        Phase::HandOver => "hand-over",
        Phase::MatchOver => "match-over",
    }
}

fn street_str(s: Street) -> &'static str {
    match s {
        Street::Preflop => "preflop",
        Street::Flop => "flop",
        Street::Turn => "turn",
        Street::River => "river",
    }
}

fn status_str(s: Status) -> &'static str {
    match s {
        Status::Active => "active",
        Status::AllIn => "allin",
        Status::Folded => "folded",
        Status::Out => "out",
    }
}

fn kind_str(k: ActionType) -> &'static str {
    match k {
        ActionType::Fold => "fold",
        ActionType::Check => "check",
        ActionType::Call => "call",
        ActionType::Bet => "bet",
        ActionType::Raise => "raise",
        ActionType::AllIn => "allin",
    }
}

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

/// Same shape as `summary()` in dump_fixtures.ts.
fn summary(s: &State) -> Value {
    json!({
        "phase": phase_str(s.phase),
        "street": street_str(s.street),
        "toAct": s.to_act,
        "pot": s.pot,
        "currentBet": s.current_bet,
        "minRaise": s.min_raise,
        "button": s.button,
        "sbSeat": s.sb_seat,
        "bbSeat": s.bb_seat,
        "community": s.community.iter().map(|&c| id(c)).collect::<Vec<_>>(),
        "players": s.players.iter().map(|p| json!([
            p.stack, p.bet, p.commit, status_str(p.status), p.has_acted, p.acted_at,
            p.hole.iter().map(|&c| id(c)).collect::<Vec<_>>(),
        ])).collect::<Vec<_>>(),
    })
}

/// Same shape as TS `legalActions()`.
fn legal_json(s: &State) -> Value {
    let l = s.legal().expect("legal actions");
    let mut actions = Vec::new();
    if l.can_fold {
        actions.push("fold");
    }
    actions.push(if l.can_check { "check" } else { "call" });
    if l.can_raise {
        actions.push(if l.opening { "bet" } else { "raise" });
    }
    json!({
        "actions": actions,
        "toCall": l.to_call,
        "callTo": l.call_to,
        "minRaiseTo": l.min_raise_to,
        "maxRaiseTo": l.max_raise_to,
        "canCheck": l.can_check,
    })
}

fn results_json(s: &State) -> Value {
    let pots = s.results.as_ref().expect("results");
    json!({
        "pots": pots.iter().map(|p| json!({"amount": p.amount, "winners": p.winners})).collect::<Vec<_>>(),
        "stacks": s.players.iter().map(|p| p.stack).collect::<Vec<_>>(),
        "history": s.actions.iter().map(|a| json!({
            "seat": a.seat,
            "street": street_str(a.street),
            "type": kind_str(a.kind),
            "allin": a.allin,
            "to": a.to,
            "added": a.added,
            "potBefore": a.pot_before,
            "currentBetBefore": a.current_bet_before,
            "stackBefore": a.stack_before,
        })).collect::<Vec<_>>(),
    })
}

#[test]
fn replay_ts_matches() {
    let fx = common::fixture("rules_traces.json");
    let matches = fx["matches"].as_array().unwrap();
    let (mut n_events, mut n_hands) = (0usize, 0usize);
    for (mi, m) in matches.iter().enumerate() {
        let mut s = State::new_table([200; 6], Some(0));
        for (ei, ev) in m["events"].as_array().unwrap().iter().enumerate() {
            let at = format!("match {mi} event {ei} ({})", ev["kind"]);
            match ev["kind"].as_str().unwrap() {
                "start" => {
                    let deck: Vec<u8> = match ev["deck"].as_array() {
                        Some(d) => d.iter().map(|c| parse(c.as_str().unwrap()).unwrap()).collect(),
                        None => Vec::new(),
                    };
                    s.start_hand(deck);
                    n_hands += 1;
                }
                "act" => {
                    assert_eq!(legal_json(&s), ev["legal"], "{at}: legal actions");
                    let a = &ev["action"];
                    let kind = parse_kind(a["type"].as_str().unwrap());
                    let amount = a.get("amount").and_then(Value::as_i64).map(|v| v as i32);
                    s.apply(Action { kind, amount });
                }
                "advance" => s.advance_street(),
                "end" => {
                    assert_eq!(results_json(&s), ev["results"], "{at}: results");
                    continue;
                }
                other => panic!("{at}: unknown event {other}"),
            }
            assert_eq!(summary(&s), ev["after"], "{at}: state after");
            n_events += 1;
        }
    }
    println!("replayed {} matches, {n_hands} hands, {n_events} steps", matches.len());
    assert!(n_hands > 100);
}
