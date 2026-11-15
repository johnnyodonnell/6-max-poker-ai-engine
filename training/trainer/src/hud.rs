//! "HUD" statistics of the self-play policy — a poker-native sanity check that
//! the bot is learning poker (and not something degenerate).
//!
//! Per seat dealt in: VPIP (voluntarily put chips in preflop), PFR (raised
//! preflop), 3-bet (re-raised a single preflop raise), c-bet (preflop
//! aggressor bets the flop when checked to), plus showdown rate, action-type
//! mix (per discrete action), all-in rate and bet-size quantiles (as pot
//! fractions).

use poker_core::actions::{ACTION_NAMES, A_ALLIN, A_CALL, A_FOLD, NUM_ACTIONS};
use poker_core::rules::{ActionType, State, Status, Street};
use serde_json::{json, Value};

#[derive(Clone, Default)]
pub struct Hud {
    pub hands: u64,
    pub seats_dealt: u64,
    pub vpip: u64,
    pub pfr: u64,
    pub threebet_opp: u64,
    pub threebet: u64,
    pub cbet_opp: u64,
    pub cbet: u64,
    pub showdowns: u64,
    pub decisions: u64,
    pub actions: [u64; NUM_ACTIONS],
    pub raise_f: Vec<f32>,
}

impl Hud {
    /// `frac`: the bet's pot fraction for a sized action (else unused).
    pub fn record_decision(&mut self, a: usize, frac: f64) {
        self.decisions += 1;
        self.actions[a] += 1;
        if a != A_FOLD && a != A_CALL && a != A_ALLIN && self.raise_f.len() < 200_000 {
            self.raise_f.push(frac as f32);
        }
    }

    /// Tally a finished hand from its action history.
    pub fn record_hand(&mut self, st: &State) {
        self.hands += 1;
        let dealt: Vec<usize> = (0..st.players.len()).filter(|&i| !st.players[i].hole.is_empty()).collect();
        self.seats_dealt += dealt.len() as u64;
        let pre: Vec<_> = st.actions.iter().filter(|a| a.street == Street::Preflop).collect();
        for &seat in &dealt {
            let mine = pre.iter().filter(|a| a.seat == seat);
            let mut v = false;
            let mut r = false;
            for a in mine {
                v |= matches!(a.kind, ActionType::Call | ActionType::Bet | ActionType::Raise);
                r |= matches!(a.kind, ActionType::Bet | ActionType::Raise);
            }
            self.vpip += v as u64;
            self.pfr += r as u64;
        }
        // 3-bet: decisions facing exactly one preflop raise.
        let mut raises = 0;
        for a in &pre {
            if raises == 1 {
                self.threebet_opp += 1;
                self.threebet += (a.kind == ActionType::Raise) as u64;
            }
            if a.kind == ActionType::Raise {
                raises += 1;
            }
        }
        // C-bet: the last preflop raiser's first flop action, if nobody bet first.
        if let Some(agg) = pre.iter().rev().find(|a| a.kind == ActionType::Raise).map(|a| a.seat) {
            let mut bet_seen = false;
            for a in st.actions.iter().filter(|a| a.street == Street::Flop) {
                if a.seat == agg {
                    if !bet_seen {
                        self.cbet_opp += 1;
                        self.cbet += (a.kind == ActionType::Bet) as u64;
                    }
                    break;
                }
                bet_seen |= matches!(a.kind, ActionType::Bet | ActionType::Raise);
            }
        }
        let in_hand = st.players.iter().filter(|p| matches!(p.status, Status::Active | Status::AllIn)).count();
        self.showdowns += (in_hand >= 2) as u64;
    }

    pub fn merge(&mut self, o: &Hud) {
        self.hands += o.hands;
        self.seats_dealt += o.seats_dealt;
        self.vpip += o.vpip;
        self.pfr += o.pfr;
        self.threebet_opp += o.threebet_opp;
        self.threebet += o.threebet;
        self.cbet_opp += o.cbet_opp;
        self.cbet += o.cbet;
        self.showdowns += o.showdowns;
        self.decisions += o.decisions;
        for i in 0..NUM_ACTIONS {
            self.actions[i] += o.actions[i];
        }
        let room = 200_000usize.saturating_sub(self.raise_f.len());
        self.raise_f.extend(o.raise_f.iter().take(room));
    }

    pub fn to_json(&self) -> Value {
        let pct = |a: u64, b: u64| if b == 0 { 0.0 } else { (1000.0 * a as f64 / b as f64).round() / 10.0 };
        let mut s = self.raise_f.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let q = |p: f64| {
            if s.is_empty() {
                0.0
            } else {
                let v = s[((p * (s.len() - 1) as f64) as usize).min(s.len() - 1)] as f64;
                (v * 100.0).round() / 100.0
            }
        };
        json!({
            "vpip": pct(self.vpip, self.seats_dealt),
            "pfr": pct(self.pfr, self.seats_dealt),
            "3bet": pct(self.threebet, self.threebet_opp),
            "cbet": pct(self.cbet, self.cbet_opp),
            "wtsd": pct(self.showdowns, self.hands),
            "fold": pct(self.actions[A_FOLD], self.decisions),
            "call": pct(self.actions[A_CALL], self.decisions),
            "raise": pct(self.decisions - self.actions[A_FOLD] - self.actions[A_CALL] - self.actions[A_ALLIN], self.decisions),
            "allin": pct(self.actions[A_ALLIN], self.decisions),
            "mix": ACTION_NAMES.iter().enumerate().map(|(i, n)| (n.to_string(), json!(pct(self.actions[i], self.decisions)))).collect::<serde_json::Map<_, _>>(),
            "size_q10_50_90": [q(0.1), q(0.5), q(0.9)],
        })
    }
}
