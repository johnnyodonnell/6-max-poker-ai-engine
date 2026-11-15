//! The policy's action space — mirror of `src/engine/actions.ts`.
//!
//! The policy picks one of NUM_ACTIONS discrete actions:
//!
//!   0 fold   1 check/call   2 min-raise   3..=8 bet/raise to a pot fraction
//!   (POT_FRACS of the pot after calling: raise-to = currentBet + f * potAfterCall,
//!   rounded to a chip)   9 all-in
//!
//! A pot-fraction size is offered only when it lands strictly between the
//! previous offered size and all-in, so every legal action is a distinct
//! amount. The four coarse action types (fold / call / raise / all-in) and the
//! legal raise band [s_lo, s_hi] (s = ln((raiseTo - currentBet) / potAfterCall))
//! are still part of the encoder's input.

use crate::rules::{Action, ActionType, Legal, State};

pub const FOLD: usize = 0;
pub const CALL: usize = 1;
pub const RAISE: usize = 2;
pub const ALLIN: usize = 3;
pub const NUM_TYPES: usize = 4;

pub const A_FOLD: usize = 0;
pub const A_CALL: usize = 1;
pub const A_MINRAISE: usize = 2;
pub const A_ALLIN: usize = 9;
pub const NUM_ACTIONS: usize = 10;
/// Pot fractions for actions 3..=8.
pub const POT_FRACS: [f64; 6] = [0.33, 0.5, 0.75, 1.0, 1.5, 2.5];
pub const ACTION_NAMES: [&str; NUM_ACTIONS] = ["fold", "call", "minr", "p33", "p50", "p75", "p100", "p150", "p250", "allin"];

#[derive(Clone, Debug)]
pub struct ActionSpace {
    pub legal: Legal,
    /// Coarse type legality (fold, check/call, sized raise, all-in) — encoder input.
    pub mask: [f32; NUM_TYPES],
    pub current_bet: i32,
    pub pot_after_call: i32,
    pub min_raise_to: i32,
    pub max_raise_to: i32,
    pub s_lo: f64,
    pub s_hi: f64,
    /// Legality of each discrete action.
    pub amask: [f32; NUM_ACTIONS],
    /// Raise-to of each legal sized action (min-raise and pot fractions), else 0.
    pub amounts: [i32; NUM_ACTIONS],
}

pub fn action_space(st: &State) -> Option<ActionSpace> {
    let legal = st.legal()?;
    let p = &st.players[st.to_act as usize];
    let pot_total = st.pot + st.players.iter().map(|q| q.bet).sum::<i32>();
    let pot_after_call = pot_total + (st.current_bet - p.bet);
    let sized = legal.can_raise && legal.min_raise_to < legal.max_raise_to;
    let min_to = legal.min_raise_to;
    let max_to = legal.max_raise_to - 1;
    let cb = st.current_bet;
    let s = |to: i32| (((to - cb) as f64) / pot_after_call as f64).ln();

    let mut amask = [0f32; NUM_ACTIONS];
    let mut amounts = [0i32; NUM_ACTIONS];
    amask[A_FOLD] = legal.can_fold as u8 as f32;
    amask[A_CALL] = 1.0;
    amask[A_ALLIN] = legal.can_raise as u8 as f32;
    if sized {
        amask[A_MINRAISE] = 1.0;
        amounts[A_MINRAISE] = min_to;
        let mut last = min_to;
        for (k, f) in POT_FRACS.iter().enumerate() {
            let to = (cb as f64 + f * pot_after_call as f64 + 0.5).floor() as i32;
            if to > last && to <= max_to {
                amask[3 + k] = 1.0;
                amounts[3 + k] = to;
                last = to;
            }
        }
    }
    Some(ActionSpace {
        mask: [legal.can_fold as u8 as f32, 1.0, sized as u8 as f32, legal.can_raise as u8 as f32],
        current_bet: cb,
        pot_after_call,
        min_raise_to: if sized { min_to } else { 0 },
        max_raise_to: if sized { max_to } else { 0 },
        s_lo: if sized { s(min_to) } else { 0.0 },
        s_hi: if sized { s(max_to) } else { 0.0 },
        amask,
        amounts,
        legal,
    })
}

/// The engine action for a discrete action.
pub fn to_action(sp: &ActionSpace, a: usize) -> Action {
    match a {
        A_FOLD => Action::new(ActionType::Fold),
        A_CALL => Action::new(if sp.legal.can_check { ActionType::Check } else { ActionType::Call }),
        A_ALLIN => Action::new(ActionType::AllIn),
        _ => {
            debug_assert!(sp.amask[a] > 0.0, "illegal sized action {a}");
            Action::to(if sp.current_bet == 0 { ActionType::Bet } else { ActionType::Raise }, sp.amounts[a])
        }
    }
}

/// Pot fraction (raise increment / pot after calling) of a sized action.
pub fn pot_fraction(sp: &ActionSpace, a: usize) -> f64 {
    (sp.amounts[a] - sp.current_bet) as f64 / sp.pot_after_call as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::table::deal_training_hand;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Every legal discrete action is a distinct, legal engine action.
    #[test]
    fn discrete_actions_are_distinct_and_legal() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut checked = 0;
        for _ in 0..3000 {
            let mut s = deal_training_hand(&mut rng);
            while !s.hand_done() {
                match s.phase {
                    crate::rules::Phase::Betting if s.to_act >= 0 => {
                        let sp = action_space(&s).unwrap();
                        let mut last = sp.min_raise_to - 1;
                        for a in A_MINRAISE..A_ALLIN {
                            if sp.amask[a] > 0.0 {
                                assert!(sp.amounts[a] > last && sp.amounts[a] >= sp.legal.min_raise_to && sp.amounts[a] < sp.legal.max_raise_to);
                                last = sp.amounts[a];
                            }
                        }
                        let legal: Vec<usize> = (0..NUM_ACTIONS).filter(|&a| sp.amask[a] > 0.0).collect();
                        let a = legal[rng.gen_range(0..legal.len())];
                        s.apply(to_action(&sp, a));
                        checked += 1;
                    }
                    _ => s.advance_street(),
                }
            }
        }
        assert!(checked > 10000);
    }
}
