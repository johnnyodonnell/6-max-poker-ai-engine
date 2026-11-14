//! The policy's action space — mirror of `src/engine/actions.ts`.
//!
//! Four action types (fold, check/call, raise, all-in) plus, for a raise, a
//! continuous size `s = ln((raiseTo - currentBet) / potAfterCall)` drawn from a
//! normal truncated to the legal interval [s_lo, s_hi] (min-raise up to one
//! chip short of all-in).

use rand::Rng;

use crate::rules::{Action, ActionType, Legal, State};

pub const FOLD: usize = 0;
pub const CALL: usize = 1;
pub const RAISE: usize = 2;
pub const ALLIN: usize = 3;
pub const NUM_TYPES: usize = 4;

#[derive(Clone, Debug)]
pub struct ActionSpace {
    pub legal: Legal,
    pub mask: [f32; NUM_TYPES],
    pub current_bet: i32,
    pub pot_after_call: i32,
    pub min_raise_to: i32,
    pub max_raise_to: i32,
    pub s_lo: f64,
    pub s_hi: f64,
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
    Some(ActionSpace {
        mask: [
            legal.can_fold as u8 as f32,
            1.0,
            sized as u8 as f32,
            legal.can_raise as u8 as f32,
        ],
        current_bet: cb,
        pot_after_call,
        min_raise_to: if sized { min_to } else { 0 },
        max_raise_to: if sized { max_to } else { 0 },
        s_lo: if sized { s(min_to) } else { 0.0 },
        s_hi: if sized { s(max_to) } else { 0.0 },
        legal,
    })
}

/// Raise-to for size `s`, rounded like JS `Math.round` and clamped.
pub fn raise_to_for(sp: &ActionSpace, s: f64) -> i32 {
    let to = (sp.current_bet as f64 + s.exp() * sp.pot_after_call as f64 + 0.5).floor() as i32;
    to.clamp(sp.min_raise_to, sp.max_raise_to)
}

/// The engine action for an action type (+ size for RAISE).
pub fn to_action(sp: &ActionSpace, ty: usize, s: f64) -> Action {
    match ty {
        FOLD => Action::new(ActionType::Fold),
        CALL => Action::new(if sp.legal.can_check { ActionType::Check } else { ActionType::Call }),
        RAISE => Action::to(
            if sp.current_bet == 0 { ActionType::Bet } else { ActionType::Raise },
            raise_to_for(sp, s),
        ),
        _ => Action::new(ActionType::AllIn),
    }
}

// --- truncated normal --------------------------------------------------------

/// Complementary error function (Numerical Recipes erfcc; |rel err| < 1.2e-7).
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let r = t * (-z * z - 1.26551223
        + t * (1.00002368
            + t * (0.37409196
                + t * (0.09678418
                    + t * (-0.18628806
                        + t * (0.27886807
                            + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
        .exp();
    if x >= 0.0 { r } else { 2.0 - r }
}

pub fn ndtr(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// Inverse standard normal CDF (Acklam).
pub fn ndtri(p: f64) -> f64 {
    const A: [f64; 6] = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.357751867269, -30.66479806614716, 2.506628277459239];
    const B: [f64; 5] = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const C: [f64; 6] = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const D: [f64; 4] = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const LO: f64 = 0.02425;
    if p < LO {
        let q = (-2.0 * p.ln()).sqrt();
        return (((((C[0] * q + C[1]) * q + C[2]) * q + C[3]) * q + C[4]) * q + C[5])
            / ((((D[0] * q + D[1]) * q + D[2]) * q + D[3]) * q + 1.0);
    }
    if p > 1.0 - LO {
        return -ndtri(1.0 - p);
    }
    let q = p - 0.5;
    let r = q * q;
    (((((A[0] * r + A[1]) * r + A[2]) * r + A[3]) * r + A[4]) * r + A[5]) * q
        / (((((B[0] * r + B[1]) * r + B[2]) * r + B[3]) * r + B[4]) * r + 1.0)
}

/// Draw from Normal(mu, sigma) truncated to [lo, hi] (inverse CDF, working in
/// the lower half for tail precision) — same algorithm as the TS sampler.
pub fn sample_trunc_normal(mu: f64, sigma: f64, lo: f64, hi: f64, rng: &mut impl Rng) -> f64 {
    if !(hi > lo) {
        return lo;
    }
    if (lo - mu) / sigma > 0.0 {
        return -sample_trunc_normal(-mu, sigma, -hi, -lo, rng);
    }
    let pa = ndtr((lo - mu) / sigma);
    let pb = ndtr((hi - mu) / sigma);
    if !(pb > pa) {
        return hi;
    }
    let u = pa + rng.gen::<f64>() * (pb - pa);
    let x = mu + sigma * ndtri(u.clamp(1e-300, 1.0 - 1e-16));
    x.clamp(lo, hi)
}

/// log density of the truncated normal at `x` (for logging/tests; the trainer
/// computes the same thing in tch).
pub fn trunc_normal_logpdf(x: f64, mu: f64, sigma: f64, lo: f64, hi: f64) -> f64 {
    if !(hi > lo) {
        return 0.0;
    }
    let z = (x - mu) / sigma;
    let mass = if (lo - mu) / sigma > 0.0 {
        ndtr(-(lo - mu) / sigma) - ndtr(-(hi - mu) / sigma)
    } else {
        ndtr((hi - mu) / sigma) - ndtr((lo - mu) / sigma)
    };
    -0.5 * z * z - (2.0 * std::f64::consts::PI).sqrt().ln() - sigma.ln() - mass.max(1e-300).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn inverse_cdf_round_trips() {
        for &x in &[-6.0, -3.0, -1.0, 0.0, 0.5, 2.0, 4.0] {
            assert!((ndtri(ndtr(x)) - x).abs() < 1e-5, "x={x}");
        }
    }

    /// Sample quantiles match the truncated CDF, including deep tails.
    #[test]
    fn trunc_normal_matches_cdf() {
        let mut rng = StdRng::seed_from_u64(11);
        for &(mu, sigma, lo, hi) in &[
            (0.0, 1.0, -1.0, 2.0),
            (-0.5, 0.6, -3.0, 1.6),
            (0.0, 0.3, 2.0, 3.0),   // interval far above mu
            (2.0, 0.2, -4.0, -1.0), // far below mu
        ] {
            let n = 40000;
            let mut xs: Vec<f64> = (0..n).map(|_| sample_trunc_normal(mu, sigma, lo, hi, &mut rng)).collect();
            assert!(xs.iter().all(|&x| x >= lo && x <= hi));
            xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if (lo - mu) / sigma > 5.0 || (mu - hi) / sigma > 5.0 {
                continue; // extreme tails: just bounds (edge-hugging by design)
            }
            let cdf = |x: f64| (ndtr((x - mu) / sigma) - ndtr((lo - mu) / sigma))
                / (ndtr((hi - mu) / sigma) - ndtr((lo - mu) / sigma));
            for q in [0.1, 0.5, 0.9] {
                let x = xs[(q * n as f64) as usize];
                assert!((cdf(x) - q).abs() < 0.01, "mu={mu} q={q} cdf={}", cdf(x));
            }
        }
    }
}
