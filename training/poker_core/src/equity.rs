//! Monte Carlo showdown equity of one hand against known opponent hands — a
//! training-only critic input (the policy never sees it).

use rand::seq::SliceRandom;
use rand::Rng;

use crate::cards::Card;
use crate::hand_eval::eval7;

/// Share of the pot `hero` wins at showdown against `opps` (their actual
/// cards), over `n` random completions of `board`. `dead` cards (e.g. folded
/// hands) are excluded from the completions.
pub fn equity(hero: &[Card], opps: &[&[Card]], board: &[Card], dead: &[Card], n: usize, rng: &mut impl Rng) -> f64 {
    if opps.is_empty() {
        return 1.0;
    }
    let mut used = [false; 52];
    for &c in hero.iter().chain(board).chain(dead) {
        used[c as usize] = true;
    }
    for o in opps {
        for &c in o.iter() {
            used[c as usize] = true;
        }
    }
    let mut rest: Vec<Card> = (0..52u8).filter(|&c| !used[c as usize]).collect();
    let need = 5 - board.len();
    let n = if need == 0 { 1 } else { n };
    let mut cards = [0u8; 7];
    cards[2..2 + board.len()].copy_from_slice(board);
    let mut tot = 0.0;
    for _ in 0..n {
        let (pick, _) = rest.partial_shuffle(rng, need);
        cards[2 + board.len()..].copy_from_slice(pick);
        cards[0] = hero[0];
        cards[1] = hero[1];
        let me = eval7(&cards);
        let mut ties = 1.0;
        let mut lost = false;
        for o in opps {
            cards[0] = o[0];
            cards[1] = o[1];
            let v = eval7(&cards);
            if v > me {
                lost = true;
                break;
            }
            if v == me {
                ties += 1.0;
            }
        }
        if !lost {
            tot += 1.0 / ties;
        }
    }
    tot / n as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::parse;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn aces_vs_kings_preflop() {
        let mut rng = StdRng::seed_from_u64(1);
        let aa = [parse("Ah").unwrap(), parse("As").unwrap()];
        let kk = [parse("Kh").unwrap(), parse("Ks").unwrap()];
        let e = equity(&aa, &[&kk], &[], &[], 20000, &mut rng);
        assert!((e - 0.82).abs() < 0.02, "AA vs KK equity {e}");
    }
}
