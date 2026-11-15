//! Canonical NN input encoding — mirror of `src/engine/encode.ts` (the source
//! of truth; see that file for the layout). Parity-tested against encodings
//! dumped from the TS engine.

use crate::actions::{action_space, ActionSpace};
use crate::cards::{rank, suit, Card};
use crate::rules::{ActionType, HandAction, State, Status, Street, BIG_BLIND, SEATS};

pub const CARD_SLOTS: usize = 7;
pub const NO_CARD: f32 = 52.0;
pub const HIST_TOKENS: usize = 48;
pub const TOKEN_FEATS: usize = 22;
pub const SEAT_FEATS: usize = 12;
pub const STATIC_SIZE: usize = 4 + 5 + 2 + SEATS * SEAT_FEATS + 4 + 4 + 5; // 96
pub const HIST_OFFSET: usize = CARD_SLOTS;
pub const STATIC_OFFSET: usize = CARD_SLOTS + HIST_TOKENS * TOKEN_FEATS;
pub const INPUT_SIZE: usize = STATIC_OFFSET + STATIC_SIZE; // 1159

#[inline]
fn lbb(chips: i32) -> f64 {
    (chips as f64 / BIG_BLIND as f64).ln_1p()
}

fn street_idx(s: Street) -> usize {
    s as usize
}

fn ordered(cards: &[Card]) -> Vec<Card> {
    let mut v = cards.to_vec();
    v.sort_by(|&a, &b| rank(b).cmp(&rank(a)).then(suit(a).cmp(&suit(b))));
    v
}

/// The groups a player's cards are canonicalized in: hole (sorted), flop
/// (sorted), turn, river.
fn card_groups(hole: &[Card], community: &[Card]) -> [Vec<Card>; 4] {
    let n = community.len();
    [
        ordered(hole),
        ordered(&community[..n.min(3)]),
        community[n.min(3)..n.min(4)].to_vec(),
        community[n.min(4)..n.min(5)].to_vec(),
    ]
}

/// Canonical suit relabeling: suits numbered by first appearance (hole cards,
/// then flop, turn, river). Unseen suits map to u8::MAX.
pub fn suit_map(hole: &[Card], community: &[Card]) -> [u8; 4] {
    let mut map = [u8::MAX; 4];
    let mut next = 0u8;
    for cards in card_groups(hole, community).iter() {
        for &c in cards {
            let su = suit(c) as usize;
            if map[su] == u8::MAX {
                map[su] = next;
                next += 1;
            }
        }
    }
    map
}

/// Canonical index of `c` under `map`, giving still-unseen suits the next free
/// labels (for cards the player can't see, e.g. a training-only critic input).
pub fn canonical_index(c: Card, map: &mut [u8; 4]) -> f32 {
    let su = suit(c) as usize;
    if map[su] == u8::MAX {
        map[su] = map.iter().filter(|&&m| m != u8::MAX).count() as u8;
    }
    (map[su] as u32 * 13 + (rank(c) as u32 - 2)) as f32
}

/// Canonical card indices (hole 2, flop 3, turn, river), NO_CARD-padded.
pub fn canonical_cards(hole: &[Card], community: &[Card]) -> [f32; CARD_SLOTS] {
    let mut map = suit_map(hole, community);
    let mut out = [NO_CARD; CARD_SLOTS];
    let starts = [0, 2, 5, 6];
    for (g, cards) in card_groups(hole, community).iter().enumerate() {
        for (j, &c) in cards.iter().enumerate() {
            out[starts[g] + j] = canonical_index(c, &mut map);
        }
    }
    out
}

fn encode_token(x: &mut [f32], a: &HandAction, mover: usize) {
    x[street_idx(a.street)] = 1.0;
    x[4 + (a.seat + SEATS - mover) % SEATS] = 1.0;
    let t = match a.kind {
        ActionType::Fold => 0,
        ActionType::Check => 1,
        ActionType::Call => 2,
        ActionType::Bet => 3,
        ActionType::Raise => 4,
        ActionType::AllIn => unreachable!("history kinds are normalized"),
    };
    x[10 + t] = 1.0;
    x[15] = a.allin as u8 as f32;
    let bet_before = a.to - a.added;
    if matches!(a.kind, ActionType::Bet | ActionType::Raise) {
        let pac = a.pot_before + (a.current_bet_before - bet_before);
        x[16] = (((a.to - a.current_bet_before) as f64) / pac as f64).ln() as f32;
    }
    if a.kind == ActionType::Call {
        x[17] = (a.added as f64 / a.pot_before as f64) as f32;
    }
    x[18] = lbb(a.added) as f32;
    x[19] = lbb(a.pot_before) as f32;
    x[20] = lbb(a.stack_before) as f32;
    x[21] = 1.0;
}

fn encode_static(x: &mut [f32], st: &State, sp: &ActionSpace) {
    let mover = st.to_act as usize;
    let me = &st.players[mover];
    let mut o = 0;
    x[street_idx(st.street)] = 1.0;
    o += 4;
    let dealt = st.players.iter().filter(|p| !p.hole.is_empty()).count();
    x[o + dealt.clamp(2, 6) - 2] = 1.0;
    o += 5;
    let in_hand = st.players.iter().filter(|p| matches!(p.status, Status::Active | Status::AllIn)).count();
    let can_act = st.players.iter().filter(|p| p.status == Status::Active).count();
    x[o] = (in_hand as f64 / SEATS as f64) as f32;
    x[o + 1] = (can_act as f64 / SEATS as f64) as f32;
    o += 2;
    for r in 0..SEATS {
        let i = (mover + r) % SEATS;
        let p = &st.players[i];
        let status = match p.status {
            Status::Out => 0,
            Status::Active => 1,
            Status::AllIn => 2,
            Status::Folded => 3,
        };
        x[o + status] = 1.0;
        x[o + 4] = (i as i32 == st.button) as u8 as f32;
        x[o + 5] = (i as i32 == st.sb_seat) as u8 as f32;
        x[o + 6] = (i as i32 == st.bb_seat) as u8 as f32;
        x[o + 7] = lbb(p.stack) as f32;
        x[o + 8] = lbb(p.bet) as f32;
        x[o + 9] = lbb(p.commit) as f32;
        x[o + 10] = (p.stack as f64 / 200.0) as f32;
        x[o + 11] = p.has_acted as u8 as f32;
        o += SEAT_FEATS;
    }
    let pot_total = st.pot + st.players.iter().map(|q| q.bet).sum::<i32>();
    let to_call = sp.legal.to_call;
    x[o] = lbb(pot_total) as f32;
    x[o + 1] = lbb(to_call) as f32;
    x[o + 2] = (to_call as f64 / (pot_total + to_call) as f64) as f32;
    x[o + 3] = (me.stack as f64 / pot_total as f64).ln_1p() as f32;
    o += 4;
    x[o] = sp.s_lo as f32;
    x[o + 1] = sp.s_hi as f32;
    x[o + 2] = lbb(sp.min_raise_to) as f32;
    x[o + 3] = lbb(sp.max_raise_to) as f32;
    o += 4;
    x[o] = sp.mask[0];
    x[o + 1] = sp.legal.can_check as u8 as f32;
    x[o + 2] = sp.mask[2];
    x[o + 3] = sp.mask[3];
    x[o + 4] = (!sp.legal.can_check) as u8 as f32;
}

/// Encode the state for the player to act into `x` (len INPUT_SIZE, any
/// contents). Returns the action space, or None if nobody is to act.
pub fn encode_into(st: &State, x: &mut [f32]) -> Option<ActionSpace> {
    let sp = action_space(st)?;
    x.fill(0.0);
    let mover = st.to_act as usize;
    x[..CARD_SLOTS].copy_from_slice(&canonical_cards(&st.players[mover].hole, &st.community));
    let acts = &st.actions;
    for t in 0..HIST_TOKENS.min(acts.len()) {
        let off = HIST_OFFSET + t * TOKEN_FEATS;
        encode_token(&mut x[off..off + TOKEN_FEATS], &acts[acts.len() - 1 - t], mover);
    }
    encode_static(&mut x[STATIC_OFFSET..], st, &sp);
    Some(sp)
}

pub fn encode(st: &State) -> Option<(Vec<f32>, ActionSpace)> {
    let mut x = vec![0f32; INPUT_SIZE];
    let sp = encode_into(st, &mut x)?;
    Some((x, sp))
}
