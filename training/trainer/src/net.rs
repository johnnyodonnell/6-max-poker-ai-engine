//! The networks, defined once from building blocks that each know how to run
//! in tch (`fwd`, used for self-play and training) and how to write themselves
//! into an ONNX graph (`emit`, used for the browser export). The exported model
//! is checked against `fwd` by training/scripts/test_onnx_parity.ts.
//!
//! Input: the flat encoding from poker_core::encode ([B, INPUT_SIZE]), sliced
//! into cards / history tokens / static features.
//!   cards   -> (a) card+rank+suit embeddings, summed per group (hole, flop,
//!              turn, river)                                     -> [B, 4*D_CARD]
//!              (b) suit x rank planes (multi-hot 4x13) for hole, flop, turn,
//!              river, board and all cards, plus each of hole/board/all's rank
//!              and suit counts -> Linear -> GELU               -> [B, D_PLANES]
//!              The planes make card *relations* (pairs, sets, flushes,
//!              straights) simple patterns; summed embeddings hide them.
//!   history -> token projection + position embedding, pre-LN self-attention
//!              layers, pooled as masked mean + masked max + learned-query
//!              readouts, zeroed when there is no history       -> [B, (R+2)*D_MODEL]
//!   static  -> as is                                           -> [B, STATIC_SIZE]
//! concat -> Linear(width) -> ResBlocks -> LN -> Linear(NUM_ACTIONS).
//!
//! The policy's head is the logits of the discrete actions (masked outside).
//! The training-only critic (`NetCfg::CRITIC`) uses the same body plus a
//! privileged input — every live opponent's hole cards (as planes), the
//! mover's showdown equity against them, and the live-opponent count — and
//! its head is Q(s, a) per action (expected chip result / 200).

use tch::nn::{self, Module};
use tch::{Device, Kind, Tensor};

use poker_core::actions::NUM_ACTIONS;
use poker_core::encode::{CARD_SLOTS, HIST_OFFSET, HIST_TOKENS, INPUT_SIZE, STATIC_OFFSET, STATIC_SIZE, TOKEN_FEATS};

use crate::onnx::{Attr, Dim, Graph, FLOAT, INT64};

pub const D_CARD: i64 = 64;
pub const D_MODEL: i64 = 128;
pub const N_HEADS: i64 = 4;
pub const HEAD_DIM: i64 = D_MODEL / N_HEADS;
pub const FFN: i64 = 256;
pub const N_HIST_LAYERS: usize = 2;
pub const N_READOUT: i64 = 4;
pub const D_PLANES: i64 = 256;
/// 6 planes of 52 + (13 rank + 4 suit counts) for hole, board, all.
const PLANE_IN: i64 = 6 * 52 + 3 * 17;
/// Privileged critic input: 5 opponent seats x 2 card slots (canonical index,
/// NO_CARD if empty), equity vs them, live opponents / 5.
pub const PRIV_SIZE: usize = 12;
const PRIV_IN: i64 = 5 * 52 + 52 + 2;
pub const D_PRIV: i64 = 256;
const MASK_NEG: f64 = 1e9;
const T: i64 = HIST_TOKENS as i64;
const NO_CARD: i64 = 52;

pub const OPSET: i64 = 17;

fn data(t: &Tensor) -> Vec<f32> {
    let t = t.to_device(Device::Cpu).to_kind(Kind::Float).contiguous().view([-1]);
    Vec::<f32>::try_from(&t).expect("tensor data")
}

fn emit_tensor(g: &mut Graph, hint: &str, t: &Tensor) -> String {
    g.init_f32(hint, &t.size(), &data(t))
}

fn gelu(x: &Tensor) -> Tensor {
    x.gelu("none")
}

/// Exact-erf GELU: 0.5 * x * (1 + erf(x / sqrt 2)).
fn emit_gelu(g: &mut Graph, x: &str) -> String {
    let s2 = g.scalar_f32(std::f32::consts::SQRT_2);
    let one = g.scalar_f32(1.0);
    let half = g.scalar_f32(0.5);
    let d = g.node("Div", &[x, &s2], &[]);
    let e = g.node("Erf", &[&d], &[]);
    let a = g.node("Add", &[&e, &one], &[]);
    let m = g.node("Mul", &[x, &a], &[]);
    g.node("Mul", &[&m, &half], &[])
}

fn emit_slice(g: &mut Graph, x: &str, axis: i64, start: i64, end: i64) -> String {
    let s = g.init_i64("st", &[1], &[start]);
    let e = g.init_i64("en", &[1], &[end]);
    let a = g.init_i64("ax", &[1], &[axis]);
    g.node("Slice", &[x, &s, &e, &a], &[])
}

fn emit_reduce_sum(g: &mut Graph, x: &str, axis: i64, keepdims: bool) -> String {
    let a = g.init_i64("ax", &[1], &[axis]);
    g.node("ReduceSum", &[x, &a], &[Attr::Int("keepdims", keepdims as i64)])
}

fn emit_reshape(g: &mut Graph, x: &str, shape: &[i64]) -> String {
    let s = g.init_i64("shape", &[shape.len() as i64], shape);
    g.node("Reshape", &[x, &s], &[])
}

fn emit_unsqueeze(g: &mut Graph, x: &str, axis: i64) -> String {
    let a = g.init_i64("ax", &[1], &[axis]);
    g.node("Unsqueeze", &[x, &a], &[])
}

// ---- building blocks ----------------------------------------------------------

struct Lin(nn::Linear);

impl Lin {
    fn new(p: nn::Path, i: i64, o: i64) -> Self {
        Lin(nn::linear(p, i, o, Default::default()))
    }
    fn fwd(&self, x: &Tensor) -> Tensor {
        self.0.forward(x)
    }
    fn emit(&self, g: &mut Graph, x: &str) -> String {
        let w = emit_tensor(g, "w", &self.0.ws.tr());
        let mm = g.node("MatMul", &[x, &w], &[]);
        let b = emit_tensor(g, "b", self.0.bs.as_ref().expect("bias"));
        g.node("Add", &[&mm, &b], &[])
    }
}

struct LN(nn::LayerNorm);

impl LN {
    fn new(p: nn::Path, d: i64) -> Self {
        LN(nn::layer_norm(p, vec![d], Default::default()))
    }
    fn fwd(&self, x: &Tensor) -> Tensor {
        self.0.forward(x)
    }
    fn emit(&self, g: &mut Graph, x: &str) -> String {
        let w = emit_tensor(g, "ln_w", self.0.ws.as_ref().expect("ln weight"));
        let b = emit_tensor(g, "ln_b", self.0.bs.as_ref().expect("ln bias"));
        g.node("LayerNormalization", &[x, &w, &b], &[Attr::Int("axis", -1), Attr::Float("epsilon", 1e-5)])
    }
}

struct Embed(nn::Embedding);

impl Embed {
    fn new(p: nn::Path, n: i64, d: i64) -> Self {
        Embed(nn::embedding(p, n, d, Default::default()))
    }
    fn fwd(&self, idx: &Tensor) -> Tensor {
        self.0.forward(idx)
    }
    fn emit(&self, g: &mut Graph, idx: &str) -> String {
        let w = emit_tensor(g, "emb", &self.0.ws);
        g.node("Gather", &[&w, idx], &[Attr::Int("axis", 0)])
    }
}

/// Pre-LN self-attention block over the history tokens.
struct HistLayer {
    ln1: LN,
    q: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ln2: LN,
    fc1: Lin,
    fc2: Lin,
}

impl HistLayer {
    fn new(p: nn::Path) -> Self {
        HistLayer {
            ln1: LN::new(&p / "ln1", D_MODEL),
            q: Lin::new(&p / "q", D_MODEL, D_MODEL),
            k: Lin::new(&p / "k", D_MODEL, D_MODEL),
            v: Lin::new(&p / "v", D_MODEL, D_MODEL),
            o: Lin::new(&p / "o", D_MODEL, D_MODEL),
            ln2: LN::new(&p / "ln2", D_MODEL),
            fc1: Lin::new(&p / "fc1", D_MODEL, FFN),
            fc2: Lin::new(&p / "fc2", FFN, D_MODEL),
        }
    }

    /// h: [B,T,D]; addmask: [B,1,1,T].
    fn fwd(&self, h: &Tensor, addmask: &Tensor) -> Tensor {
        let x = self.ln1.fwd(h);
        let heads = |t: Tensor| t.view([-1, T, N_HEADS, HEAD_DIM]).transpose(1, 2);
        let q = heads(self.q.fwd(&x));
        let k = heads(self.k.fwd(&x));
        let v = heads(self.v.fwd(&x));
        let att = q.matmul(&k.transpose(-2, -1)) * (1.0 / (HEAD_DIM as f64).sqrt()) + addmask;
        let att = att.softmax(-1, Kind::Float);
        let a = att.matmul(&v).transpose(1, 2).reshape([-1, T, D_MODEL]);
        let h = h + self.o.fwd(&a);
        let f = self.fc2.fwd(&gelu(&self.fc1.fwd(&self.ln2.fwd(&h))));
        h + f
    }

    fn emit(&self, g: &mut Graph, h: &str, addmask: &str) -> String {
        let x = self.ln1.emit(g, h);
        let heads = |g: &mut Graph, lin: &Lin, perm: Vec<i64>| {
            let t = lin.emit(g, &x);
            let t = emit_reshape(g, &t, &[-1, T, N_HEADS, HEAD_DIM]);
            g.node("Transpose", &[&t], &[Attr::Ints("perm", perm)])
        };
        let q = heads(g, &self.q, vec![0, 2, 1, 3]); // [B,H,T,Dh]
        let kt = heads(g, &self.k, vec![0, 2, 3, 1]); // [B,H,Dh,T]
        let v = heads(g, &self.v, vec![0, 2, 1, 3]);
        let scale = g.scalar_f32((1.0 / (HEAD_DIM as f64).sqrt()) as f32);
        let att = g.node("MatMul", &[&q, &kt], &[]);
        let att = g.node("Mul", &[&att, &scale], &[]);
        let att = g.node("Add", &[&att, addmask], &[]);
        let att = g.node("Softmax", &[&att], &[Attr::Int("axis", -1)]);
        let a = g.node("MatMul", &[&att, &v], &[]);
        let a = g.node("Transpose", &[&a], &[Attr::Ints("perm", vec![0, 2, 1, 3])]);
        let a = emit_reshape(g, &a, &[-1, T, D_MODEL]);
        let o = self.o.emit(g, &a);
        let h = g.node("Add", &[h, &o], &[]);
        let f = self.ln2.emit(g, &h);
        let f = self.fc1.emit(g, &f);
        let f = emit_gelu(g, &f);
        let f = self.fc2.emit(g, &f);
        g.node("Add", &[&h, &f], &[])
    }
}

struct ResBlock {
    ln1: LN,
    fc1: Lin,
    ln2: LN,
    fc2: Lin,
}

impl ResBlock {
    fn new(p: nn::Path, w: i64) -> Self {
        ResBlock {
            ln1: LN::new(&p / "ln1", w),
            fc1: Lin::new(&p / "fc1", w, w),
            ln2: LN::new(&p / "ln2", w),
            fc2: Lin::new(&p / "fc2", w, w),
        }
    }
    fn fwd(&self, x: &Tensor) -> Tensor {
        let h = self.fc1.fwd(&gelu(&self.ln1.fwd(x)));
        let h = self.fc2.fwd(&gelu(&self.ln2.fwd(&h)));
        x + h
    }
    fn emit(&self, g: &mut Graph, x: &str) -> String {
        let h = self.ln1.emit(g, x);
        let h = emit_gelu(g, &h);
        let h = self.fc1.emit(g, &h);
        let h = self.ln2.emit(g, &h);
        let h = emit_gelu(g, &h);
        let h = self.fc2.emit(g, &h);
        g.node("Add", &[x, &h], &[])
    }
}

// ---- the network ----------------------------------------------------------------

#[derive(Clone, Copy)]
pub struct NetCfg {
    pub width: i64,
    pub blocks: usize,
    pub privileged: bool,
}

impl NetCfg {
    pub const POLICY: NetCfg = NetCfg { width: 512, blocks: 4, privileged: false };
    pub const CRITIC: NetCfg = NetCfg { width: 384, blocks: 3, privileged: true };
}

pub struct PokerNet {
    cfg: NetCfg,
    card_emb: Embed,
    rank_emb: Embed,
    suit_emb: Embed,
    /// [53, 52]: one-hot rows for the 52 cards, zeros for NO_CARD (a constant).
    plane_tbl: Tensor,
    planes_fc: Lin,
    tok_proj: Lin,
    hist_pos: Tensor,
    hist_layers: Vec<HistLayer>,
    hist_ln: LN,
    readout_q: Tensor,
    priv_fc: Option<Lin>,
    stem: Lin,
    blocks: Vec<ResBlock>,
    head_ln: LN,
    head_fc: Lin,
}

/// Initial policy-logit biases: fold, check/call, min-raise, six pot-fraction
/// sizes, all-in. The sizes share a raise's prior mass, and all-in starts
/// unlikely, so the untrained policy neither raises nor shoves most of the time.
const ACTION_BIAS_INIT: [f64; NUM_ACTIONS] = [0.0, 0.5, -1.0, -1.8, -1.8, -1.8, -1.8, -1.8, -1.8, -2.5];

fn plane_table(dev: Device) -> Tensor {
    Tensor::cat(&[Tensor::eye(52, (Kind::Float, dev)), Tensor::zeros([1, 52], (Kind::Float, dev))], 0)
}

/// [B, 52] multi-hot planes -> rank counts [B,13] and suit counts [B,4].
fn counts(p: &Tensor) -> (Tensor, Tensor) {
    let m = p.view([-1, 4, 13]);
    (m.sum_dim_intlist(1, false, Kind::Float), m.sum_dim_intlist(2, false, Kind::Float))
}

fn emit_counts(g: &mut Graph, p: &str) -> (String, String) {
    let m = emit_reshape(g, p, &[-1, 4, 13]);
    (emit_reduce_sum(g, &m, 1, false), emit_reduce_sum(g, &m, 2, false))
}

impl PokerNet {
    pub fn new(vs: &nn::Path, cfg: NetCfg) -> Self {
        let w = cfg.width;
        let trunk_in = STATIC_SIZE as i64 + 4 * D_CARD + D_PLANES + (N_READOUT + 2) * D_MODEL + if cfg.privileged { D_PRIV } else { 0 };
        let net = PokerNet {
            cfg,
            card_emb: Embed::new(vs / "card_emb", NO_CARD + 1, D_CARD),
            rank_emb: Embed::new(vs / "rank_emb", 13, D_CARD),
            suit_emb: Embed::new(vs / "suit_emb", 4, D_CARD),
            plane_tbl: plane_table(vs.device()),
            planes_fc: Lin::new(vs / "planes_fc", PLANE_IN, D_PLANES),
            tok_proj: Lin::new(vs / "tok_proj", TOKEN_FEATS as i64, D_MODEL),
            hist_pos: vs.var("hist_pos", &[T, D_MODEL], nn::Init::Randn { mean: 0.0, stdev: 0.02 }),
            hist_layers: (0..N_HIST_LAYERS).map(|i| HistLayer::new(vs / "hist_layers" / i)).collect(),
            hist_ln: LN::new(vs / "hist_ln", D_MODEL),
            readout_q: vs.var("readout_q", &[N_READOUT, D_MODEL], nn::Init::Randn { mean: 0.0, stdev: 0.02 }),
            priv_fc: cfg.privileged.then(|| Lin::new(vs / "priv_fc", PRIV_IN, D_PRIV)),
            stem: Lin::new(vs / "stem", trunk_in, w),
            blocks: (0..cfg.blocks).map(|i| ResBlock::new(vs / "blocks" / i, w)).collect(),
            head_ln: LN::new(vs / "head_ln", w),
            head_fc: Lin::new(vs / "head_fc", w, NUM_ACTIONS as i64),
        };
        // The head starts near its bias: the policy at ACTION_BIAS_INIT, the
        // critic's Q at 0.
        tch::no_grad(|| {
            let dev = net.head_fc.0.ws.device();
            let mut hw = net.head_fc.0.ws.shallow_clone();
            let _ = hw.f_mul_scalar_(0.01).unwrap();
            let bias: Vec<f64> = if cfg.privileged { vec![0.0; NUM_ACTIONS] } else { ACTION_BIAS_INIT.to_vec() };
            let mut hb = net.head_fc.0.bs.as_ref().unwrap().shallow_clone();
            hb.copy_(&Tensor::from_slice(&bias).to_kind(Kind::Float).to_device(dev));
        });
        net
    }

    /// Policy logits [B, NUM_ACTIONS] (unmasked).
    pub fn forward(&self, x: &Tensor) -> Tensor {
        assert!(!self.cfg.privileged, "critic needs forward_priv");
        self.body(x, None)
    }

    /// Critic Q values [B, NUM_ACTIONS]; `p` is the [B, PRIV_SIZE] privileged input.
    pub fn forward_priv(&self, x: &Tensor, p: &Tensor) -> Tensor {
        assert!(self.cfg.privileged);
        self.body(x, Some(p))
    }

    fn planes(&self, idx: &Tensor) -> Tensor {
        let n = idx.size()[1];
        self.plane_tbl.index_select(0, &idx.reshape([-1])).view([-1, n, 52])
    }

    fn body(&self, x: &Tensor, p: Option<&Tensor>) -> Tensor {
        // --- cards ---
        let idx = x.narrow(1, 0, CARD_SLOTS as i64).to_kind(Kind::Int64); // [B,7]
        let dealt = idx.lt(NO_CARD).to_kind(Kind::Float).unsqueeze(-1);
        let suit = idx.divide_scalar_mode(13, "floor").clamp_max(3);
        let rank = idx.remainder(13);
        let e = (self.card_emb.fwd(&idx) + self.rank_emb.fwd(&rank) + self.suit_emb.fwd(&suit)) * dealt;
        let group = |s: i64, n: i64| e.narrow(1, s, n).sum_dim_intlist(1, false, Kind::Float);
        let cards = Tensor::cat(&[group(0, 2), group(2, 3), group(5, 1), group(6, 1)], 1);

        let oh = self.planes(&idx); // [B,7,52]
        let pg = |s: i64, n: i64| oh.narrow(1, s, n).sum_dim_intlist(1, false, Kind::Float);
        let (hole, flop, turn, river) = (pg(0, 2), pg(2, 3), pg(5, 1), pg(6, 1));
        let board = &flop + &turn + &river;
        let all = &hole + &board;
        let (hr, hs) = counts(&hole);
        let (br, bs) = counts(&board);
        let (ar, asu) = counts(&all);
        let planes = Tensor::cat(&[&hole, &flop, &turn, &river, &board, &all, &hr, &hs, &br, &bs, &ar, &asu], 1);
        let planes = gelu(&self.planes_fc.fwd(&planes));

        // --- history ---
        let tok = x.narrow(1, HIST_OFFSET as i64, T * TOKEN_FEATS as i64).view([-1, T, TOKEN_FEATS as i64]);
        let valid = tok.select(2, TOKEN_FEATS as i64 - 1); // [B,T]
        let neg = (&valid - 1.0) * MASK_NEG; // 0 valid, -1e9 padding
        let mut h = self.tok_proj.fwd(&tok) + &self.hist_pos;
        let addmask = neg.view([-1, 1, 1, T]);
        for l in &self.hist_layers {
            h = l.fwd(&h, &addmask);
        }
        let h = self.hist_ln.fwd(&h);
        let scores = h.matmul(&self.readout_q.tr()) * (1.0 / (D_MODEL as f64).sqrt()) + neg.unsqueeze(-1);
        let att = scores.softmax(1, Kind::Float); // [B,T,R]
        let readout = att.transpose(1, 2).matmul(&h).view([-1, N_READOUT * D_MODEL]);
        let vm = valid.unsqueeze(-1);
        let mean = (&h * &vm).sum_dim_intlist(1, false, Kind::Float)
            / vm.sum_dim_intlist(1, false, Kind::Float).clamp_min(1.0);
        let mx = (&h + (&vm - 1.0) * MASK_NEG).amax(1, false);
        let any = valid.amax(1, true);
        let pooled = Tensor::cat(&[mean, mx, readout], 1) * any;

        // --- trunk ---
        let stat = x.narrow(1, STATIC_OFFSET as i64, STATIC_SIZE as i64);
        let mut parts = vec![stat, cards, planes, pooled];
        if let (Some(fc), Some(p)) = (&self.priv_fc, p) {
            let oidx = p.narrow(1, 0, 10).to_kind(Kind::Int64);
            let opp = self.planes(&oidx).view([-1, 5, 2, 52]).sum_dim_intlist(2, false, Kind::Float); // [B,5,52]
            let any_opp = opp.sum_dim_intlist(1, false, Kind::Float);
            let pin = Tensor::cat(&[opp.view([-1, 5 * 52]), any_opp, p.narrow(1, 10, 2)], 1);
            parts.push(gelu(&fc.fwd(&pin)));
        }
        let mut t = self.stem.fwd(&Tensor::cat(&parts, 1));
        for b in &self.blocks {
            t = b.fwd(&t);
        }
        self.head_fc.fwd(&self.head_ln.fwd(&t))
    }

    /// Serialize the policy (current weights) as a self-contained ONNX model:
    /// input "input" [batch, INPUT_SIZE]; output "logits" [batch, NUM_ACTIONS].
    pub fn export_onnx(&self) -> Vec<u8> {
        assert!(!self.cfg.privileged, "only the policy is exported");
        let mut g = Graph::new();
        let x = g.input("input", FLOAT, &[Dim::Named("batch"), Dim::Fixed(INPUT_SIZE as i64)]);

        // --- cards: embeddings ---
        let cf = emit_slice(&mut g, &x, 1, 0, CARD_SLOTS as i64);
        let idx = g.node("Cast", &[&cf], &[Attr::Int("to", INT64)]);
        let lim = g.scalar_f32(NO_CARD as f32 - 0.5);
        let dealt = g.node("Less", &[&cf, &lim], &[]);
        let dealt = g.node("Cast", &[&dealt], &[Attr::Int("to", FLOAT)]);
        let dealt = emit_unsqueeze(&mut g, &dealt, -1);
        let c13 = g.init_i64("c13", &[], &[13]);
        let c3 = g.init_i64("c3", &[], &[3]);
        let suit = g.node("Div", &[&idx, &c13], &[]);
        let suit = g.node("Min", &[&suit, &c3], &[]);
        let rank = g.node("Mod", &[&idx, &c13], &[Attr::Int("fmod", 0)]);
        let ec = self.card_emb.emit(&mut g, &idx);
        let er = self.rank_emb.emit(&mut g, &rank);
        let es = self.suit_emb.emit(&mut g, &suit);
        let e = g.node("Add", &[&ec, &er], &[]);
        let e = g.node("Add", &[&e, &es], &[]);
        let e = g.node("Mul", &[&e, &dealt], &[]);
        let mut groups = Vec::new();
        for (s, n) in [(0, 2), (2, 3), (5, 1), (6, 1)] {
            let sl = emit_slice(&mut g, &e, 1, s, s + n);
            groups.push(emit_reduce_sum(&mut g, &sl, 1, false));
        }
        let gref: Vec<&str> = groups.iter().map(String::as_str).collect();
        let cards = g.node("Concat", &gref, &[Attr::Int("axis", 1)]);

        // --- cards: planes ---
        let tbl = emit_tensor(&mut g, "plane_tbl", &self.plane_tbl);
        let oh = g.node("Gather", &[&tbl, &idx], &[Attr::Int("axis", 0)]); // [B,7,52]
        let mut pgs = Vec::new();
        for (s, n) in [(0, 2), (2, 3), (5, 1), (6, 1)] {
            let sl = emit_slice(&mut g, &oh, 1, s, s + n);
            pgs.push(emit_reduce_sum(&mut g, &sl, 1, false));
        }
        let board = g.node("Add", &[&pgs[1], &pgs[2]], &[]);
        let board = g.node("Add", &[&board, &pgs[3]], &[]);
        let all = g.node("Add", &[&pgs[0], &board], &[]);
        let (hr, hs) = emit_counts(&mut g, &pgs[0]);
        let (br, bs) = emit_counts(&mut g, &board);
        let (ar, asu) = emit_counts(&mut g, &all);
        let planes = g.node(
            "Concat",
            &[&pgs[0], &pgs[1], &pgs[2], &pgs[3], &board, &all, &hr, &hs, &br, &bs, &ar, &asu],
            &[Attr::Int("axis", 1)],
        );
        let planes = self.planes_fc.emit(&mut g, &planes);
        let planes = emit_gelu(&mut g, &planes);

        // --- history ---
        let tok = emit_slice(&mut g, &x, 1, HIST_OFFSET as i64, HIST_OFFSET as i64 + T * TOKEN_FEATS as i64);
        let tok = emit_reshape(&mut g, &tok, &[-1, T, TOKEN_FEATS as i64]);
        let valid = emit_slice(&mut g, &tok, 2, TOKEN_FEATS as i64 - 1, TOKEN_FEATS as i64); // [B,T,1]
        let valid = emit_reshape(&mut g, &valid, &[-1, T]);
        let one = g.scalar_f32(1.0);
        let big = g.scalar_f32(MASK_NEG as f32);
        let vm1 = g.node("Sub", &[&valid, &one], &[]);
        let neg = g.node("Mul", &[&vm1, &big], &[]); // [B,T]
        let pt = self.tok_proj.emit(&mut g, &tok);
        let pos = emit_tensor(&mut g, "hist_pos", &self.hist_pos);
        let mut h = g.node("Add", &[&pt, &pos], &[]);
        let addmask = emit_reshape(&mut g, &neg, &[-1, 1, 1, T]);
        for l in &self.hist_layers {
            h = l.emit(&mut g, &h, &addmask);
        }
        let h = self.hist_ln.emit(&mut g, &h);
        let qt = emit_tensor(&mut g, "readout_q", &self.readout_q.tr());
        let scores = g.node("MatMul", &[&h, &qt], &[]);
        let sc = g.scalar_f32((1.0 / (D_MODEL as f64).sqrt()) as f32);
        let scores = g.node("Mul", &[&scores, &sc], &[]);
        let neg3 = emit_unsqueeze(&mut g, &neg, -1);
        let scores = g.node("Add", &[&scores, &neg3], &[]);
        let att = g.node("Softmax", &[&scores], &[Attr::Int("axis", 1)]);
        let att_t = g.node("Transpose", &[&att], &[Attr::Ints("perm", vec![0, 2, 1])]);
        let readout = g.node("MatMul", &[&att_t, &h], &[]);
        let readout = emit_reshape(&mut g, &readout, &[-1, N_READOUT * D_MODEL]);
        let vm = emit_unsqueeze(&mut g, &valid, -1);
        let hv = g.node("Mul", &[&h, &vm], &[]);
        let num = emit_reduce_sum(&mut g, &hv, 1, false);
        let den = emit_reduce_sum(&mut g, &vm, 1, false);
        let den = g.node("Max", &[&den, &one], &[]);
        let mean = g.node("Div", &[&num, &den], &[]);
        let vm1b = g.node("Sub", &[&vm, &one], &[]);
        let negb = g.node("Mul", &[&vm1b, &big], &[]);
        let hm = g.node("Add", &[&h, &negb], &[]);
        let mx = g.node("ReduceMax", &[&hm], &[Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 0)]);
        let any = g.node("ReduceMax", &[&valid], &[Attr::Ints("axes", vec![1]), Attr::Int("keepdims", 1)]);
        let pooled = g.node("Concat", &[&mean, &mx, &readout], &[Attr::Int("axis", 1)]);
        let pooled = g.node("Mul", &[&pooled, &any], &[]);

        // --- trunk ---
        let stat = emit_slice(&mut g, &x, 1, STATIC_OFFSET as i64, INPUT_SIZE as i64);
        let tin = g.node("Concat", &[&stat, &cards, &planes, &pooled], &[Attr::Int("axis", 1)]);
        let mut t = self.stem.emit(&mut g, &tin);
        for b in &self.blocks {
            t = b.emit(&mut g, &t);
        }

        // --- head ---
        let hl = self.head_ln.emit(&mut g, &t);
        let hl = self.head_fc.emit(&mut g, &hl);
        g.node_named("Identity", &[&hl], &[], "logits");
        g.output("logits", FLOAT, &[Dim::Named("batch"), Dim::Fixed(NUM_ACTIONS as i64)]);
        g.model_bytes(OPSET)
    }
}

pub fn n_params(vs: &nn::VarStore) -> i64 {
    vs.trainable_variables().iter().map(|t| t.numel() as i64).sum()
}
