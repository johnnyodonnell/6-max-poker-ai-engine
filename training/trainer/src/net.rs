//! The policy/value network, defined once from building blocks that each know
//! how to run in tch (`fwd`, used for self-play and training) and how to write
//! themselves into an ONNX graph (`emit`, used for the browser export). The
//! exported model is checked against `fwd` by training/scripts/test_onnx_parity.ts.
//!
//! Input: the flat encoding from poker_core::encode ([B, INPUT_SIZE]), sliced
//! into cards / history tokens / static features.
//!   cards   -> card+rank+suit embeddings, summed per group (hole, flop, turn,
//!              river), undealt slots masked to zero            -> [B, 4*D_CARD]
//!   history -> token projection + position embedding, pre-LN self-attention
//!              layers, pooled as masked mean + masked max + learned-query
//!              readouts, zeroed when there is no history       -> [B, (R+2)*D_MODEL]
//!   static  -> as is                                           -> [B, STATIC_SIZE]
//! concat -> Linear(WIDTH) -> ResBlocks -> heads:
//!   type_logits [B,4]   fold / check-call / raise / all-in (masked outside)
//!   size_params [B,2]   (mu, sigma) of the truncated normal over log pot-fraction
//!   value       [B]     expected chip result / 200

use tch::nn::{self, Module};
use tch::{Device, Kind, Tensor};

use poker_core::encode::{CARD_SLOTS, HIST_OFFSET, HIST_TOKENS, INPUT_SIZE, STATIC_OFFSET, STATIC_SIZE, TOKEN_FEATS};

use crate::onnx::{Attr, Dim, Graph, FLOAT, INT64};

pub const D_CARD: i64 = 64;
pub const D_MODEL: i64 = 128;
pub const N_HEADS: i64 = 4;
pub const HEAD_DIM: i64 = D_MODEL / N_HEADS;
pub const FFN: i64 = 256;
pub const N_HIST_LAYERS: usize = 2;
pub const N_READOUT: i64 = 4;
pub const WIDTH: i64 = 512;
pub const N_BLOCKS: usize = 4;
pub const VALUE_HIDDEN: i64 = 256;
pub const SIGMA_MIN: f64 = 0.05;
pub const SIGMA_MAX: f64 = 1.5;
/// Initial raise size center (pot fraction) and spread.
pub const MU_INIT: f64 = 0.6;
pub const SIGMA_INIT: f64 = 0.6;
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
    fn new(p: nn::Path) -> Self {
        ResBlock {
            ln1: LN::new(&p / "ln1", WIDTH),
            fc1: Lin::new(&p / "fc1", WIDTH, WIDTH),
            ln2: LN::new(&p / "ln2", WIDTH),
            fc2: Lin::new(&p / "fc2", WIDTH, WIDTH),
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

pub struct NetOutput {
    pub type_logits: Tensor, // [B,4]
    pub size_params: Tensor, // [B,2] (mu, sigma)
    pub value: Tensor,       // [B]
}

pub struct PokerNet {
    card_emb: Embed,
    rank_emb: Embed,
    suit_emb: Embed,
    tok_proj: Lin,
    hist_pos: Tensor,
    hist_layers: Vec<HistLayer>,
    hist_ln: LN,
    readout_q: Tensor,
    stem: Lin,
    blocks: Vec<ResBlock>,
    type_ln: LN,
    type_fc: Lin,
    size_ln: LN,
    size_fc: Lin,
    value_ln: LN,
    value_fc1: Lin,
    value_fc2: Lin,
}

const TRUNK_IN: i64 = STATIC_SIZE as i64 + 4 * D_CARD + (N_READOUT + 2) * D_MODEL;

/// Initial type-logit biases: fold, check/call, raise, all-in. All-in starts
/// unlikely so the untrained policy doesn't shove a third of the time.
const TYPE_BIAS_INIT: [f64; 4] = [0.0, 0.5, 0.0, -1.5];

impl PokerNet {
    pub fn new(vs: &nn::Path) -> Self {
        let net = PokerNet {
            card_emb: Embed::new(vs / "card_emb", NO_CARD + 1, D_CARD),
            rank_emb: Embed::new(vs / "rank_emb", 13, D_CARD),
            suit_emb: Embed::new(vs / "suit_emb", 4, D_CARD),
            tok_proj: Lin::new(vs / "tok_proj", TOKEN_FEATS as i64, D_MODEL),
            hist_pos: vs.var("hist_pos", &[T, D_MODEL], nn::Init::Randn { mean: 0.0, stdev: 0.02 }),
            hist_layers: (0..N_HIST_LAYERS).map(|i| HistLayer::new(vs / "hist_layers" / i)).collect(),
            hist_ln: LN::new(vs / "hist_ln", D_MODEL),
            readout_q: vs.var("readout_q", &[N_READOUT, D_MODEL], nn::Init::Randn { mean: 0.0, stdev: 0.02 }),
            stem: Lin::new(vs / "stem", TRUNK_IN, WIDTH),
            blocks: (0..N_BLOCKS).map(|i| ResBlock::new(vs / "blocks" / i)).collect(),
            type_ln: LN::new(vs / "type_ln", WIDTH),
            type_fc: Lin::new(vs / "type_fc", WIDTH, 4),
            size_ln: LN::new(vs / "size_ln", WIDTH),
            size_fc: Lin::new(vs / "size_fc", WIDTH, 2),
            value_ln: LN::new(vs / "value_ln", WIDTH),
            value_fc1: Lin::new(vs / "value_fc1", WIDTH, VALUE_HIDDEN),
            value_fc2: Lin::new(vs / "value_fc2", VALUE_HIDDEN, 1),
        };
        // Heads start near their biases: a mild preference among action types,
        // raises centered on MU_INIT pot with spread SIGMA_INIT.
        tch::no_grad(|| {
            let dev = net.type_fc.0.ws.device();
            let mut tw = net.type_fc.0.ws.shallow_clone();
            let _ = tw.f_mul_scalar_(0.01).unwrap();
            let mut sw = net.size_fc.0.ws.shallow_clone();
            let _ = sw.f_mul_scalar_(0.01).unwrap();
            let mut tb = net.type_fc.0.bs.as_ref().unwrap().shallow_clone();
            tb.copy_(&Tensor::from_slice(&TYPE_BIAS_INIT).to_kind(Kind::Float).to_device(dev));
            let frac = (SIGMA_INIT - SIGMA_MIN) / (SIGMA_MAX - SIGMA_MIN);
            let sb_init = [MU_INIT.ln(), (frac / (1.0 - frac)).ln()];
            let mut sb = net.size_fc.0.bs.as_ref().unwrap().shallow_clone();
            sb.copy_(&Tensor::from_slice(&sb_init).to_kind(Kind::Float).to_device(dev));
        });
        net
    }

    /// Size-head affine constants: params = raw * M + C0 + C1 * sigmoid(raw),
    /// i.e. mu = raw0, sigma = SIGMA_MIN + (SIGMA_MAX - SIGMA_MIN) * sigmoid(raw1).
    fn size_consts() -> ([f32; 2], [f32; 2], [f32; 2]) {
        ([1.0, 0.0], [0.0, SIGMA_MIN as f32], [0.0, (SIGMA_MAX - SIGMA_MIN) as f32])
    }

    pub fn forward(&self, x: &Tensor) -> NetOutput {
        let dev = x.device();
        // --- cards ---
        let idx = x.narrow(1, 0, CARD_SLOTS as i64).to_kind(Kind::Int64); // [B,7]
        let dealt = idx.lt(NO_CARD).to_kind(Kind::Float).unsqueeze(-1);
        let suit = idx.divide_scalar_mode(13, "floor").clamp_max(3);
        let rank = idx.remainder(13);
        let e = (self.card_emb.fwd(&idx) + self.rank_emb.fwd(&rank) + self.suit_emb.fwd(&suit)) * dealt;
        let group = |s: i64, n: i64| e.narrow(1, s, n).sum_dim_intlist(1, false, Kind::Float);
        let cards = Tensor::cat(&[group(0, 2), group(2, 3), group(5, 1), group(6, 1)], 1);

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
        let mut t = self.stem.fwd(&Tensor::cat(&[stat, cards, pooled], 1));
        for b in &self.blocks {
            t = b.fwd(&t);
        }

        // --- heads ---
        let type_logits = self.type_fc.fwd(&self.type_ln.fwd(&t));
        let raw = self.size_fc.fwd(&self.size_ln.fwd(&t));
        let (m, c0, c1) = Self::size_consts();
        let f = |a: [f32; 2]| Tensor::from_slice(&a).to_device(dev);
        let size_params = &raw * f(m) + f(c0) + f(c1) * raw.sigmoid();
        let value = self
            .value_fc2
            .fwd(&gelu(&self.value_fc1.fwd(&self.value_ln.fwd(&t))))
            .view([-1]);
        NetOutput { type_logits, size_params, value }
    }

    /// Serialize the network (current weights) as a self-contained ONNX model:
    /// input "input" [batch, INPUT_SIZE]; outputs "type_logits" [batch,4],
    /// "size_params" [batch,2], "value" [batch].
    pub fn export_onnx(&self) -> Vec<u8> {
        let mut g = Graph::new();
        let x = g.input("input", FLOAT, &[Dim::Named("batch"), Dim::Fixed(INPUT_SIZE as i64)]);

        // --- cards ---
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
        let tin = g.node("Concat", &[&stat, &cards, &pooled], &[Attr::Int("axis", 1)]);
        let mut t = self.stem.emit(&mut g, &tin);
        for b in &self.blocks {
            t = b.emit(&mut g, &t);
        }

        // --- heads ---
        let tl = self.type_ln.emit(&mut g, &t);
        let tl = self.type_fc.emit(&mut g, &tl);
        g.node_named("Identity", &[&tl], &[], "type_logits");

        let sl = self.size_ln.emit(&mut g, &t);
        let raw = self.size_fc.emit(&mut g, &sl);
        let (m, c0, c1) = Self::size_consts();
        let m = g.init_f32("size_m", &[2], &m);
        let c0 = g.init_f32("size_c0", &[2], &c0);
        let c1 = g.init_f32("size_c1", &[2], &c1);
        let a = g.node("Mul", &[&raw, &m], &[]);
        let a = g.node("Add", &[&a, &c0], &[]);
        let sg = g.node("Sigmoid", &[&raw], &[]);
        let sg = g.node("Mul", &[&sg, &c1], &[]);
        g.node_named("Add", &[&a, &sg], &[], "size_params");

        let v = self.value_ln.emit(&mut g, &t);
        let v = self.value_fc1.emit(&mut g, &v);
        let v = emit_gelu(&mut g, &v);
        let v = self.value_fc2.emit(&mut g, &v);
        let shape = g.init_i64("shape", &[1], &[-1]);
        g.node_named("Reshape", &[&v, &shape], &[], "value");

        g.output("type_logits", FLOAT, &[Dim::Named("batch"), Dim::Fixed(4)]);
        g.output("size_params", FLOAT, &[Dim::Named("batch"), Dim::Fixed(2)]);
        g.output("value", FLOAT, &[Dim::Named("batch")]);
        g.model_bytes(OPSET)
    }
}

pub fn n_params(vs: &nn::VarStore) -> i64 {
    vs.trainable_variables().iter().map(|t| t.numel() as i64).sum()
}
