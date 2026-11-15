//! The policy update over one cohort: clipped PPO with all-actions
//! advantages from a training-only critic, a KL limit, and a pull toward an
//! anchor policy. The critic is trained alongside.
//!
//! Advantages (per row; the critic's Q is detached):
//!   Q^(b)  = critic Q(s, h, b) for every legal action b — except fold, whose
//!            value is known exactly (the chips already put in, lost)
//!   V      = sum_b p(b) Q^(b)             (p = the behavior policy)
//!   A_s    = z - Q^(a)                    the sampled action's residual
//!   A(b)   = Q^(b) - V                    every legal action's advantage
//! The policy gradient  grad log pi(a) A_s + sum_b grad pi(b) A(b)  is unbiased
//! for any Q^ (Q-Prop's action-dependent control variate): the critic only
//! has to be good to cut variance, and fold's term is exact. Both terms get
//! PPO's clipping (the second one per action, as a clip of pi(b) to
//! p(b)(1 +- eps)), and advantages are scaled by the batch std of z - V.
//!
//!   loss = -(clipped surrogate) - c_entropy H(pi) + c_anchor KL(anchor || pi)
//!
//! The forward KL to the anchor (the last accepted snapshot, or the initial
//! policy) keeps every action the anchor plays alive — its pull on a dying
//! action is the anchor's probability, not ~0 like the entropy bonus's — and
//! bounds how far self-play can drift from a known-good strategy.
//!
//! Epochs over the cohort in minibatches stop early once KL(behavior || pi)
//! exceeds 1.5 x target_kl. AdamW is implemented directly on tensors so its
//! moments can be saved in checkpoints.

use std::collections::HashMap;

use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};

use poker_core::actions::{A_FOLD, NUM_ACTIONS};
use poker_core::encode::INPUT_SIZE;

use crate::net::{PokerNet, PRIV_SIZE};
use crate::selfplay::Row;

const MASK_NEG: f64 = 1e9;
const NA: i64 = NUM_ACTIONS as i64;

/// Which actions get the critic's all-actions term.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ControlVariate {
    /// Every legal action (fold exact, the rest from the critic's Q).
    All,
    /// Fold only (exact); other actions use the critic's state value.
    Fold,
    /// Plain PPO with the critic's state value as the baseline.
    None,
}

pub struct TrainCfg {
    pub lr: f64,
    pub critic_lr: f64,
    pub weight_decay: f64,
    pub beta2: f64,
    pub c_entropy: f64,
    pub c_anchor: f64,
    pub clip: f64,
    pub target_kl: f64,
    pub epochs: usize,
    pub minibatch: usize,
    pub micro_batch: usize,
    pub grad_clip: f64,
    pub cv: ControlVariate,
}

/// The cohort as device tensors.
pub struct Batch {
    pub n: i64,
    x: Tensor,
    p: Tensor,
    amask: Tensor,
    probs: Tensor,
    a: Tensor,
    fold_q: Tensor,
    z: Tensor,
}

impl Batch {
    pub fn from_rows(rows: &[Row], dev: Device) -> Batch {
        let n = rows.len() as i64;
        let mut x = Vec::with_capacity(rows.len() * INPUT_SIZE);
        let mut p = Vec::with_capacity(rows.len() * PRIV_SIZE);
        let mut amask = Vec::with_capacity(rows.len() * NUM_ACTIONS);
        let mut probs = Vec::with_capacity(rows.len() * NUM_ACTIONS);
        for r in rows {
            x.extend_from_slice(&r.x);
            p.extend_from_slice(&r.p);
            amask.extend_from_slice(&r.amask);
            probs.extend_from_slice(&r.probs);
        }
        let col = |f: &dyn Fn(&Row) -> f32| Tensor::from_slice(&rows.iter().map(f).collect::<Vec<f32>>()).to_device(dev);
        Batch {
            n,
            x: Tensor::from_slice(&x).view([n, INPUT_SIZE as i64]).to_device(dev),
            p: Tensor::from_slice(&p).view([n, PRIV_SIZE as i64]).to_device(dev),
            amask: Tensor::from_slice(&amask).view([n, NA]).to_device(dev),
            probs: Tensor::from_slice(&probs).view([n, NA]).to_device(dev),
            a: Tensor::from_slice(&rows.iter().map(|r| r.a as i64).collect::<Vec<i64>>()).to_device(dev),
            fold_q: col(&|r| r.fold_q),
            z: col(&|r| r.z),
        }
    }
}

fn masked_log_softmax(logits: &Tensor, amask: &Tensor) -> Tensor {
    (logits + (amask - 1.0) * MASK_NEG).log_softmax(1, Kind::Float)
}

pub struct AdamW {
    pub step: i64,
    b1: f64,
    b2: f64,
    names: Vec<String>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    const EPS: f64 = 1e-8;

    pub fn new(vs: &VarStore, b1: f64, b2: f64) -> AdamW {
        let vars = vs.variables();
        let mut names: Vec<String> = vars.keys().cloned().collect();
        names.sort();
        let m = names.iter().map(|k| vars[k].zeros_like()).collect();
        let v = names.iter().map(|k| vars[k].zeros_like()).collect();
        AdamW { step: 0, b1, b2, names, m, v }
    }

    /// Clip the global grad norm to `clip`, then take one AdamW step.
    /// Returns the pre-clip grad norm.
    pub fn step(&mut self, vs: &VarStore, lr: f64, wd: f64, clip: f64) -> f64 {
        let vars = vs.variables();
        let params: Vec<&Tensor> = self.names.iter().map(|k| &vars[k]).collect();
        tch::no_grad(|| {
            let sq: Vec<Tensor> =
                params.iter().filter(|p| p.grad().defined()).map(|p| p.grad().pow_tensor_scalar(2).sum(Kind::Float)).collect();
            let norm = Tensor::stack(&sq, 0).sum(Kind::Float).sqrt().double_value(&[]);
            let scale = if norm > clip { clip / (norm + 1e-6) } else { 1.0 };
            self.step += 1;
            let bc1 = 1.0 - self.b1.powi(self.step as i32);
            let bc2 = 1.0 - self.b2.powi(self.step as i32);
            for (i, p) in params.iter().enumerate() {
                if !p.grad().defined() {
                    continue;
                }
                let g = p.grad() * scale;
                let m = &mut self.m[i];
                let v = &mut self.v[i];
                *m = &*m * self.b1 + &g * (1.0 - self.b1);
                *v = &*v * self.b2 + (&g * &g) * (1.0 - self.b2);
                let upd = (&*m / bc1) / ((&*v / bc2).sqrt() + Self::EPS);
                let mut p = p.shallow_clone();
                let _ = p.f_mul_scalar_(1.0 - lr * wd).unwrap();
                let _ = p.f_sub_(&(upd * lr)).unwrap();
            }
            norm
        })
    }

    pub fn save(&self, path: &str) {
        let mut named: Vec<(String, Tensor)> = Vec::new();
        for (i, k) in self.names.iter().enumerate() {
            named.push((format!("m.{k}"), self.m[i].to_device(Device::Cpu)));
            named.push((format!("v.{k}"), self.v[i].to_device(Device::Cpu)));
        }
        named.push(("step".into(), Tensor::from_slice(&[self.step])));
        let refs: Vec<(&str, &Tensor)> = named.iter().map(|(k, t)| (k.as_str(), t)).collect();
        Tensor::write_safetensors(&refs, path).expect("save optimizer");
    }

    pub fn load(&mut self, path: &str, dev: Device) {
        let map: HashMap<String, Tensor> = Tensor::read_safetensors(path).expect("load optimizer").into_iter().collect();
        for (i, k) in self.names.iter().enumerate() {
            self.m[i] = map[&format!("m.{k}")].to_device(dev);
            self.v[i] = map[&format!("v.{k}")].to_device(dev);
        }
        self.step = map["step"].int64_value(&[0]);
    }
}

fn zero_grads(vs: &VarStore) {
    for p in vs.trainable_variables() {
        let mut p = p;
        p.zero_grad();
    }
}

fn grad_sq(vs: &VarStore) -> f64 {
    tch::no_grad(|| {
        vs.trainable_variables()
            .iter()
            .filter(|p| p.grad().defined())
            .map(|p| p.grad().pow_tensor_scalar(2).sum(Kind::Float).double_value(&[]))
            .sum()
    })
}

#[derive(Default, Debug)]
pub struct StepStats {
    pub policy: f64,
    pub entropy: f64,
    pub anchor_kl: f64,
    /// KL(behavior || pi) after the update.
    pub kl: f64,
    pub clip_frac: f64,
    pub epochs: f64,
    pub steps: usize,
    pub grad_norm: f64,
    pub critic_loss: f64,
    /// 1 - Var(z - V) / Var(z): how much of the result variance the critic's
    /// baseline explains.
    pub critic_ev: f64,
    /// Gradient noise of the policy term (McCandlish et al.), measured on
    /// one minibatch: tr(Sigma) per row and |G|^2. B_noise = tr / g2.
    pub noise_tr: f64,
    pub noise_g2: f64,
}

/// Per-row advantages, computed once per cohort (no grad).
struct Adv {
    sampled: Tensor, // [N]
    all: Tensor,     // [N, A]
}

fn advantages(critic: &PokerNet, b: &Batch, cfg: &TrainCfg, st: &mut StepStats) -> Adv {
    tch::no_grad(|| {
        let mb = cfg.micro_batch as i64;
        let mut qs = Vec::new();
        let mut start = 0;
        while start < b.n {
            let len = mb.min(b.n - start);
            qs.push(critic.forward_priv(&b.x.narrow(0, start, len), &b.p.narrow(0, start, len)));
            start += len;
        }
        let q = Tensor::cat(&qs, 0); // [N,A]
        let fold_col = Tensor::zeros([NA], (Kind::Float, q.device())).index_fill(0, &Tensor::from_slice(&[A_FOLD as i64]).to_device(q.device()), 1.0);
        let state_v = (&b.probs * &q).sum_dim_intlist(1, false, Kind::Float); // critic V under behavior
        let qhat = match cfg.cv {
            ControlVariate::All => &q * (1.0 - &fold_col) + b.fold_q.unsqueeze(1) * &fold_col,
            ControlVariate::Fold => state_v.unsqueeze(1) * (1.0 - &fold_col) + b.fold_q.unsqueeze(1) * &fold_col,
            ControlVariate::None => state_v.unsqueeze(1).expand([b.n, NA], false).copy(),
        };
        let v = (&b.probs * &qhat).sum_dim_intlist(1, false, Kind::Float);
        let q_a = qhat.gather(1, &b.a.unsqueeze(1), false).squeeze_dim(1);
        let resid = &b.z - &v;
        let scale = resid.std(true).double_value(&[]).max(1e-6);
        let var_z = b.z.var(true).double_value(&[]).max(1e-12);
        st.critic_ev = 1.0 - resid.var(true).double_value(&[]) / var_z;
        Adv { sampled: (&b.z - q_a) / scale, all: (&qhat - v.unsqueeze(1)) * &b.amask / scale }
    })
}

struct PolicyTerms {
    surrogate: Tensor,
    entropy: Tensor,
    anchor_kl: Tensor,
    kl_b: Tensor,
    clipped: Tensor,
}

/// Summed (not averaged) per-row terms of the policy objective on a slice.
fn policy_terms(net: &PokerNet, b: &Batch, adv: &Adv, rho: &Tensor, idx: &Tensor, clip: f64) -> PolicyTerms {
    let sel = |t: &Tensor| t.index_select(0, idx);
    let amask = sel(&b.amask);
    let probs = sel(&b.probs);
    let a = sel(&b.a);
    let logp = masked_log_softmax(&net.forward(&sel(&b.x)), &amask);
    let pi = logp.exp();
    // Sampled action: ratio clip.
    let lp_a = logp.gather(1, &a.unsqueeze(1), false).squeeze_dim(1);
    let pb_a = probs.gather(1, &a.unsqueeze(1), false).squeeze_dim(1);
    let ratio = (&lp_a - pb_a.log()).exp();
    let a_s = sel(&adv.sampled);
    let s1 = &ratio * &a_s;
    let s2 = ratio.clamp(1.0 - clip, 1.0 + clip) * &a_s;
    let sampled = s1.minimum(&s2);
    // All actions: pi(b) clipped to p(b)(1 +- clip).
    let a_all = sel(&adv.all);
    let lo = &probs * (1.0 - clip);
    let hi = &probs * (1.0 + clip);
    let e1 = &pi * &a_all;
    let e2 = pi.maximum(&lo).minimum(&hi) * &a_all;
    let expected = e1.minimum(&e2).sum_dim_intlist(1, false, Kind::Float);
    let clipped = ratio.gt(1.0 + clip).logical_or(&ratio.lt(1.0 - clip)).to_kind(Kind::Float).sum(Kind::Float);
    let entropy = -(&pi * &logp * &amask).sum(Kind::Float);
    let rho = sel(rho);
    let anchor_kl = (&rho * ((&rho + 1e-12).log() - &logp) * &amask).sum(Kind::Float);
    let kl_b = (&probs * ((&probs + 1e-12).log() - &logp) * &amask).sum(Kind::Float);
    PolicyTerms { surrogate: (sampled + expected).sum(Kind::Float), entropy, anchor_kl, kl_b, clipped }
}

/// One cohort's update of the policy (PPO epochs) and the critic.
#[allow(clippy::too_many_arguments)]
pub fn ppo_update(
    policy: &PokerNet,
    pvs: &VarStore,
    popt: &mut AdamW,
    critic: &PokerNet,
    cvs: &VarStore,
    copt: &mut AdamW,
    anchor: &PokerNet,
    b: &Batch,
    cfg: &TrainCfg,
    lr: f64,
    seed: i64,
    measure_noise: bool,
) -> StepStats {
    let mut st = StepStats::default();
    let dev = b.x.device();
    let n = b.n;
    let mb = cfg.micro_batch as i64;
    let adv = advantages(critic, b, cfg, &mut st);
    // Anchor policy on every row (no grad).
    let rho = tch::no_grad(|| {
        let mut out = Vec::new();
        let mut start = 0;
        while start < n {
            let len = mb.min(n - start);
            out.push(masked_log_softmax(&anchor.forward(&b.x.narrow(0, start, len)), &b.amask.narrow(0, start, len)).exp());
            start += len;
        }
        Tensor::cat(&out, 0)
    });

    // Gradient noise of the policy surrogate on the first minibatch: per
    // micro-batch gradients (size m) and their mean (size k*m).
    if measure_noise {
        tch::manual_seed(seed);
        let k = (cfg.minibatch as i64 / mb).max(2).min(n / mb);
        if k >= 2 {
            let perm = Tensor::randperm(n, (Kind::Int64, dev));
            let mut sum_sq = 0.0;
            let mut acc: Option<Vec<Tensor>> = None;
            for j in 0..k {
                zero_grads(pvs);
                let idx = perm.narrow(0, j * mb, mb);
                let t = policy_terms(policy, b, &adv, &rho, &idx, cfg.clip);
                (-t.surrogate / mb as f64).backward();
                sum_sq += grad_sq(pvs);
                let gs: Vec<Tensor> = pvs.trainable_variables().iter().map(|p| if p.grad().defined() { p.grad().copy() } else { p.zeros_like() }).collect();
                acc = Some(match acc {
                    None => gs,
                    Some(a) => a.iter().zip(gs.iter()).map(|(x, y)| x + y).collect(),
                });
            }
            let mean_small = sum_sq / k as f64;
            let big: f64 = acc.unwrap().iter().map(|g| (g / k as f64).pow_tensor_scalar(2).sum(Kind::Float).double_value(&[])).sum();
            let (bs, bb) = (mb as f64, (k * mb) as f64);
            let tr = (mean_small - big) / (1.0 / bs - 1.0 / bb);
            st.noise_tr = tr;
            st.noise_g2 = big - tr / bb;
            zero_grads(pvs);
        }
    }

    // PPO epochs over shuffled minibatches.
    let mut stop = false;
    let (mut pol_sum, mut ent_sum, mut akl_sum, mut clip_sum, mut rows_seen) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let mut critic_sum = 0.0;
    let mut critic_rows = 0.0;
    let mut epochs_done = 0.0;
    for epoch in 0..cfg.epochs {
        tch::manual_seed(seed.wrapping_add(1 + epoch as i64));
        let perm = Tensor::randperm(n, (Kind::Int64, dev));
        let n_mb = (n as usize).div_ceil(cfg.minibatch);
        for j in 0..n_mb {
            let s0 = (j * cfg.minibatch) as i64;
            let len = (cfg.minibatch as i64).min(n - s0);
            let idx = perm.narrow(0, s0, len);
            // --- policy ---
            if !stop {
                zero_grads(pvs);
                let mut kl_b = 0.0;
                let mut m0 = 0;
                while m0 < len {
                    let ml = mb.min(len - m0);
                    let sub = idx.narrow(0, m0, ml);
                    let t = policy_terms(policy, b, &adv, &rho, &sub, cfg.clip);
                    let loss = (-&t.surrogate - cfg.c_entropy * &t.entropy + cfg.c_anchor * &t.anchor_kl) / len as f64;
                    loss.backward();
                    pol_sum += -t.surrogate.double_value(&[]);
                    ent_sum += t.entropy.double_value(&[]);
                    akl_sum += t.anchor_kl.double_value(&[]);
                    clip_sum += t.clipped.double_value(&[]);
                    kl_b += t.kl_b.double_value(&[]);
                    rows_seen += ml as f64;
                    m0 += ml;
                }
                kl_b /= len as f64;
                if (epoch > 0 || j > 0) && kl_b > 1.5 * cfg.target_kl {
                    stop = true; // drift limit reached before this step
                } else {
                    st.grad_norm = popt.step(pvs, lr, cfg.weight_decay, cfg.grad_clip);
                    st.steps += 1;
                }
            }
            // --- critic: Huber regression of Q(s, h, a) on z, one pass per cohort ---
            if epoch > 0 {
                continue;
            }
            zero_grads(cvs);
            let mut m0 = 0;
            while m0 < len {
                let ml = mb.min(len - m0);
                let sub = idx.narrow(0, m0, ml);
                let q = critic.forward_priv(&b.x.index_select(0, &sub), &b.p.index_select(0, &sub));
                let q_a = q.gather(1, &b.a.index_select(0, &sub).unsqueeze(1), false).squeeze_dim(1);
                let loss = q_a.huber_loss(&b.z.index_select(0, &sub), tch::Reduction::Sum, 1.0) / len as f64;
                loss.backward();
                critic_sum += loss.double_value(&[]) * len as f64;
                critic_rows += ml as f64;
                m0 += ml;
            }
            copt.step(cvs, cfg.critic_lr, cfg.weight_decay, cfg.grad_clip);
        }
        epochs_done += if stop { 0.0 } else { 1.0 };
        if stop {
            break;
        }
    }
    let r = rows_seen.max(1.0);
    st.policy = pol_sum / r;
    st.entropy = ent_sum / r;
    st.anchor_kl = akl_sum / r;
    st.clip_frac = clip_sum / r;
    st.epochs = epochs_done;
    st.critic_loss = critic_sum / critic_rows.max(1.0);

    // KL(behavior || pi) after the update.
    st.kl = tch::no_grad(|| {
        let mut tot = 0.0;
        let mut start = 0;
        while start < n {
            let len = mb.min(n - start);
            let idx = Tensor::arange_start(start, start + len, (Kind::Int64, dev));
            tot += policy_terms(policy, b, &adv, &rho, &idx, cfg.clip).kl_b.double_value(&[]);
            start += len;
        }
        tot / n as f64
    });
    st
}
