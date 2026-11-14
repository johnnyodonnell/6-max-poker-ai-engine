//! REINFORCE update over one on-policy cohort (one optimizer step, gradient
//! accumulated over micro-batches):
//!
//!   log pi(a|s) = log_softmax(masked type logits)[type]
//!               + [raise] * log TruncNormal(size; mu, sigma, lo, hi)
//!   advantage   = z - V(s).detach()
//!   loss        = -mean(advantage * log pi) + c_value * Huber(V, z)
//!                 - c_entropy * H(type) - c_sigma * mean(log sigma)
//!
//! AdamW is implemented directly on tensors so its moments can be saved in
//! checkpoints (tch's built-in optimizer state can't be serialized).

use std::collections::HashMap;

use tch::nn::VarStore;
use tch::{Device, Kind, Tensor};

use poker_core::actions::{NUM_TYPES, RAISE};
use poker_core::encode::INPUT_SIZE;

use crate::net::PokerNet;
use crate::selfplay::Row;

const MASK_NEG: f64 = 1e9;

pub struct TrainCfg {
    pub lr: f64,
    pub weight_decay: f64,
    pub c_value: f64,
    pub c_entropy: f64,
    pub c_sigma: f64,
    pub grad_clip: f64,
    pub micro_batch: usize,
}

/// The cohort as device tensors.
pub struct Batch {
    pub n: i64,
    x: Tensor,
    mask: Tensor,
    lo: Tensor,
    hi: Tensor,
    ty: Tensor,
    s: Tensor,
    z: Tensor,
    logp_b: Tensor,
}

impl Batch {
    pub fn from_rows(rows: &[Row], dev: Device) -> Batch {
        let n = rows.len();
        let mut x = Vec::with_capacity(n * INPUT_SIZE);
        let mut mask = Vec::with_capacity(n * NUM_TYPES);
        for r in rows {
            x.extend_from_slice(&r.x);
            mask.extend_from_slice(&r.mask);
        }
        let col = |f: &dyn Fn(&Row) -> f32| Tensor::from_slice(&rows.iter().map(f).collect::<Vec<f32>>()).to_device(dev);
        Batch {
            n: n as i64,
            x: Tensor::from_slice(&x).view([n as i64, INPUT_SIZE as i64]).to_device(dev),
            mask: Tensor::from_slice(&mask).view([n as i64, NUM_TYPES as i64]).to_device(dev),
            lo: col(&|r| r.lo),
            hi: col(&|r| r.hi),
            ty: Tensor::from_slice(&rows.iter().map(|r| r.ty as i64).collect::<Vec<i64>>()).to_device(dev),
            s: col(&|r| r.s),
            z: col(&|r| r.z),
            logp_b: col(&|r| r.logp),
        }
    }
}

/// log(Phi(b) - Phi(a)) for a < b, stable in both tails.
fn log_mass(a: &Tensor, b: &Tensor) -> Tensor {
    // Upper tail (a > 0): use Phi(-a) - Phi(-b) instead.
    let upper = a.gt(0.0);
    let (lo, hi) = (a.where_self(&upper.logical_not(), &(-b)), b.where_self(&upper.logical_not(), &(-a)));
    let lhi = hi.special_log_ndtr();
    let llo = lo.special_log_ndtr();
    let d = (&llo - &lhi).clamp_max(-1e-12); // log(Phi(lo)/Phi(hi)) < 0
    lhi + (-d.expm1()).log()
}

/// Per-row log pi(a|s) and the type distribution's log-probs, from net outputs.
fn log_probs(type_logits: &Tensor, size: &Tensor, mask: &Tensor, lo: &Tensor, hi: &Tensor, ty: &Tensor, s: &Tensor) -> (Tensor, Tensor, Tensor) {
    let masked = type_logits + (mask - 1.0) * MASK_NEG;
    let logp_all = masked.log_softmax(1, Kind::Float);
    let logp_ty = logp_all.gather(1, &ty.unsqueeze(1), false).squeeze_dim(1);
    let mu = size.select(1, 0);
    let sigma = size.select(1, 1);
    let sized = ty.eq(RAISE as i64).logical_and(&(hi - lo).gt(1e-6)).to_kind(Kind::Float);
    let zs = (s - &mu) / &sigma;
    let a = (lo - &mu) / &sigma;
    let b = (hi - &mu) / &sigma;
    let half_ln_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    let size_logp: Tensor = (&zs * &zs) * -0.5 - half_ln_2pi - sigma.log() - log_mass(&a, &b);
    // Zero (not NaN) on rows without a sized raise.
    let size_logp = size_logp.nan_to_num(0.0, 0.0, 0.0) * &sized;
    (logp_ty + size_logp, logp_all, sigma)
}

pub struct AdamW {
    pub step: i64,
    names: Vec<String>,
    m: Vec<Tensor>,
    v: Vec<Tensor>,
}

impl AdamW {
    const B1: f64 = 0.9;
    const B2: f64 = 0.999;
    const EPS: f64 = 1e-8;

    pub fn new(vs: &VarStore) -> AdamW {
        let vars = vs.variables();
        let mut names: Vec<String> = vars.keys().cloned().collect();
        names.sort();
        let m = names.iter().map(|k| vars[k].zeros_like()).collect();
        let v = names.iter().map(|k| vars[k].zeros_like()).collect();
        AdamW { step: 0, names, m, v }
    }

    /// Clip the global grad norm to `clip`, then take one AdamW step.
    /// Returns the pre-clip grad norm.
    pub fn step(&mut self, vs: &VarStore, lr: f64, wd: f64, clip: f64) -> f64 {
        let vars = vs.variables();
        let params: Vec<&Tensor> = self.names.iter().map(|k| &vars[k]).collect();
        tch::no_grad(|| {
            let sq: Vec<Tensor> = params.iter().map(|p| p.grad().pow_tensor_scalar(2).sum(Kind::Float)).collect();
            let norm = Tensor::stack(&sq, 0).sum(Kind::Float).sqrt().double_value(&[]);
            let scale = if norm > clip { clip / (norm + 1e-6) } else { 1.0 };
            self.step += 1;
            let bc1 = 1.0 - Self::B1.powi(self.step as i32);
            let bc2 = 1.0 - Self::B2.powi(self.step as i32);
            for (i, p) in params.iter().enumerate() {
                let g = p.grad() * scale;
                let m = &mut self.m[i];
                let v = &mut self.v[i];
                *m = &*m * Self::B1 + &g * (1.0 - Self::B1);
                *v = &*v * Self::B2 + (&g * &g) * (1.0 - Self::B2);
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

#[derive(Default, Debug)]
pub struct StepStats {
    pub loss: f64,
    pub policy: f64,
    pub value: f64,
    pub entropy: f64,
    pub sigma: f64,
    pub grad_norm: f64,
    /// mean |log pi(trainer) - log pi(sampler)| before the update: should be
    /// ~0 (on-policy sanity check).
    pub logp_gap: f64,
    /// mean(log pi_behavior - log pi_new) after the update: approx KL.
    pub kl: f64,
}

pub fn reinforce_step(net: &PokerNet, vs: &VarStore, opt: &mut AdamW, b: &Batch, cfg: &TrainCfg, lr: f64) -> StepStats {
    let n = b.n;
    let mut st = StepStats::default();
    vs.trainable_variables().iter().for_each(|p| {
        let mut p = p.shallow_clone();
        p.zero_grad();
    });
    let mb = cfg.micro_batch as i64;
    let raise_rows = b.mask.select(1, RAISE as i64).sum(Kind::Float).double_value(&[]).max(1.0);
    let mut start = 0;
    while start < n {
        let len = mb.min(n - start);
        let sl = |t: &Tensor| t.narrow(0, start, len);
        let out = net.forward(&sl(&b.x));
        let (logp, logp_all, sigma) = log_probs(&out.type_logits, &out.size_params, &sl(&b.mask), &sl(&b.lo), &sl(&b.hi), &sl(&b.ty), &sl(&b.s));
        let z = sl(&b.z);
        let adv = &z - out.value.detach();
        let policy = -(adv * &logp).sum(Kind::Float) / n as f64;
        // Huber: chip results are heavy-tailed (multiway all-ins), and the
        // value gradient shares the trunk with the policy.
        let value = out.value.huber_loss(&z, tch::Reduction::Sum, 1.0) / n as f64;
        let entropy = -(logp_all.exp() * &logp_all).sum(Kind::Float) / n as f64;
        let raise_legal = sl(&b.mask).select(1, RAISE as i64);
        let logsig = (sigma.log() * &raise_legal).sum(Kind::Float) / raise_rows;
        let loss = &policy + cfg.c_value * &value - cfg.c_entropy * &entropy - cfg.c_sigma * &logsig;
        loss.backward();
        st.loss += loss.double_value(&[]);
        st.policy += policy.double_value(&[]);
        st.value += value.double_value(&[]);
        st.entropy += entropy.double_value(&[]);
        st.sigma += (sigma.detach() * &raise_legal).sum(Kind::Float).double_value(&[]) / raise_rows;
        st.logp_gap += (logp.detach() - sl(&b.logp_b)).abs().sum(Kind::Float).double_value(&[]) / n as f64;
        start += len;
    }
    st.grad_norm = opt.step(vs, lr, cfg.weight_decay, cfg.grad_clip);

    // Approximate KL(behavior || new) on the cohort.
    tch::no_grad(|| {
        let mut start = 0;
        while start < n {
            let len = mb.min(n - start);
            let sl = |t: &Tensor| t.narrow(0, start, len);
            let out = net.forward(&sl(&b.x));
            let (logp, _, _) = log_probs(&out.type_logits, &out.size_params, &sl(&b.mask), &sl(&b.lo), &sl(&b.hi), &sl(&b.ty), &sl(&b.s));
            st.kl += (sl(&b.logp_b) - logp).sum(Kind::Float).double_value(&[]) / n as f64;
            start += len;
        }
    });
    st
}
