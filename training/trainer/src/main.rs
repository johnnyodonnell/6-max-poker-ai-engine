//! `poker` — the 6-max poker trainer CLI.
//!
//! Subcommands:
//!   gpu-check                       libtorch sees the GPU and can train on it
//!   export --out m.onnx [--weights w.safetensors | --init SEED]
//!          [--fixture fx.json] [--n 256]
//!                                   write the browser ONNX (+ a parity fixture:
//!                                   real encoded positions and tch's outputs)
//!   train --run-dir DIR [...]       the self-play REINFORCE daemon (see run.rs)
//!   eval / match                    bb/100 evaluation and the pool ladder (eval.rs)

mod eval;
mod hud;
mod net;
mod onnx;
mod run;
mod selfplay;
mod train;
mod util;

use tch::nn::{self, Module, OptimizerConfig};
use tch::{Device, Kind, Tensor};

use util::{flag, has_flag, random_positions};

/// Phase-0 smoke test: CUDA visible, and a small MLP's loss drops under AdamW
/// with forward + backward on the GPU.
fn gpu_check() -> bool {
    let cuda = tch::Cuda::is_available();
    println!("cuda available: {cuda} (devices: {})", tch::Cuda::device_count());
    let dev = if cuda { Device::Cuda(0) } else { Device::Cpu };

    let vs = nn::VarStore::new(dev);
    let root = vs.root();
    let mlp = nn::seq()
        .add(nn::linear(&root / "l1", 64, 256, Default::default()))
        .add_fn(|x| x.gelu("none"))
        .add(nn::linear(&root / "l2", 256, 1, Default::default()));
    let mut opt = nn::AdamW::default().build(&vs, 1e-3).expect("optimizer");

    tch::manual_seed(0);
    let x = Tensor::randn([8192, 64], (Kind::Float, dev));
    let y = x.sum_dim_intlist(1, true, Kind::Float).sin();
    let mut first = f64::NAN;
    let mut last = f64::NAN;
    for step in 0..300 {
        let loss = mlp.forward(&x).mse_loss(&y, tch::Reduction::Mean);
        opt.backward_step(&loss);
        let l = loss.double_value(&[]);
        if step == 0 {
            first = l;
        }
        last = l;
    }
    println!("device {dev:?}: loss {first:.4} -> {last:.4}");

    // The poker net itself: forward on the GPU.
    let vs2 = nn::VarStore::new(dev);
    let pnet = net::PokerNet::new(&vs2.root());
    let xs = Tensor::from_slice(&random_positions(256, 1).concat())
        .view([256, poker_core::encode::INPUT_SIZE as i64])
        .to_device(dev);
    let out = pnet.forward(&xs);
    println!(
        "poker net: {} params, outputs {:?} {:?} {:?}",
        net::n_params(&vs2),
        out.type_logits.size(),
        out.size_params.size(),
        out.value.size()
    );

    let ok = cuda && last < 0.5 * first;
    println!("{}", if ok { "GPU-CHECK OK" } else { "GPU-CHECK FAILED" });
    ok
}

fn export(args: &[String]) {
    let out = flag(args, "--out").expect("--out model.onnx");
    let mut vs = nn::VarStore::new(Device::Cpu);
    if let Some(seed) = flag(args, "--init") {
        tch::manual_seed(seed.parse().expect("--init SEED"));
    }
    let pnet = net::PokerNet::new(&vs.root());
    if let Some(w) = flag(args, "--weights") {
        vs.load(&w).unwrap_or_else(|e| panic!("load {w}: {e}"));
    } else if !has_flag(args, "--init") {
        panic!("pass --weights w.safetensors or --init SEED");
    }
    std::fs::write(&out, pnet.export_onnx()).expect("write onnx");
    println!("wrote {out} ({} params)", net::n_params(&vs));

    if let Some(fx) = flag(args, "--fixture") {
        let n: usize = flag(args, "--n").map(|v| v.parse().unwrap()).unwrap_or(256);
        let rows = random_positions(n, 12345);
        let xs = Tensor::from_slice(&rows.concat()).view([n as i64, poker_core::encode::INPUT_SIZE as i64]);
        let o = tch::no_grad(|| pnet.forward(&xs));
        let v = |t: &Tensor| Vec::<f32>::try_from(&t.contiguous().view([-1])).unwrap();
        let json = serde_json::json!({
            "n": n,
            "inputs": rows,
            "type_logits": v(&o.type_logits),
            "size_params": v(&o.size_params),
            "value": v(&o.value),
        });
        std::fs::write(&fx, json.to_string()).expect("write fixture");
        println!("wrote {fx} ({n} positions)");
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gpu-check") => std::process::exit(if gpu_check() { 0 } else { 1 }),
        Some("export") => export(&args),
        Some("train") => run::train(&args),
        Some("eval") => eval::eval_cmd(&args),
        Some("match") => eval::match_cmd(&args),
        other => {
            eprintln!("unknown subcommand {other:?}; expected: gpu-check | export | train | eval | match");
            std::process::exit(2);
        }
    }
}
