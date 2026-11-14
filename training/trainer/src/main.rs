//! `poker` — the 6-max poker trainer CLI.
//!
//! Subcommands:
//!   gpu-check   confirm libtorch sees the GPU and can train a tiny net on it

use tch::nn::{self, Module, OptimizerConfig};
use tch::{Device, Kind, Tensor};

/// Phase-0 smoke test: CUDA visible, and a small MLP's loss drops under AdamW
/// with forward + backward on the GPU.
fn gpu_check() -> bool {
    let cuda = tch::Cuda::is_available();
    println!("cuda available: {cuda} (devices: {})", tch::Cuda::device_count());
    let dev = if cuda { Device::Cuda(0) } else { Device::Cpu };

    let vs = nn::VarStore::new(dev);
    let root = vs.root();
    let net = nn::seq()
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
        let loss = net.forward(&x).mse_loss(&y, tch::Reduction::Mean);
        opt.backward_step(&loss);
        let l = loss.double_value(&[]);
        if step == 0 {
            first = l;
        }
        last = l;
    }
    println!("device {dev:?}: loss {first:.4} -> {last:.4}");
    let ok = cuda && last < 0.5 * first;
    println!("{}", if ok { "GPU-CHECK OK" } else { "GPU-CHECK FAILED" });
    ok
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("gpu-check") => std::process::exit(if gpu_check() { 0 } else { 1 }),
        other => {
            eprintln!("unknown subcommand {other:?}; expected: gpu-check");
            std::process::exit(2);
        }
    }
}
