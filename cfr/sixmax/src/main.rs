//! `sixmax`: the 6-max CFR engine's command line. Built and run on the box
//! (see cfr/README.md).
//!
//!   abstraction --name NAME --buckets 169,F,T,R [--fit-boards 1500,2000,2000] [--seed 1]
//!       card groups -> $SIXMAX_HOME/data/NAME
//!   abscheck --name NAME
//!       every (hand, board) has a card group; group sizes
//!   tree --players 6|2 --stack CHIPS [--menu v1] [--buckets 169,200,200,200]
//!       the size of a betting tree
//!   train / export   (see train.rs)
//!   eval / probe     (see play.rs)

mod play;
mod train;
mod util;

use std::time::Instant;

use sixmax_cfr::menu::Menu;
use sixmax_cfr::tree;
use util::{flag, list, log, num, sixmax_home};

fn abstraction_cmd(args: &[String]) {
    let name = flag(args, "--name").expect("--name NAME");
    let b = list(args, "--buckets").expect("--buckets 169,F,T,R");
    let fit = list(args, "--fit-boards").unwrap_or(vec![1500, 2000, 2000]);
    let seed: u64 = num(args, "--seed", 1);
    let dir = sixmax_home().join("data").join(&name);
    log(format!("building abstraction {name} {b:?} (fit boards {fit:?}) -> {}", dir.display()));
    let t = Instant::now();
    let abs = sixmax_cfr::abs::build(&name, [b[0], b[1], b[2], b[3]], [fit[0] as usize, fit[1] as usize, fit[2] as usize], seed, &log);
    abs.save(&dir).expect("save abstraction");
    log(format!("done in {:.1} min", t.elapsed().as_secs_f64() / 60.0));
}

fn abscheck_cmd(args: &[String]) {
    let name = flag(args, "--name").expect("--name NAME");
    let abs = sixmax_cfr::abs::Abstraction::load(&sixmax_home().join("data").join(&name)).expect("load abstraction");
    let b = abs.buckets();
    // Hands that share a card with the board have no group: C(52,2) - C(52-k,2) per board.
    let hands = |k: usize| 1326 - (52 - k) * (51 - k) / 2;
    for (street, table, k, boards) in [("flop", &abs.flop, 3, 22100usize), ("turn", &abs.turn, 4, 270725), ("river", &abs.river, 5, 2598960)] {
        let nb = b[[0, 1, 2, 3][k - 2]] as usize;
        let mut sizes = vec![0u64; nb];
        let mut none = 0u64;
        for &g in table.iter() {
            if g == u16::MAX {
                none += 1;
            } else {
                assert!((g as usize) < nb, "{street}: group {g} out of range");
                sizes[g as usize] += 1;
            }
        }
        assert_eq!(none, (hands(k) * boards) as u64, "{street}: wrong number of hands without a group");
        let (min, max) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
        log(format!("{street}: {} hands on {boards} boards in {nb} groups (smallest {min}, largest {max}); every valid hand has one", table.len() as u64 - none));
    }
}

fn tree_cmd(args: &[String]) {
    let players: usize = num(args, "--players", 6);
    let stack: i32 = num(args, "--stack", 200);
    let menu = Menu::by_name(&flag(args, "--menu").unwrap_or("v1".into())).expect("unknown menu");
    let b = list(args, "--buckets").unwrap_or(vec![169, 200, 200, 200]);
    let t0 = Instant::now();
    let t = tree::build(players, stack, &menu, [b[0], b[1], b[2], b[3]]);
    let acts = t.street_actions();
    log(format!(
        "{players} players, {}bb, menu {}: decision nodes {:?}, actions {:?}, terminals {}, entries {} ({:.2} GiB at 4 B), built in {:.1}s",
        stack / 2,
        menu.name,
        t.street_counts(),
        acts,
        t.terms.len(),
        t.n_entries,
        t.n_entries as f64 * 4.0 / (1u64 << 30) as f64,
        t0.elapsed().as_secs_f64()
    ));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("abstraction") => abstraction_cmd(&args),
        Some("tree") => tree_cmd(&args),
        Some("abscheck") => abscheck_cmd(&args),
        Some("train") => train::train_cmd(&args),
        Some("export") => train::export_cmd(&args),
        Some("eval") => play::eval_cmd(&args),
        Some("probe") => play::probe_cmd(&args),
        other => {
            eprintln!("unknown subcommand {other:?}; expected: abstraction | abscheck | tree | train | export | eval | probe");
            std::process::exit(2);
        }
    }
}
