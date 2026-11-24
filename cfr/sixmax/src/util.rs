//! Small CLI helpers.

/// The value after `--name`, if present.
pub fn flag(args: &[String], name: &str) -> Option<String> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
}

/// A numeric flag with a default (accepts 1e6-style and 1_000_000 forms).
pub fn num<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    match flag(args, name) {
        None => default,
        Some(v) => {
            let v = v.replace('_', "");
            v.parse::<T>()
                .ok()
                .or_else(|| v.parse::<f64>().ok().and_then(|f| format!("{}", f as u64).parse::<T>().ok()))
                .unwrap_or_else(|| panic!("bad value for {name}: {v}"))
        }
    }
}

/// A comma-separated list of numbers, e.g. "2000,2000,1000".
pub fn list(args: &[String], name: &str) -> Option<Vec<u32>> {
    flag(args, name).map(|v| v.split(',').map(|x| x.trim().replace('_', "").parse().expect("number list")).collect())
}

pub fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1)
}

/// The project folder on the box ($SIXMAX_HOME, set by cfr/box_env.sh).
pub fn sixmax_home() -> std::path::PathBuf {
    std::env::var("SIXMAX_HOME").map(Into::into).unwrap_or_else(|_| ".".into())
}

/// Timestamped progress line.
pub fn log(msg: String) {
    use std::io::Write;
    let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let mut out = std::io::stdout().lock();
    let _ = writeln!(out, "[{:02}:{:02}:{:02}] {msg}", (t / 3600) % 24, (t / 60) % 60, t % 60);
    let _ = out.flush();
}

/// Resident memory of this process, GiB.
pub fn rss_gib() -> f64 {
    let s = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    s.lines()
        .find_map(|l| l.strip_prefix("VmRSS:"))
        .and_then(|r| r.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
        .map(|kb| kb / (1u64 << 20) as f64)
        .unwrap_or(0.0)
}
