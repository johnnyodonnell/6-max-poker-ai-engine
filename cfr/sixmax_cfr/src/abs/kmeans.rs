//! k-means with L1 distance on small u8 feature vectors.
//!
//! For cumulative histograms (the flop/turn features) L1 distance is the
//! earth mover's distance between the underlying distributions. Centers are
//! seeded with k-means++ on a subsample and refined with Lloyd iterations
//! (assign by L1, update to the mean).

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

#[inline]
pub fn l1<const D: usize>(a: &[u8; D], b: &[u8; D]) -> u32 {
    let mut s = 0u32;
    for i in 0..D {
        s += a[i].abs_diff(b[i]) as u32;
    }
    s
}

/// Index of the nearest center (and its distance).
#[inline]
pub fn nearest<const D: usize>(centers: &[[u8; D]], x: &[u8; D]) -> (u32, u32) {
    let mut best = (0u32, u32::MAX);
    for (i, c) in centers.iter().enumerate() {
        let d = l1(c, x);
        if d < best.1 {
            best = (i as u32, d);
        }
    }
    best
}

fn to_u8<const D: usize>(c: &[f64; D]) -> [u8; D] {
    std::array::from_fn(|i| c[i].round().clamp(0.0, 255.0) as u8)
}

/// Fit `k` centers to `points`. `log` receives progress lines.
pub fn fit<const D: usize>(points: &[[u8; D]], k: usize, iters: usize, seed: u64, log: &dyn Fn(String)) -> Vec<[u8; D]> {
    assert!(points.len() >= k, "{} points for {k} centers", points.len());
    let mut rng = SmallRng::seed_from_u64(seed);

    // k-means++ seeding on a subsample.
    let sub_n = points.len().min((40 * k).max(20_000));
    let sub: Vec<[u8; D]> = (0..sub_n).map(|_| points[rng.gen_range(0..points.len())]).collect();
    let mut centers: Vec<[u8; D]> = vec![sub[rng.gen_range(0..sub_n)]];
    let mut dist: Vec<u32> = sub.par_iter().map(|p| l1(p, &centers[0])).collect();
    while centers.len() < k {
        let total: u64 = dist.iter().map(|&d| d as u64).sum();
        let next = if total == 0 {
            sub[rng.gen_range(0..sub_n)]
        } else {
            let mut r = rng.gen_range(0..total);
            let mut pick = sub_n - 1;
            for (i, &d) in dist.iter().enumerate() {
                if r < d as u64 {
                    pick = i;
                    break;
                }
                r -= d as u64;
            }
            sub[pick]
        };
        centers.push(next);
        dist.par_iter_mut().zip(&sub).for_each(|(d, p)| *d = (*d).min(l1(p, &next)));
    }

    // Lloyd iterations on all points.
    let mut prev: Vec<u32> = vec![u32::MAX; points.len()];
    for it in 0..iters {
        let assign: Vec<(u32, u32)> = points.par_iter().map(|p| nearest(&centers, p)).collect();
        let changed = assign.iter().zip(&prev).filter(|(a, p)| a.0 != **p).count();
        let mean_d = assign.iter().map(|a| a.1 as f64).sum::<f64>() / points.len() as f64;
        let (sums, counts) = points
            .par_iter()
            .zip(&assign)
            .fold(
                || (vec![[0f64; D]; k], vec![0u64; k]),
                |(mut s, mut c), (p, a)| {
                    let j = a.0 as usize;
                    for i in 0..D {
                        s[j][i] += p[i] as f64;
                    }
                    c[j] += 1;
                    (s, c)
                },
            )
            .reduce(
                || (vec![[0f64; D]; k], vec![0u64; k]),
                |(mut s1, mut c1), (s2, c2)| {
                    for j in 0..k {
                        for i in 0..D {
                            s1[j][i] += s2[j][i];
                        }
                        c1[j] += c2[j];
                    }
                    (s1, c1)
                },
            );
        let mut empty = 0;
        for j in 0..k {
            if counts[j] == 0 {
                centers[j] = points[rng.gen_range(0..points.len())];
                empty += 1;
            } else {
                let mean: [f64; D] = std::array::from_fn(|i| sums[j][i] / counts[j] as f64);
                centers[j] = to_u8(&mean);
            }
        }
        log(format!("  k-means iter {it}: mean L1 {mean_d:.2}, {changed} reassigned, {empty} empty re-seeded"));
        prev = assign.iter().map(|a| a.0).collect();
        if it > 0 && changed * 1000 < points.len() && empty == 0 {
            break;
        }
    }
    centers
}
