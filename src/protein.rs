//! Protein topology from per-atom roles, and the smooth backbone curve.

use glam::Vec3;

/// Atom roles, as riemann's `atom_role`.
pub const HYDROGEN: u8 = 0;
pub const SIDECHAIN: u8 = 1;
pub const BACKBONE: u8 = 2; // N, C, O
pub const ALPHA: u8 = 3; // CA

/// Consecutive CA atoms farther apart than this start a new chain.
const CHAIN_BREAK: f32 = 4.3;

/// CA indices in residue order, split into chains where consecutive CAs are far apart.
pub fn chains(roles: &[u8], positions: &[Vec3]) -> Vec<Vec<usize>> {
    let mut chains: Vec<Vec<usize>> = Vec::new();
    for (atom, _) in roles.iter().enumerate().filter(|(_, r)| **r == ALPHA) {
        match chains.last_mut() {
            Some(chain) if positions[*chain.last().unwrap()].distance(positions[atom]) < CHAIN_BREAK => {
                chain.push(atom)
            }
            _ => chains.push(vec![atom]),
        }
    }
    chains
}

/// Uniform Catmull-Rom curve through `points`, `per_segment` samples per
/// segment. Returns (point, index of the segment start, fraction along it).
pub fn spline(points: &[Vec3], per_segment: usize) -> Vec<(Vec3, usize, f32)> {
    let n = points.len();
    if n < 2 {
        return points.iter().map(|p| (*p, 0, 0.0)).collect();
    }
    let at = |i: isize| points[i.clamp(0, n as isize - 1) as usize];
    let mut samples = Vec::with_capacity((n - 1) * per_segment + 1);
    for segment in 0..n - 1 {
        let i = segment as isize;
        let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
        for step in 0..per_segment {
            let t = step as f32 / per_segment as f32;
            let (t2, t3) = (t * t, t * t * t);
            let point = 0.5
                * (2.0 * p1
                    + (p2 - p0) * t
                    + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * t2
                    + (3.0 * p1 - p0 - 3.0 * p2 + p3) * t3);
            samples.push((point, segment, t));
        }
    }
    samples.push((points[n - 1], n - 2, 1.0));
    samples
}
