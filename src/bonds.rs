//! Bonds from distances: atom pairs closer than a tolerance times the sum of their
//! covalent radii. Fast enough (a cell grid, linear in the atoms) to redo every frame.

use std::collections::HashMap;

use glam::{IVec3, Vec3};

/// The default tolerance on the sum of covalent radii.
pub const TOLERANCE: f32 = 1.15;

/// Bonds [i, j] with i < j, sorted.
pub fn infer(positions: &[Vec3], radii: &[f32], tolerance: f32) -> Vec<[u32; 2]> {
    let largest = radii.iter().copied().fold(0.0, f32::max);
    let cell = (2.0 * tolerance * largest).max(1e-3);
    let cell_of = |p: Vec3| (p / cell).floor().as_ivec3();
    let mut grid: HashMap<IVec3, Vec<usize>> = HashMap::new();
    for (a, p) in positions.iter().enumerate() {
        grid.entry(cell_of(*p)).or_default().push(a);
    }
    let mut bonds = Vec::new();
    for (i, p) in positions.iter().enumerate() {
        let home = cell_of(*p);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(others) = grid.get(&(home + IVec3::new(dx, dy, dz))) else {
                        continue;
                    };
                    for &j in others.iter().filter(|&&j| j > i) {
                        if p.distance(positions[j]) < tolerance * (radii[i] + radii[j]) {
                            bonds.push([i as u32, j as u32]);
                        }
                    }
                }
            }
        }
    }
    bonds.sort_unstable();
    bonds
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bonds_follow_the_distances() {
        let radii = [0.76, 0.76, 0.31];
        let close = [Vec3::ZERO, Vec3::X * 1.5, Vec3::X * 2.5];
        assert_eq!(infer(&close, &radii, TOLERANCE), vec![[0, 1], [1, 2]]);
        let apart = [Vec3::ZERO, Vec3::X * 1.5, Vec3::X * 4.0];
        assert_eq!(infer(&apart, &radii, TOLERANCE), vec![[0, 1]]);
    }
}
