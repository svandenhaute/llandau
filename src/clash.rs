//! Steric clashes: atom pairs whose van der Waals spheres overlap by more than
//! a threshold, ignoring pairs within a few bonds of each other.

use std::collections::{HashMap, HashSet};

use glam::Vec3;

/// Bonds between atoms that are never counted as clashing (1-2, 1-3 and 1-4 pairs).
const EXCLUDED_BONDS: usize = 3;

fn key(i: usize, j: usize) -> (u32, u32) {
    (i.min(j) as u32, i.max(j) as u32)
}

/// Pairs of atoms at most EXCLUDED_BONDS bonds apart.
pub fn exclusions(num_atoms: usize, edges: &[[u32; 2]]) -> HashSet<(u32, u32)> {
    let mut neighbors = vec![Vec::new(); num_atoms];
    for &[i, j] in edges {
        neighbors[i as usize].push(j as usize);
        neighbors[j as usize].push(i as usize);
    }
    let mut excluded = HashSet::new();
    for start in 0..num_atoms {
        let mut frontier = vec![start];
        let mut seen = HashSet::from([start]);
        for _ in 0..EXCLUDED_BONDS {
            frontier = frontier
                .iter()
                .flat_map(|&a| neighbors[a].iter().copied())
                .filter(|b| seen.insert(*b))
                .collect();
            excluded.extend(frontier.iter().map(|&b| key(start, b)));
        }
    }
    excluded
}

/// Clashing pairs (i, j, overlap) among `atoms`.
pub fn find(
    positions: &[Vec3],
    vdw: &[f32],
    atoms: &[usize],
    excluded: &HashSet<(u32, u32)>,
    min_overlap: f32,
) -> Vec<(usize, usize, f32)> {
    let largest = atoms.iter().map(|&a| vdw[a]).fold(0.0, f32::max);
    let cell = (2.0 * largest).max(1e-3);
    let cell_of = |p: Vec3| (p / cell).floor().as_ivec3();
    let mut grid: HashMap<_, Vec<usize>> = HashMap::new();
    for &a in atoms {
        grid.entry(cell_of(positions[a])).or_default().push(a);
    }
    let mut clashes = Vec::new();
    for &i in atoms {
        let home = cell_of(positions[i]);
        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    let Some(others) = grid.get(&(home + glam::IVec3::new(dx, dy, dz))) else {
                        continue;
                    };
                    for &j in others.iter().filter(|&&j| j > i) {
                        let overlap = vdw[i] + vdw[j] - positions[i].distance(positions[j]);
                        if overlap > min_overlap && !excluded.contains(&key(i, j)) {
                            clashes.push((i, j, overlap));
                        }
                    }
                }
            }
        }
    }
    clashes
}
