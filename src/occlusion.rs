//! Ambient occlusion per atom: how buried it is, as how much of the space
//! around it its neighbors' van der Waals spheres fill (Gaussian weighted).
//! The atoms' volumes are spread on a coarse grid and blurred (separable), so
//! the cost grows with the atoms and the grid, not with their neighbors, and
//! it is computed for every frame.

use glam::Vec3;

/// Width of the Gaussian neighborhood (angstrom).
const SIGMA: f32 = 2.7;
/// Grid spacing (angstrom); larger for structures that would need too many cells.
const SPACING: f32 = 2.0;
const MAX_CELLS: f32 = 4e6;
/// Filled amounts of an atom out in the open (at most; the frame's most
/// exposed atoms set it lower) and of a buried one.
const OPEN: f32 = 0.35;
const BURIED: f32 = 1.1;

/// How buried each of `atoms` is among them, 0 (as exposed as the most exposed atoms) to 1.
pub fn buried(positions: &[Vec3], vdw: &[f32], atoms: &[usize]) -> Vec<f32> {
    let filled = filled(positions, vdw, atoms);
    if filled.is_empty() {
        return filled;
    }
    // the 5th percentile: what counts as exposed in this frame
    let mut sorted = filled.clone();
    sorted.sort_unstable_by(f32::total_cmp);
    let open = sorted[sorted.len() / 20].max(OPEN);
    filled.iter().map(|f| ((f - open) / (BURIED - OPEN)).clamp(0.0, 1.0)).collect()
}

/// Per atom, the others' volumes weighted by a normalized Gaussian of their distance.
fn filled(positions: &[Vec3], vdw: &[f32], atoms: &[usize]) -> Vec<f32> {
    if atoms.is_empty() {
        return vec![];
    }
    let (lo, hi) = atoms.iter().fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), &a| {
        (lo.min(positions[a]), hi.max(positions[a]))
    });
    let reach = 3.0 * SIGMA;
    let size = hi - lo + Vec3::splat(2.0 * reach);
    let h = SPACING.max((size.x * size.y * size.z / MAX_CELLS).cbrt());
    let origin = lo - Vec3::splat(reach);
    let dims = (size / h).ceil().as_uvec3() + glam::UVec3::splat(2);
    let (nx, ny, nz) = (dims.x as usize, dims.y as usize, dims.z as usize);
    let index = |x: usize, y: usize, z: usize| (z * ny + y) * nx + x;

    // 1D kernel, summing to 1/h: the blurred grid is a density (per cubic angstrom)
    let radius = (reach / h).ceil() as i64;
    // spreading and reading back trilinearly blur too, by a variance of h²/6 each
    let sigma2 = (SIGMA * SIGMA - h * h / 3.0).max(0.25 * SIGMA * SIGMA);
    let mut kernel: Vec<f32> = (-radius..=radius).map(|m| (-(m as f32 * h).powi(2) / (2.0 * sigma2)).exp()).collect();
    let total: f32 = kernel.iter().sum();
    kernel.iter_mut().for_each(|k| *k /= total * h);
    let k = |m: i64| kernel.get((m + radius) as usize).copied().unwrap_or(0.0);

    // each atom's 8 cells and trilinear weights
    let corners = |p: Vec3| {
        let g = (p - origin) / h;
        let base = g.floor();
        let f = g - base;
        let base = base.as_uvec3();
        (0..8).map(move |c| {
            let (dx, dy, dz) = (c & 1, (c >> 1) & 1, (c >> 2) & 1);
            let w = [1.0 - f.x, f.x][dx] * [1.0 - f.y, f.y][dy] * [1.0 - f.z, f.z][dz];
            ([base.x as usize + dx, base.y as usize + dy, base.z as usize + dz], w)
        })
    };
    let volume = |a: usize| 4.0 / 3.0 * std::f32::consts::PI * vdw[a].powi(3);

    let mut grid = vec![0.0f32; nx * ny * nz];
    for &a in atoms {
        for ([x, y, z], w) in corners(positions[a]) {
            grid[index(x, y, z)] += w * volume(a);
        }
    }
    // blur along x, y and z
    let mut out = vec![0.0f32; grid.len()];
    for (axis, n) in [(0, nx), (1, ny), (2, nz)] {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let at = [x, y, z][axis] as i64;
                    let mut sum = 0.0;
                    for m in (at - radius).max(0)..=(at + radius).min(n as i64 - 1) {
                        let mut p = [x, y, z];
                        p[axis] = m as usize;
                        sum += k(at - m) * grid[index(p[0], p[1], p[2])];
                    }
                    out[index(x, y, z)] = sum;
                }
            }
        }
        std::mem::swap(&mut grid, &mut out);
    }

    atoms
        .iter()
        .map(|&a| {
            let cells: Vec<_> = corners(positions[a]).collect();
            let around: f32 = cells.iter().map(|&([x, y, z], w)| w * grid[index(x, y, z)]).sum();
            // the atom's own volume, through the same spreading and blur
            let own: f32 = cells
                .iter()
                .flat_map(|&(c, w)| cells.iter().map(move |&(d, v)| (c, d, w * v)))
                .map(|(c, d, w)| w * (0..3).map(|i| k(c[i] as i64 - d[i] as i64)).product::<f32>())
                .sum();
            around - own * volume(a)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_matches_the_direct_sum() {
        // a loose blob of 300 atoms
        let positions: Vec<Vec3> = (0..300)
            .map(|i| {
                let t = i as f32;
                Vec3::new((t * 0.618).fract(), (t * 0.381).fract(), (t * 0.127).fract()) * 18.0
            })
            .collect();
        let vdw = vec![1.6; positions.len()];
        let atoms: Vec<usize> = (0..positions.len()).collect();
        let volume = 4.0 / 3.0 * std::f32::consts::PI * 1.6f32.powi(3);
        let norm = (2.0 * std::f32::consts::PI).powf(1.5) * SIGMA.powi(3);
        let grid = filled(&positions, &vdw, &atoms);
        for (i, &p) in positions.iter().enumerate() {
            let direct: f32 = positions
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, q)| volume * (-p.distance_squared(*q) / (2.0 * SIGMA * SIGMA)).exp() / norm)
                .sum();
            assert!((grid[i] - direct).abs() < 0.05 + 0.1 * direct, "atom {i}: grid {} direct {direct}", grid[i]);
        }
    }

    #[test]
    fn a_small_molecule_is_not_buried() {
        let positions = [Vec3::ZERO, Vec3::X * 1.5, Vec3::Y * 1.5];
        assert!(buried(&positions, &[1.7; 3], &[0, 1, 2]).iter().all(|&b| b == 0.0));
    }
}
