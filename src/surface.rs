//! Gaussian molecular surface: each atom adds a smooth density blob of its van
//! der Waals size, and the surface is where the summed density crosses 1 (for a
//! lone atom, exactly its radius; neighbors blend smoothly). The mesh comes from
//! surface nets: a vertex in each grid cell the surface crosses, joined into a
//! quad across each crossed grid edge.

use bytemuck::{Pod, Zeroable};
use glam::{IVec3, Vec3};

use crate::theme::Rgba;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: Rgba,
}

/// Contributions below this are cut off.
const CUTOFF: f32 = 1e-3;
/// Grids larger than this get a coarser spacing.
const MAX_POINTS: usize = 6_000_000;

pub struct Settings {
    /// grid spacing (Å)
    pub spacing: f32,
    /// larger: closer to the union of spheres; smaller: smoother, blobbier
    pub sharpness: f32,
    /// multiplies the radii
    pub scale: f32,
}

struct Grid {
    origin: Vec3,
    spacing: f32,
    size: IVec3,
    density: Vec<f32>,
    /// atom with the largest contribution at each point, for colors
    owner: Vec<u32>,
}

impl Grid {
    fn index(&self, p: IVec3) -> usize {
        (p.x + self.size.x * (p.y + self.size.y * p.z)) as usize
    }

    fn at(&self, p: IVec3) -> f32 {
        self.density[self.index(p.clamp(IVec3::ZERO, self.size - 1))]
    }

    fn gradient(&self, p: IVec3) -> Vec3 {
        let d = |axis: IVec3| self.at(p + axis) - self.at(p - axis);
        Vec3::new(d(IVec3::X), d(IVec3::Y), d(IVec3::Z))
    }
}

fn splat(positions: &[Vec3], radii: &[f32], atoms: &[usize], settings: &Settings) -> Option<Grid> {
    let reach = |r: f32| r * (1.0 + (1.0 / CUTOFF).ln() / settings.sharpness).sqrt();
    let largest = atoms.iter().map(|&a| reach(settings.scale * radii[a])).fold(0.0, f32::max);
    let (lo, hi) = atoms.iter().fold((Vec3::INFINITY, Vec3::NEG_INFINITY), |(lo, hi), &a| {
        (lo.min(positions[a]), hi.max(positions[a]))
    });
    if !lo.is_finite() {
        return None;
    }
    let mut spacing = settings.spacing.max(0.05);
    let extent = hi - lo + Vec3::splat(2.0 * largest + 4.0 * spacing);
    while (extent / spacing).ceil().element_product() as usize > MAX_POINTS {
        spacing *= 1.25;
    }
    let origin = lo - Vec3::splat(largest + 2.0 * spacing);
    let size = (extent / spacing).ceil().as_ivec3() + 1;
    let count = (size.x * size.y * size.z) as usize;
    let mut grid = Grid { origin, spacing, size, density: vec![0.0; count], owner: vec![u32::MAX; count] };
    let mut best = vec![0.0f32; count];

    for &a in atoms {
        let (center, r) = (positions[a], settings.scale * radii[a]);
        let cut = reach(r);
        let first = ((center - origin - cut) / spacing).floor().as_ivec3().max(IVec3::ZERO);
        let last = ((center - origin + cut) / spacing).ceil().as_ivec3().min(size - 1);
        for z in first.z..=last.z {
            for y in first.y..=last.y {
                for x in first.x..=last.x {
                    let p = IVec3::new(x, y, z);
                    let d2 = (origin + p.as_vec3() * spacing).distance_squared(center);
                    if d2 > cut * cut {
                        continue;
                    }
                    let value = (-settings.sharpness * (d2 / (r * r) - 1.0)).exp();
                    let i = grid.index(p);
                    grid.density[i] += value;
                    if value > best[i] {
                        best[i] = value;
                        grid.owner[i] = a as u32;
                    }
                }
            }
        }
    }
    Some(grid)
}

/// Triangles of the surface around `atoms`, colored like the atom that dominates each vertex.
pub fn gaussian_surface(
    positions: &[Vec3],
    radii: &[f32],
    colors: &[Rgba],
    atoms: &[usize],
    settings: &Settings,
) -> Vec<MeshVertex> {
    let Some(grid) = splat(positions, radii, atoms, settings) else {
        return vec![];
    };
    let inside = |p: IVec3| grid.density[grid.index(p)] > 1.0;
    let size = grid.size;
    let cells = size - 1;
    let cell_index = |c: IVec3| (c.x + cells.x * (c.y + cells.y * c.z)) as usize;
    const CORNERS: [IVec3; 8] = [
        IVec3::new(0, 0, 0),
        IVec3::new(1, 0, 0),
        IVec3::new(0, 1, 0),
        IVec3::new(1, 1, 0),
        IVec3::new(0, 0, 1),
        IVec3::new(1, 0, 1),
        IVec3::new(0, 1, 1),
        IVec3::new(1, 1, 1),
    ];
    const EDGES: [(usize, usize); 12] =
        [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];

    // one vertex per crossed cell: the mean of its edges' crossing points
    let mut vertex_of = vec![u32::MAX; (cells.x * cells.y * cells.z) as usize];
    let mut vertices: Vec<MeshVertex> = Vec::new();
    for z in 0..cells.z {
        for y in 0..cells.y {
            for x in 0..cells.x {
                let cell = IVec3::new(x, y, z);
                let values = CORNERS.map(|c| grid.density[grid.index(cell + c)] - 1.0);
                let positive = values.iter().filter(|v| **v > 0.0).count();
                if positive == 0 || positive == 8 {
                    continue;
                }
                let (mut sum, mut crossings) = (Vec3::ZERO, 0.0);
                for (a, b) in EDGES {
                    if (values[a] > 0.0) != (values[b] > 0.0) {
                        let t = values[a] / (values[a] - values[b]);
                        sum += CORNERS[a].as_vec3().lerp(CORNERS[b].as_vec3(), t);
                        crossings += 1.0;
                    }
                }
                let local = sum / crossings;
                // the density falls outwards: trilinear gradient of the cell's corners
                let gradient: Vec3 = CORNERS
                    .iter()
                    .map(|c| {
                        let w = Vec3::ONE - (c.as_vec3() - local).abs();
                        w.element_product() * grid.gradient(cell + *c)
                    })
                    .sum();
                let normal = -gradient.normalize_or(Vec3::Z);
                let nearest = cell + local.round().as_ivec3();
                let owner = grid.owner[grid.index(nearest)];
                let color = colors.get(owner as usize).copied().unwrap_or([0.7, 0.7, 0.7, 1.0]);
                vertex_of[cell_index(cell)] = vertices.len() as u32;
                vertices.push(MeshVertex {
                    position: (grid.origin + (cell.as_vec3() + local) * grid.spacing).to_array(),
                    normal: normal.to_array(),
                    color,
                });
            }
        }
    }

    // a quad across each crossed grid edge, joining the four cells around it
    let mut triangles = Vec::new();
    for z in 1..cells.z {
        for y in 1..cells.y {
            for x in 1..cells.x {
                let p = IVec3::new(x, y, z);
                for (axis, u, v) in [(IVec3::X, IVec3::Y, IVec3::Z), (IVec3::Y, IVec3::Z, IVec3::X), (IVec3::Z, IVec3::X, IVec3::Y)] {
                    if (p + axis).cmpge(size).any() || inside(p) == inside(p + axis) {
                        continue;
                    }
                    let quad = [p - u - v, p - v, p, p - u].map(|c| vertex_of[cell_index(c)]);
                    if quad.contains(&u32::MAX) {
                        continue;
                    }
                    for i in [quad[0], quad[1], quad[2], quad[0], quad[2], quad[3]] {
                        triangles.push(vertices[i as usize]);
                    }
                }
            }
        }
    }
    triangles
}
