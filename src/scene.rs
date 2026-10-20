//! Scene data on the CPU: a trajectory with fixed topology plus optional
//! layers (trails, vector fields, colorings, a ghost), turned into GPU
//! instances for the current frame.

use std::collections::{BTreeMap, HashSet};

use bytemuck::{Pod, Zeroable};
use glam::Vec3;

use crate::clash;
use crate::occlusion;
use crate::surface::{self, MeshVertex};
use crate::protein::{self, ALPHA, HYDROGEN, SIDECHAIN};
use crate::theme::{self, Rgba, Theme};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SphereInstance {
    pub center: [f32; 3],
    /// negative for joints: spheres that fill the gaps between cylinders,
    /// which flat shading draws without an outline
    pub radius: f32,
    pub color: Rgba,
}

/// A cylinder or cone from `start` to `end`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct LatheInstance {
    pub start: [f32; 3],
    pub radius: f32,
    pub end: [f32; 3],
    pub _pad: f32,
    pub color_start: Rgba,
    pub color_end: Rgba,
}

impl LatheInstance {
    pub fn new(start: Vec3, end: Vec3, radius: f32, color_start: Rgba, color_end: Rgba) -> Self {
        Self { start: start.to_array(), radius, end: end.to_array(), _pad: 0.0, color_start, color_end }
    }
}

/// Instances of one frame, by how they are drawn.
#[derive(Default)]
pub struct Geometry {
    pub spheres: Vec<SphereInstance>,
    pub tubes: Vec<LatheInstance>,
    pub cones: Vec<LatheInstance>,
    pub ghost_spheres: Vec<SphereInstance>,
    pub ghost_tubes: Vec<LatheInstance>,
    /// drawn over everything (clash links)
    pub overlay: Vec<LatheInstance>,
    /// triangle list of the molecular surface, opaque or translucent (front layer only)
    pub surface: Vec<MeshVertex>,
    pub ghost_surface: Vec<MeshVertex>,
}

pub struct Structure {
    pub num_atoms: usize,
    /// frame-major: positions[frame * num_atoms + atom]
    pub positions: Vec<Vec3>,
    /// ball-and-stick radii (covalent)
    pub radii: Vec<f32>,
    /// van der Waals radii, for side chain spheres and clashes
    pub vdw: Vec<f32>,
    pub colors: Vec<Rgba>,
    pub edges: Vec<[u32; 2]>,
    /// protein roles (see `protein`), or None for a plain molecule
    pub roles: Option<Vec<u8>>,
    /// per CA, in residue order
    pub backbone_colors: Option<Vec<Rgba>>,
    pub chains: Vec<Vec<usize>>,
    pub excluded: HashSet<(u32, u32)>,
    /// discrete states of the same atoms (frame-major), an alternative to scrub through
    pub discrete: Option<Vec<Vec3>>,
    /// element colors per theme (light, dark), swapped in on a theme change
    pub theme_colors: [Option<Vec<Rgba>>; 2],
}

impl Structure {
    pub fn new(
        positions: Vec<Vec3>,
        num_atoms: usize,
        radii: Vec<f32>,
        vdw: Vec<f32>,
        colors: Vec<Rgba>,
        edges: Vec<[u32; 2]>,
        roles: Option<Vec<u8>>,
    ) -> Self {
        let chains = roles.as_deref().map_or_else(Vec::new, |r| protein::chains(r, &positions[..num_atoms]));
        let excluded = clash::exclusions(num_atoms, &edges);
        Self {
            num_atoms,
            positions,
            radii,
            vdw,
            colors,
            edges,
            roles,
            backbone_colors: None,
            chains,
            excluded,
            discrete: None,
            theme_colors: [None, None],
        }
    }

    pub fn num_frames(&self) -> usize {
        self.positions.len() / self.num_atoms.max(1)
    }

    pub fn frame(&self, frame: usize) -> &[Vec3] {
        let start = frame.min(self.num_frames().saturating_sub(1)) * self.num_atoms;
        &self.positions[start..start + self.num_atoms]
    }

    pub fn num_discrete_frames(&self) -> usize {
        self.discrete.as_ref().map_or(0, |d| d.len() / self.num_atoms.max(1))
    }

    fn role(&self, atom: usize) -> Option<u8> {
        self.roles.as_ref().map(|r| r[atom])
    }

    /// Center and radius of a sphere holding every atom of every frame.
    pub fn bounds(&self) -> (Vec3, f32) {
        let discrete = self.discrete.iter().flatten();
        let (lo, hi) = self.positions.iter().chain(discrete.clone()).fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), p| (lo.min(*p), hi.max(*p)),
        );
        if !lo.is_finite() {
            return (Vec3::ZERO, 1.0);
        }
        let center = (lo + hi) / 2.0;
        let reach = self.positions.iter().chain(discrete).map(|p| p.distance(center)).fold(0.0, f32::max);
        let largest = self.radii.iter().copied().fold(0.0, f32::max);
        (center, reach + largest)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Style {
    /// ball and stick, every atom
    Atoms,
    /// backbone tube and side chains, no hydrogens
    Protein,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sidechains {
    Sticks,
    /// van der Waals spheres: the space the side chains take up
    Spheres,
}

pub struct VectorField {
    /// frames * atoms, or atoms (the same every frame); or one per origin
    pub vectors: Vec<Vec3>,
    /// fixed arrow origins; None: from the atoms of the current frame
    pub origins: Option<Vec<Vec3>>,
    pub color: Option<Rgba>,
    pub scale: f32,
    pub visible: bool,
}

/// Fixed spheres, e.g. the frames of a discrete trajectory as dots.
pub struct Points {
    pub positions: Vec<Vec3>,
    pub colors: Vec<Rgba>,
    pub radius: f32,
    pub visible: bool,
}

pub struct Coloring {
    /// frames * atoms, or atoms (the same every frame)
    pub colors: Vec<Rgba>,
}

pub struct Trails {
    pub visible: bool,
    /// per segment (frames - 1): explicit colors, else alternating theme colors by segment id
    pub colors: Option<Vec<Rgba>>,
    pub segment_ids: Option<Vec<u32>>,
    pub opacity: f32,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceColors {
    /// the theme's surface color
    Uniform,
    /// the color of the nearest atom (element colors or the coloring in use)
    Atoms,
}

/// A Gaussian molecular surface around the heavy atoms (protein style) or all atoms.
pub struct Surface {
    pub visible: bool,
    pub opacity: f32,
    pub colors: SurfaceColors,
    pub settings: surface::Settings,
}

pub struct Ghost {
    pub positions: Vec<Vec3>,
    pub visible: bool,
    pub opacity: f32,
    /// how far colors move towards the background
    pub fade: f32,
}

pub struct Scene {
    pub structure: Option<Structure>,
    pub theme: Theme,
    pub style: Style,
    pub sidechains: Sidechains,
    pub show_atoms: bool,
    pub radius_scale: f32,
    pub bond_radius: f32,
    pub bond_color: Option<Rgba>,
    pub stick_radius: f32,
    pub tube_radius: f32,
    pub arrow_radius: f32,
    pub trails: Trails,
    pub vectors: BTreeMap<String, VectorField>,
    pub points: BTreeMap<String, Points>,
    pub colorings: BTreeMap<String, Coloring>,
    /// name of the coloring in use; None: element colors
    pub coloring: Option<String>,
    pub ghost: Option<Ghost>,
    pub surface: Surface,
    pub show_clashes: bool,
    pub clash_overlap: f32,
    /// the frame index walks the discrete states instead of the path frames
    pub scrub_discrete: bool,
    /// how much buried atoms darken (ambient occlusion per atom; 0: none)
    pub occlusion: f32,
}

/// Colors of the non-atom parts of a structure (faded for the ghost).
struct Palette {
    bond: Rgba,
    tube: Rgba,
    /// per CA, else `tube`
    backbone: Option<Vec<Rgba>>,
}

/// Curve samples per residue of the backbone tube.
const TUBE_SAMPLES: usize = 6;

impl Default for Scene {
    fn default() -> Self {
        Self {
            structure: None,
            theme: theme::LIGHT,
            style: Style::Atoms,
            sidechains: Sidechains::Sticks,
            show_atoms: true,
            radius_scale: 1.0,
            bond_radius: 0.12,
            bond_color: None,
            stick_radius: 0.16,
            tube_radius: 0.32,
            arrow_radius: 0.05,
            trails: Trails { visible: true, colors: None, segment_ids: None, opacity: 1.0, radius: 0.02 },
            vectors: BTreeMap::new(),
            points: BTreeMap::new(),
            colorings: BTreeMap::new(),
            coloring: None,
            ghost: None,
            surface: Surface {
                visible: false,
                opacity: 0.4,
                colors: SurfaceColors::Uniform,
                settings: surface::Settings { spacing: 0.5, sharpness: 2.0, scale: 1.0 },
            },
            show_clashes: false,
            clash_overlap: 0.4,
            scrub_discrete: false,
            occlusion: 0.5,
        }
    }
}

/// Index into per-frame (frames * atoms) or static (atoms) data.
fn per_frame<T: Copy>(data: &[T], num_atoms: usize, frame: usize) -> &[T] {
    if data.len() == num_atoms {
        data
    } else {
        let frames = data.len() / num_atoms.max(1);
        let start = frame.min(frames.saturating_sub(1)) * num_atoms;
        &data[start..start + num_atoms]
    }
}

impl Scene {
    /// Atoms that carry trails and arrows in the current style.
    fn marked_atoms(&self, s: &Structure, trails: bool) -> Vec<usize> {
        (0..s.num_atoms)
            .filter(|&a| match (self.style, s.role(a)) {
                (Style::Protein, Some(role)) if trails => role == ALPHA,
                (Style::Protein, Some(role)) => role != HYDROGEN,
                _ => true,
            })
            .collect()
    }

    fn atom_colors(&self, s: &Structure, frame: usize) -> Vec<Rgba> {
        match self.coloring.as_ref().and_then(|name| self.colorings.get(name)) {
            Some(coloring) => per_frame(&coloring.colors, s.num_atoms, frame).to_vec(),
            None => s.colors.clone(),
        }
    }

    /// Clashing pairs (i, j, overlap) at `positions`; in the protein style, among heavy atoms.
    pub fn clashes(&self, s: &Structure, positions: &[Vec3]) -> Vec<(usize, usize, f32)> {
        let atoms: Vec<usize> = match self.style {
            Style::Protein if s.roles.is_some() => (0..s.num_atoms).filter(|&a| s.role(a) != Some(HYDROGEN)).collect(),
            _ => (0..s.num_atoms).collect(),
        };
        clash::find(positions, &s.vdw, &atoms, &s.excluded, self.clash_overlap)
    }

    /// Positions of `frame` and the path frame it corresponds to: with
    /// `scrub_discrete`, the discrete states are spread evenly over the path.
    pub fn frame_positions<'a>(&self, s: &'a Structure, frame: usize) -> (&'a [Vec3], Option<usize>) {
        match (&s.discrete, self.scrub_discrete) {
            (Some(discrete), true) => {
                let count = s.num_discrete_frames();
                let frame = frame.min(count.saturating_sub(1));
                let path = (frame as f32 / (count.max(2) - 1) as f32 * (s.num_frames() - 1) as f32).round();
                (&discrete[frame * s.num_atoms..(frame + 1) * s.num_atoms], Some(path as usize))
            }
            _ => (s.frame(frame), None),
        }
    }

    pub fn geometry(&self, frame: usize) -> Geometry {
        let mut out = Geometry::default();
        let Some(s) = &self.structure else {
            return out;
        };
        let (positions, discrete) = self.frame_positions(s, frame);
        let on_path = discrete.is_none();
        let path_frame = discrete.unwrap_or(frame);
        let mut colors = self.atom_colors(s, path_frame);
        let mut backbone = s.backbone_colors.clone();
        if self.occlusion > 0.0 {
            let atoms: Vec<usize> = (0..s.num_atoms).collect();
            let shade: Vec<f32> =
                occlusion::buried(positions, &s.vdw, &atoms).iter().map(|b| 1.0 - self.occlusion * b).collect();
            let darken = |c: Rgba, f: f32| [c[0] * f, c[1] * f, c[2] * f, c[3]];
            for (c, &f) in colors.iter_mut().zip(&shade) {
                *c = darken(*c, f);
            }
            // the backbone tube, per residue by its CA
            if self.style == Style::Protein && !s.chains.is_empty() {
                let cas = s.chains.iter().flatten();
                backbone = Some(match &backbone {
                    Some(b) => cas.zip(b).map(|(&a, &c)| darken(c, shade[a])).collect(),
                    None => cas.map(|&a| darken(self.theme.tube, shade[a])).collect(),
                });
            }
        }

        let clashes = if self.show_clashes { self.clashes(s, positions) } else { vec![] };
        for &(i, j, _) in &clashes {
            colors[i] = self.theme.clash;
            colors[j] = self.theme.clash;
            let link = 0.06;
            out.overlay.push(LatheInstance::new(positions[i], positions[j], link, self.theme.clash, self.theme.clash));
        }

        if self.show_atoms {
            let bond = self.bond_color.unwrap_or(self.theme.bond);
            let palette = Palette { bond, tube: self.theme.tube, backbone };
            self.draw_structure(s, positions, &colors, &palette, &mut out.spheres, &mut out.tubes);
        }

        if let Some(ghost) = self.ghost.as_ref().filter(|g| g.visible && g.positions.len() == s.num_atoms) {
            let fade = |c: &Rgba| theme::fade(*c, self.theme.background, ghost.fade, ghost.opacity);
            let colors: Vec<Rgba> = s.colors.iter().map(fade).collect();
            let palette = Palette {
                bond: fade(&self.bond_color.unwrap_or(self.theme.bond)),
                tube: fade(&self.theme.tube),
                backbone: s.backbone_colors.as_ref().map(|b| b.iter().map(fade).collect()),
            };
            self.draw_structure(s, &ghost.positions, &colors, &palette, &mut out.ghost_spheres, &mut out.ghost_tubes);
        }

        let marked = self.marked_atoms(s, false);
        for field in self.vectors.values().filter(|f| f.visible) {
            let color = field.color.unwrap_or(self.theme.vector);
            match &field.origins {
                Some(origins) => {
                    for (origin, vector) in origins.iter().zip(&field.vectors) {
                        self.arrow(*origin, field.scale * *vector, color, &mut out);
                    }
                }
                // atom vectors belong to the path frames
                None if on_path => {
                    let vectors = per_frame(&field.vectors, s.num_atoms, frame);
                    for &a in &marked {
                        self.arrow(positions[a], field.scale * vectors[a], color, &mut out);
                    }
                }
                None => {}
            }
        }
        if self.surface.visible {
            let opacity = self.surface.opacity.clamp(0.0, 1.0);
            let atoms: Vec<usize> = match self.style {
                Style::Protein if s.roles.is_some() => (0..s.num_atoms).filter(|&a| s.role(a) != Some(HYDROGEN)).collect(),
                _ => (0..s.num_atoms).collect(),
            };
            let tint: Vec<Rgba> = match self.surface.colors {
                SurfaceColors::Uniform => vec![self.theme.surface; s.num_atoms],
                SurfaceColors::Atoms => colors.clone(),
            };
            let tint: Vec<Rgba> = tint.iter().map(|c| [c[0], c[1], c[2], opacity]).collect();
            let mesh = surface::gaussian_surface(positions, &s.vdw, &tint, &atoms, &self.surface.settings);
            if opacity < 1.0 { out.ghost_surface = mesh } else { out.surface = mesh }
        }
        for points in self.points.values().filter(|p| p.visible) {
            out.spheres.extend(points.positions.iter().zip(&points.colors).map(|(p, c)| SphereInstance {
                center: p.to_array(),
                radius: points.radius,
                color: *c,
            }));
        }
        out
    }

    fn arrow(&self, start: Vec3, vector: Vec3, color: Rgba, out: &mut Geometry) {
        let length = vector.length();
        if length < 1e-4 {
            return;
        }
        let r = self.arrow_radius;
        let head = (8.0 * r).min(0.45 * length);
        let base = start + vector * (1.0 - head / length);
        out.tubes.push(LatheInstance::new(start, base, r, color, color));
        out.cones.push(LatheInstance::new(base, start + vector, 2.6 * r, color, color));
    }

    /// Atoms and bonds, or the protein view, of one set of positions.
    fn draw_structure(
        &self,
        s: &Structure,
        positions: &[Vec3],
        colors: &[Rgba],
        palette: &Palette,
        spheres: &mut Vec<SphereInstance>,
        tubes: &mut Vec<LatheInstance>,
    ) {
        let sphere = |p: Vec3, radius: f32, color: Rgba| SphereInstance { center: p.to_array(), radius, color };
        let joint = |p: Vec3, radius: f32, color: Rgba| sphere(p, -radius, color);

        if self.style == Style::Atoms || s.roles.is_none() {
            let bond = palette.bond;
            spheres.extend((0..s.num_atoms).map(|a| sphere(positions[a], self.radius_scale * s.radii[a], colors[a])));
            tubes.extend(s.edges.iter().map(|&[i, j]| {
                LatheInstance::new(positions[i as usize], positions[j as usize], self.bond_radius, bond, bond)
            }));
            return;
        }

        let roles = s.roles.as_deref().unwrap();
        // backbone tube: cylinders between curve samples, a sphere at each joint
        let mut residue = 0;
        for chain in &s.chains {
            let points: Vec<Vec3> = chain.iter().map(|&a| positions[a]).collect();
            let color_at = |segment: usize, t: f32| -> Rgba {
                let Some(backbone) = &palette.backbone else {
                    return palette.tube;
                };
                let (a, b) = (backbone[residue + segment], backbone[(residue + segment + 1).min(backbone.len() - 1)]);
                theme::fade(a, b, t, a[3])
            };
            let samples = protein::spline(&points, TUBE_SAMPLES);
            for pair in samples.windows(2) {
                let ((p, si, ti), (q, sj, tj)) = (pair[0], pair[1]);
                tubes.push(LatheInstance::new(p, q, self.tube_radius, color_at(si, ti), color_at(sj, tj)));
            }
            spheres.extend(samples.iter().map(|&(p, si, ti)| joint(p, self.tube_radius, color_at(si, ti))));
            residue += chain.len();
        }

        // side chains: bonds with a side chain atom at either end
        let drawn = |i: usize, j: usize| {
            roles[i] != HYDROGEN && roles[j] != HYDROGEN && (roles[i] == SIDECHAIN || roles[j] == SIDECHAIN)
        };
        match self.sidechains {
            Sidechains::Sticks => {
                let mut joints = HashSet::new();
                for &[i, j] in &s.edges {
                    let (i, j) = (i as usize, j as usize);
                    if drawn(i, j) {
                        tubes.push(LatheInstance::new(positions[i], positions[j], self.stick_radius, colors[i], colors[j]));
                        joints.extend([i, j]);
                    }
                }
                let mut joints: Vec<usize> = joints.into_iter().filter(|&a| roles[a] != ALPHA).collect();
                joints.sort_unstable();
                spheres.extend(joints.iter().map(|&a| joint(positions[a], self.stick_radius, colors[a])));
            }
            Sidechains::Spheres => {
                spheres.extend(
                    (0..s.num_atoms)
                        .filter(|&a| roles[a] == SIDECHAIN)
                        .map(|a| sphere(positions[a], s.vdw[a], colors[a])),
                );
            }
        }
    }

    /// Trails of the marked atoms through all frames (they do not change with the frame).
    pub fn trails(&self) -> Vec<LatheInstance> {
        let Some(s) = self.structure.as_ref().filter(|s| self.trails.visible && s.num_frames() > 1) else {
            return vec![];
        };
        let t = &self.trails;
        let segment_color = |segment: usize| -> Rgba {
            let color = match (&t.colors, &t.segment_ids) {
                (Some(colors), _) => colors[segment.min(colors.len() - 1)],
                (None, Some(ids)) => self.theme.trail[ids[segment.min(ids.len() - 1)] as usize % 2],
                (None, None) => self.theme.trail[0],
            };
            [color[0], color[1], color[2], color[3] * t.opacity]
        };
        let frames = s.num_frames();
        let mut out = Vec::new();
        for a in self.marked_atoms(s, true) {
            for segment in 0..frames - 1 {
                let (p, q) = (s.positions[segment * s.num_atoms + a], s.positions[(segment + 1) * s.num_atoms + a]);
                let color = segment_color(segment);
                out.push(LatheInstance::new(p, q, t.radius, color, color));
            }
        }
        out
    }
}
