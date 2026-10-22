//! llandau: smooth scientific slides drawn live (`slides`), and a viewer for atomic
//! trajectories that draws into decks, the explorer window or offscreen.

pub mod bonds;
pub mod camera;
pub mod clash;
pub mod occlusion;
pub mod plots;
pub mod protein;
pub mod recording;
pub mod render;
pub mod scene;
pub mod slides;
pub mod surface;
pub mod theme;
pub mod window;

use glam::Vec3;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use camera::Camera;
use render::{Gpu, Offscreen, Renderer};
use scene::{Ghost, Points, Scene, Sidechains, Structure, Style, SurfaceColors, VectorField};
use theme::{Rgba, Theme};

pub(crate) fn runtime_error(message: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(message.to_string())
}

pub(crate) fn value_error(message: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(message.to_string())
}

// --- numpy conversions ---------------------------------------------------------

/// Points from an array whose last axis has length 3 (any leading shape).
fn points(array: &PyReadonlyArrayDyn<f32>, name: &str) -> PyResult<Vec<Vec3>> {
    if array.shape().last() != Some(&3) {
        return Err(value_error(format!("{name}: expected shape (..., 3), got {:?}", array.shape())));
    }
    Ok(array.as_slice()?.chunks_exact(3).map(Vec3::from_slice).collect())
}

/// Colors from an array whose last axis has length 3 or 4, in [0, 1].
fn colors(array: &PyReadonlyArrayDyn<f32>, name: &str) -> PyResult<Vec<Rgba>> {
    let channels = *array.shape().last().unwrap_or(&0);
    if !(3..=4).contains(&channels) {
        return Err(value_error(format!("{name}: expected shape (..., 3 or 4), got {:?}", array.shape())));
    }
    Ok(array.as_slice()?.chunks_exact(channels).map(|c| [c[0], c[1], c[2], c.get(3).copied().unwrap_or(1.0)]).collect())
}

fn rgba(color: &[f32]) -> PyResult<Rgba> {
    match color {
        [r, g, b] => Ok([*r, *g, *b, 1.0]),
        [r, g, b, a] => Ok([*r, *g, *b, *a]),
        _ => Err(value_error(format!("expected (r, g, b) or (r, g, b, a), got {color:?}"))),
    }
}

/// A name among `options` (as a parsed value), else a ValueError naming them.
fn parsed<T>(value: Option<T>, what: &str, name: &str, options: &[&str]) -> PyResult<T> {
    value.ok_or_else(|| value_error(format!("unknown {what} {name:?}; {}", options.join(" or "))))
}

#[derive(Clone, Copy, Default)]
struct Stale {
    frame: bool,
    trails: bool,
    ghost: bool,
}

const ALL: Stale = Stale { frame: true, trails: true, ghost: true };
/// the ghost copies the structure in its look (sizes, colors, side chains)
const LOOK: Stale = Stale { frame: true, trails: false, ghost: true };
const FRAME: Stale = Stale { frame: true, trails: false, ghost: false };

/// A trajectory viewer: draws into a texture (the explorer, a deck's live view) or
/// offscreen with `render_image`. The explorer and live views use its fields and
/// methods directly; Python sees the `pymethods`.
#[pyclass(unsendable, subclass)]
pub struct Viewer {
    pub(crate) gpu: Gpu,
    renderer: Renderer,
    pub(crate) camera: Camera,
    home: Camera,
    pub(crate) scene: Scene,
    pub(crate) frame: usize,
    /// what needs uploading: the frame's geometry, the trails, the ghost
    stale: Stale,
    offscreen: Option<Offscreen>,
    pub(crate) plots: plots::Plots,
    /// the plots shown large: under the explorer's view, over a live view
    pub(crate) show_plots: bool,
    /// the fog was set (by hand or a state): loading a structure leaves it
    fog_set: bool,
}

/// Structures within this radius (Å) get no fog by default: depth cues mean little on a
/// small molecule and just grey its far atoms.
const SMALL_EXTENT: f32 = 7.0;

impl Viewer {
    /// Draw into `view` (`size` pixels, `format`); with a viewport (left, top,
    /// width, height in pixels), only there.
    pub(crate) fn draw_into(
        &mut self,
        view: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        size: (u32, u32),
        viewport: Option<[f32; 4]>,
    ) {
        self.upload();
        self.renderer.draw(&self.gpu.device, &self.gpu.queue, view, format, size, viewport, &self.camera);
    }

    fn upload(&mut self) {
        let (device, queue) = (&self.gpu.device, &self.gpu.queue);
        if self.stale.frame {
            self.renderer.upload(device, queue, &self.scene.geometry(self.frame));
        }
        if self.stale.trails {
            self.renderer.upload_trails(device, queue, &self.scene.trails());
            self.renderer.upload_discrete_trails(device, queue, &self.scene.discrete_trails());
        }
        if self.stale.ghost {
            let (spheres, tubes) = self.scene.ghost_geometry();
            self.renderer.upload_ghost(device, queue, &spheres, &tubes);
        }
        self.renderer.trail_segments = self.scene.trail_segments(self.frame);
        self.renderer.background = self.scene.theme.background;
        self.renderer.outline = self.scene.theme.outline;
        self.stale = Stale::default();
    }

    /// Mark (some of) the geometry for upload.
    fn changed(&mut self, what: Stale) {
        let s = &mut self.stale;
        *s = Stale { frame: s.frame | what.frame, trails: s.trails | what.trails, ghost: s.ghost | what.ghost };
    }

    fn structure(&self) -> PyResult<&Structure> {
        self.scene.structure.as_ref().ok_or_else(|| runtime_error("no structure; call set_structure first"))
    }

    /// Check that per-atom data has `atoms` or `frames * atoms` entries.
    fn check_per_atom(&self, len: usize, name: &str) -> PyResult<()> {
        let s = self.structure()?;
        if len == s.num_atoms || len == s.num_atoms * s.num_frames() {
            return Ok(());
        }
        Err(value_error(format!(
            "{name}: {len} entries; expected atoms ({}) or frames * atoms ({})",
            s.num_atoms,
            s.num_atoms * s.num_frames()
        )))
    }

    /// Toggleable layers, as the explorer lists them: (name, visible).
    pub(crate) fn layers(&self) -> Vec<(String, bool)> {
        let s = &self.scene;
        let mut layers = vec![("atoms".into(), s.show_atoms), ("surface".into(), s.surface.visible)];
        if self.num_path_frames() > 1 {
            layers.push(("trails".into(), s.trails.visible));
        }
        if self.num_discrete_frames() > 1 {
            layers.push(("discrete trails".into(), s.discrete_trails.visible));
        }
        if let Some(ghost) = &s.ghost {
            layers.push(("ghost".into(), ghost.visible));
        }
        layers.push(("clashes".into(), s.show_clashes));
        layers.extend(s.vectors.iter().map(|(name, field)| (name.clone(), field.visible)));
        layers.extend(s.points.iter().map(|(name, points)| (name.clone(), points.visible)));
        layers
    }

    fn add_field(&mut self, name: String, vectors: Vec<Vec3>, origins: Option<Vec<Vec3>>, color: Option<Vec<f32>>, scale: f32, visible: bool) -> PyResult<()> {
        let color = color.as_deref().map(rgba).transpose()?;
        self.scene.vectors.insert(name, VectorField { vectors, origins, color, scale, visible });
        self.changed(FRAME);
        Ok(())
    }

    pub(crate) fn vector_scales(&self) -> Vec<(String, f32)> {
        self.scene.vectors.iter().map(|(name, field)| (name.clone(), field.scale)).collect()
    }

    pub(crate) fn set_radius_scale(&mut self, scale: f32) {
        self.scene.radius_scale = scale;
        self.changed(LOOK);
    }

    pub(crate) fn set_surface_opacity(&mut self, opacity: f32) {
        self.scene.surface.opacity = opacity;
        self.changed(FRAME);
    }

    pub(crate) fn set_surface_colors_by_atom(&mut self, by_atom: bool) {
        self.scene.surface.colors = if by_atom { SurfaceColors::Atoms } else { SurfaceColors::Uniform };
        self.changed(FRAME);
    }
}

#[pymethods]
impl Viewer {
    /// Arguments are for subclasses' __init__ (e.g. llandau.Viewer's theme).
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, pyo3::types::PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let gpu = Gpu::shared().map_err(runtime_error)?;
        let renderer = Renderer::new(&gpu.device);
        Ok(Self {
            gpu,
            renderer,
            camera: Camera::default(),
            home: Camera::default(),
            scene: Scene::default(),
            frame: 0,
            stale: ALL,
            offscreen: None,
            plots: plots::Plots::default(),
            show_plots: false,
            fog_set: false,
        })
    }

    // --- output ----------------------------------------------------------------

    /// The GPU in use: name, backend (Metal, Vulkan, Gl) and type (a CPU type means software rendering).
    #[getter]
    fn adapter(&self) -> String {
        let info = self.gpu.adapter.get_info();
        format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
    }

    /// Draw offscreen; returns an (height, width, 4) uint8 RGBA array.
    fn render_image<'py>(&mut self, py: Python<'py>, width: u32, height: u32) -> PyResult<Bound<'py, PyAny>> {
        self.upload();
        let pixels =
            render::render_image(&self.gpu, &mut self.renderer, &mut self.offscreen, (width, height), &self.camera);
        let array = PyArray1::from_vec(py, pixels).reshape([height.max(1) as usize, width.max(1) as usize, 4])?;
        Ok(array.into_any())
    }

    /// Open the explorer window and return when it closes, with the viewer's
    /// plots (see set_plots). buttons:
    /// (label, function of the camera dict returning text to copy); screenshot:
    /// write one frame of the window to this PNG file and close; recorder:
    /// function (width, height) -> video writer for the record button, which
    /// renders record_size pixels (default: the 3D view's) at record_fps;
    /// keyframes: function (samples) -> path saving the record button's
    /// keyframes instead (samples: list of (seconds, state dict)).
    #[pyo3(signature = (info = None, buttons = None, title = None, size = (1300.0, 800.0),
                        screenshot = None, recorder = None, record_size = None, record_fps = 60.0, keyframes = None))]
    #[allow(clippy::too_many_arguments)]
    fn explore(
        &mut self,
        py: Python<'_>,
        info: Option<String>,
        buttons: Option<Vec<(String, Py<PyAny>)>>,
        title: Option<String>,
        size: (f64, f64),
        screenshot: Option<String>,
        recorder: Option<Py<PyAny>>,
        record_size: Option<(u32, u32)>,
        record_fps: f64,
        keyframes: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        if record_fps <= 0.0 {
            return Err(value_error("record_fps: must be positive"));
        }
        let options = window::Options {
            title: title.unwrap_or_else(|| "llandau".into()),
            size,
            info,
            buttons: buttons.unwrap_or_default(),
            screenshot,
            recorder,
            record_size,
            record_fps,
            keyframes,
        };
        window::explore(self, py, options)
    }

    // --- structure ---------------------------------------------------------------

    /// Set the trajectory: positions (frames, atoms, 3), radii and vdw_radii
    /// (atoms,), colors (atoms, 3 or 4), bonds (edges, 2; drawn with dynamic_bonds
    /// off) and optionally protein roles (atoms,): 0 hydrogen, 1 side chain, 2
    /// backbone N C O, 3 CA. Clears the layers and frames the camera on all frames.
    #[pyo3(signature = (positions, radii, vdw_radii, colors, edges, roles = None))]
    fn set_structure(
        &mut self,
        positions: PyReadonlyArrayDyn<f32>,
        radii: PyReadonlyArrayDyn<f32>,
        vdw_radii: PyReadonlyArrayDyn<f32>,
        colors: PyReadonlyArrayDyn<f32>,
        edges: PyReadonlyArrayDyn<i64>,
        roles: Option<PyReadonlyArrayDyn<i64>>,
    ) -> PyResult<()> {
        let shape = positions.shape();
        if shape.len() != 3 || shape[2] != 3 {
            return Err(value_error(format!("positions: expected (frames, atoms, 3), got {shape:?}")));
        }
        let num_atoms = shape[1];
        let colors = self::colors(&colors, "colors")?;
        let (radii, vdw) = (radii.as_slice()?.to_vec(), vdw_radii.as_slice()?.to_vec());
        for (name, len) in [("radii", radii.len()), ("vdw_radii", vdw.len()), ("colors", colors.len())] {
            if len != num_atoms {
                return Err(value_error(format!("{name}: {len} entries for {num_atoms} atoms")));
            }
        }
        let edges = edges
            .as_slice()?
            .chunks_exact(2)
            .map(|e| match e {
                [i, j] if (0..num_atoms as i64).contains(i) && (0..num_atoms as i64).contains(j) => {
                    Ok([*i as u32, *j as u32])
                }
                _ => Err(value_error(format!("bond {e:?} out of range"))),
            })
            .collect::<PyResult<Vec<_>>>()?;
        let roles = match roles {
            Some(roles) => {
                let roles: Vec<u8> = roles.as_slice()?.iter().map(|&r| r.clamp(0, 3) as u8).collect();
                if roles.len() != num_atoms {
                    return Err(value_error(format!("roles: {} entries for {num_atoms} atoms", roles.len())));
                }
                Some(roles)
            }
            None => None,
        };
        let structure = Structure::new(points(&positions, "positions")?, num_atoms, radii, vdw, colors, edges, roles);
        let (center, radius) = structure.bounds();
        self.camera.fit(center, radius);
        self.home = self.camera;
        if !self.fog_set {
            self.renderer.fog = if radius < SMALL_EXTENT { 0.0 } else { render::DEFAULT_FOG };
        }
        self.frame = self.frame.min(structure.num_frames().saturating_sub(1));
        self.scene.structure = Some(structure);
        self.scene.vectors.clear();
        self.scene.points.clear();
        self.scene.colorings.clear();
        self.scene.coloring = None;
        self.scene.ghost = None;
        self.scene.scrub_discrete = false;
        self.scene.trails.colors = None;
        self.scene.trails.segment_ids = None;
        self.changed(ALL);
        Ok(())
    }

    /// Discrete states of the same atoms, (states, atoms, 3), to scrub through
    /// instead of the path frames with `scrub_discrete`. None removes them.
    /// `path_frames`: the path frame (fractional) each state corresponds to, which
    /// places the trails and the plots' cursor; without, the states are spread evenly
    /// over the path.
    #[pyo3(signature = (positions = None, path_frames = None))]
    fn set_discrete_frames(
        &mut self,
        positions: Option<PyReadonlyArrayDyn<f32>>,
        path_frames: Option<Vec<f32>>,
    ) -> PyResult<()> {
        let (num_atoms, num_frames) = {
            let s = self.structure()?;
            (s.num_atoms, s.num_frames())
        };
        let discrete = positions.map(|p| points(&p, "positions")).transpose()?;
        if let Some(d) = &discrete
            && (d.is_empty() || d.len() % num_atoms != 0)
        {
            return Err(value_error(format!("discrete frames: expected (states, {num_atoms}, 3)")));
        }
        if let Some(frames) = &path_frames {
            let states = discrete.as_ref().map_or(0, |d| d.len() / num_atoms);
            if frames.len() != states {
                return Err(value_error(format!("path_frames: {} values for {states} states", frames.len())));
            }
            let last = (num_frames.max(1) - 1) as f32;
            if frames.iter().any(|f| !f.is_finite() || *f < 0.0 || *f > last) {
                return Err(value_error(format!("path_frames: every value must be within 0 and {last}")));
            }
        }
        let structure = self.scene.structure.as_mut().unwrap();
        structure.discrete = discrete;
        structure.discrete_frames = path_frames;
        if structure.discrete.is_none() {
            self.scene.scrub_discrete = false;
        }
        self.frame = 0;
        self.changed(ALL); // the discrete trails too
        Ok(())
    }

    /// Trails through the discrete states (see set_discrete_frames), thicker than the
    /// path's: `radius` in angstrom, `color` (3 or 4 components; None: the theme's) and
    /// `opacity`. They are the "discrete trails" layer; `visible` false hides them.
    #[pyo3(signature = (radius = 0.06, color = None, opacity = 1.0, visible = true))]
    fn set_discrete_trails(&mut self, radius: f32, color: Option<Vec<f32>>, opacity: f32, visible: bool) -> PyResult<()> {
        let color = match color.as_deref() {
            None => None,
            Some([r, g, b]) => Some([*r, *g, *b, 1.0]),
            Some([r, g, b, a]) => Some([*r, *g, *b, *a]),
            Some(c) => return Err(value_error(format!("color: expected 3 or 4 components, got {}", c.len()))),
        };
        self.scene.discrete_trails = scene::DiscreteTrails { visible, color, opacity, radius };
        self.changed(ALL);
        Ok(())
    }

    #[getter]
    fn num_discrete_frames(&self) -> usize {
        self.scene.structure.as_ref().map_or(0, Structure::num_discrete_frames)
    }

    /// Whether `frame` walks the discrete states (see set_discrete_frames) instead of the
    /// path frames. Switching keeps the relative position along the path.
    #[getter]
    fn scrub_discrete(&self) -> bool {
        self.scene.scrub_discrete
    }

    #[setter]
    fn set_scrub_discrete(&mut self, on: bool) -> PyResult<()> {
        if on == self.scene.scrub_discrete {
            return Ok(());
        }
        if on && self.num_discrete_frames() == 0 {
            return Err(runtime_error("no discrete frames; call set_discrete_frames first"));
        }
        let old = self.num_frames().max(2) - 1;
        self.scene.scrub_discrete = on;
        let new = self.num_frames().max(2) - 1;
        self.frame = (self.frame as f32 * new as f32 / old as f32).round() as usize;
        self.changed(FRAME);
        Ok(())
    }

    /// Forget the structure, the plots and every layer and setting (the theme stays).
    fn reset(&mut self) {
        let theme = self.scene.theme;
        self.scene = Scene { theme, ..Scene::default() };
        self.frame = 0;
        self.plots = plots::Plots::default();
        self.show_plots = false;
        self.changed(ALL);
    }

    /// Scalars to plot along the trajectory, with a cursor at the current frame:
    /// curves (name, xs, values) on the scale of t, the parameter of each path frame
    /// (empty: the frame index).
    #[pyo3(signature = (curves, t = None))]
    fn set_plots(&mut self, curves: Vec<plots::Curve>, t: Option<Vec<f64>>) -> PyResult<()> {
        for (name, xs, values) in &curves {
            if xs.len() != values.len() {
                return Err(value_error(format!("plot {name:?}: {} x values for {} values", xs.len(), values.len())));
            }
        }
        self.plots = plots::Plots { curves, t: t.unwrap_or_default() };
        Ok(())
    }

    /// Whether the plots are shown large: under the explorer's 3D view (key G), over a
    /// deck's live view.
    #[getter]
    fn show_plots(&self) -> bool {
        self.show_plots
    }

    #[setter]
    fn set_show_plots(&mut self, on: bool) {
        self.show_plots = on;
    }

    /// Set the element colors (atoms, 3 or 4). With `theme` ("light" or "dark")
    /// they are that theme's, used whenever it is the theme; without, they are
    /// used for every theme.
    #[pyo3(signature = (colors, theme = None))]
    fn set_colors(&mut self, colors: PyReadonlyArrayDyn<f32>, theme: Option<&str>) -> PyResult<()> {
        let colors = self::colors(&colors, "colors")?;
        if colors.len() != self.structure()?.num_atoms {
            return Err(value_error("colors: one per atom"));
        }
        let current = self.scene.theme;
        let structure = self.scene.structure.as_mut().unwrap();
        match theme {
            None => {
                structure.theme_colors = [None, None];
                structure.colors = colors;
            }
            Some(name) => {
                let theme = parsed(Theme::named(name), "theme", name, Theme::NAMES)?;
                if theme == current {
                    structure.colors = colors.clone();
                }
                structure.theme_colors[(theme == theme::DARK) as usize] = Some(colors);
            }
        }
        self.changed(LOOK);
        Ok(())
    }

    /// Colors along the backbone tube, one per CA in residue order; None: the theme's.
    #[pyo3(signature = (colors = None))]
    fn set_backbone_colors(&mut self, colors: Option<PyReadonlyArrayDyn<f32>>) -> PyResult<()> {
        let colors = colors.map(|c| self::colors(&c, "colors")).transpose()?;
        let structure = self.scene.structure.as_mut().ok_or_else(|| runtime_error("no structure"))?;
        let num_alpha = structure.chains.iter().map(Vec::len).sum::<usize>();
        if let Some(c) = &colors
            && c.len() != num_alpha
        {
            return Err(value_error(format!("backbone colors: {} for {num_alpha} CA atoms", c.len())));
        }
        structure.backbone_colors = colors;
        self.changed(LOOK);
        Ok(())
    }

    /// Where the plots' cursor is: the path parameter of the current frame (of the discrete
    /// state's path frame with scrub_discrete).
    pub(crate) fn plot_x(&self) -> f64 {
        match &self.scene.structure {
            Some(s) => self.plots.x(self.scene.path_frame(s, self.frame)),
            None => self.frame as f64,
        }
    }

    /// Frames that `frame` walks: the path frames, or the discrete states with scrub_discrete.
    #[getter]
    fn num_frames(&self) -> usize {
        match &self.scene.structure {
            Some(s) if self.scene.scrub_discrete => s.num_discrete_frames(),
            Some(s) => s.num_frames(),
            None => 0,
        }
    }

    #[getter]
    fn num_path_frames(&self) -> usize {
        self.scene.structure.as_ref().map_or(0, Structure::num_frames)
    }

    /// Whether the structure has protein roles (needed for the protein style).
    #[getter]
    fn has_roles(&self) -> bool {
        self.scene.structure.as_ref().is_some_and(|s| s.roles.is_some())
    }

    #[getter]
    fn frame(&self) -> usize {
        self.frame
    }

    #[setter]
    fn set_frame(&mut self, frame: usize) {
        let frame = frame.min(self.num_frames().saturating_sub(1));
        if frame != self.frame {
            self.frame = frame;
            self.stale.frame = true;
        }
    }

    // --- appearance --------------------------------------------------------------

    /// "light" or "dark": the background and every non-element color.
    #[getter]
    fn theme(&self) -> &'static str {
        if self.scene.theme == theme::DARK { "dark" } else { "light" }
    }

    #[setter]
    fn set_theme(&mut self, name: &str) -> PyResult<()> {
        self.scene.theme = parsed(Theme::named(name), "theme", name, Theme::NAMES)?;
        if let Some(structure) = &mut self.scene.structure
            && let Some(colors) = &structure.theme_colors[(self.scene.theme == theme::DARK) as usize]
        {
            structure.colors = colors.clone();
        }
        self.changed(ALL);
        Ok(())
    }

    /// How colors from the tables are decoded before lighting: 1 keeps them as
    /// they are (light, soft), 2.2 decodes sRGB exactly (darker, saturated).
    #[getter]
    fn color_gamma(&self) -> f32 {
        self.renderer.color_gamma
    }

    #[setter]
    fn set_color_gamma(&mut self, gamma: f32) {
        self.renderer.color_gamma = gamma.max(0.1);
    }

    /// Depth cueing: how far the farthest atoms fade towards the background
    /// (0: none, 1: all the way). By default 0.45 for structures larger than 7 Å in
    /// radius and 0 for smaller ones; once set, loading a structure keeps it.
    #[getter]
    fn fog(&self) -> f32 {
        self.renderer.fog
    }

    #[setter]
    fn set_fog(&mut self, fog: f32) {
        self.renderer.fog = fog.clamp(0.0, 1.0);
        self.fog_set = true;
    }

    /// Ambient occlusion per atom: how much the most buried atoms darken
    /// (0: none, 1: to black), by how much their neighbors surround them.
    #[getter]
    fn occlusion(&self) -> f32 {
        self.scene.occlusion
    }

    #[setter]
    fn set_occlusion(&mut self, occlusion: f32) {
        self.scene.occlusion = occlusion.clamp(0.0, 1.0);
        self.changed(FRAME);
    }

    /// "lit" (lighting and highlights) or "flat" (plain colors with a dark
    /// outline, as cinemol draws them; atoms in front of others far behind get a halo).
    #[getter]
    fn shading(&self) -> &'static str {
        if self.renderer.flat { "flat" } else { "lit" }
    }

    #[setter]
    fn set_shading(&mut self, shading: &str) -> PyResult<()> {
        self.renderer.flat = parsed(["lit", "flat"].iter().position(|n| *n == shading), "shading", shading, &["lit", "flat"])? == 1;
        Ok(())
    }

    /// "atoms" (ball and stick) or "protein" (backbone tube and side chains; needs roles).
    #[getter]
    fn style(&self) -> &'static str {
        self.scene.style.name()
    }

    #[setter]
    fn set_style(&mut self, style: &str) -> PyResult<()> {
        self.scene.style = parsed(Style::parse(style), "style", style, Style::NAMES)?;
        self.changed(ALL);
        Ok(())
    }

    /// Side chains in the protein style: "sticks" or "spheres" (van der Waals).
    #[getter]
    fn sidechains(&self) -> &'static str {
        self.scene.sidechains.name()
    }

    #[setter]
    fn set_sidechains(&mut self, sidechains: &str) -> PyResult<()> {
        self.scene.sidechains = parsed(Sidechains::parse(sidechains), "side chains", sidechains, Sidechains::NAMES)?;
        self.changed(LOOK);
        Ok(())
    }

    /// Infer the bonds anew in every frame (and the ghost) from the atoms' distances,
    /// instead of keeping the bonds the structure was loaded with (on by default).
    #[getter]
    fn dynamic_bonds(&self) -> bool {
        self.scene.dynamic_bonds
    }

    #[setter]
    fn set_dynamic_bonds(&mut self, dynamic: bool) {
        self.scene.dynamic_bonds = dynamic;
        self.changed(LOOK);
    }

    /// Set any of the sizes (in Å): radius_scale (ball and stick atoms),
    /// bond_radius, stick_radius, tube_radius, arrow_radius, trail_radius.
    #[pyo3(signature = (**sizes))]
    fn set_sizes(&mut self, sizes: Option<Bound<'_, PyDict>>) -> PyResult<()> {
        for (key, value) in sizes.iter().flat_map(|d| d.iter()) {
            let value: f32 = value.extract()?;
            let s = &mut self.scene;
            match key.extract::<String>()?.as_str() {
                "radius_scale" => s.radius_scale = value,
                "bond_radius" => s.bond_radius = value,
                "stick_radius" => s.stick_radius = value,
                "tube_radius" => s.tube_radius = value,
                "arrow_radius" => s.arrow_radius = value,
                "trail_radius" => s.trails.radius = value,
                other => return Err(PyKeyError::new_err(other.to_string())),
            }
        }
        self.changed(ALL);
        Ok(())
    }

    /// Visibility of a layer: "atoms", "trails", "ghost", "clashes", "surface", or the name of
    /// a vector field or a point set.
    fn set_visible(&mut self, layer: &str, visible: bool) -> PyResult<()> {
        let s = &mut self.scene;
        match layer {
            "atoms" => s.show_atoms = visible,
            "trails" => s.trails.visible = visible,
            "discrete trails" => s.discrete_trails.visible = visible,
            "clashes" => s.show_clashes = visible,
            "surface" => s.surface.visible = visible,
            "ghost" => match &mut s.ghost {
                Some(ghost) => ghost.visible = visible,
                None => return Err(PyKeyError::new_err("no ghost")),
            },
            name => match (s.vectors.get_mut(name), s.points.get_mut(name)) {
                (Some(field), _) => field.visible = visible,
                (None, Some(points)) => points.visible = visible,
                (None, None) => return Err(PyKeyError::new_err(name.to_string())),
            },
        }
        self.changed(ALL);
        Ok(())
    }

    // --- layers ------------------------------------------------------------------

    /// Trails of every atom (CA atoms in the protein style) through all frames.
    /// Per segment between consecutive frames, either `colors` (frames - 1, 3|4),
    /// or `segment_ids` (frames - 1,) whose parity picks one of two theme colors.
    #[pyo3(signature = (colors = None, segment_ids = None, opacity = 1.0))]
    fn set_trails(
        &mut self,
        colors: Option<PyReadonlyArrayDyn<f32>>,
        segment_ids: Option<PyReadonlyArrayDyn<i64>>,
        opacity: f32,
    ) -> PyResult<()> {
        let segments = self.num_frames().saturating_sub(1);
        let colors = colors.map(|c| self::colors(&c, "colors")).transpose()?;
        let segment_ids: Option<Vec<u32>> =
            segment_ids.map(|s| s.as_slice().map(|s| s.iter().map(|&i| i.max(0) as u32).collect())).transpose()?;
        let lengths = [colors.as_ref().map(Vec::len), segment_ids.as_ref().map(Vec::len)];
        if lengths.iter().flatten().any(|&len| len != segments) {
            return Err(value_error(format!("trails: expected one entry per segment ({segments})")));
        }
        let trails = &mut self.scene.trails;
        trails.colors = colors;
        trails.segment_ids = segment_ids;
        trails.opacity = opacity;
        self.changed(ALL);
        Ok(())
    }

    /// Frames of trail drawn on each side of the current frame; None: the whole trails.
    #[getter]
    fn trail_window(&self) -> Option<usize> {
        self.scene.trails.window
    }

    #[setter]
    fn set_trail_window(&mut self, window: Option<usize>) {
        self.scene.trails.window = window;
        self.changed(FRAME);
    }

    /// Arrows from each atom: vectors (frames, atoms, 3) or (atoms, 3), times `scale`.
    #[pyo3(signature = (name, vectors, color = None, scale = 1.0, visible = true))]
    fn add_vectors(
        &mut self,
        name: String,
        vectors: PyReadonlyArrayDyn<f32>,
        color: Option<Vec<f32>>,
        scale: f32,
        visible: bool,
    ) -> PyResult<()> {
        let vectors = points(&vectors, &name)?;
        self.check_per_atom(vectors.len(), &name)?;
        self.add_field(name, vectors, None, color, scale, visible)
    }

    /// Arrows that do not move with the frames: from `origins` (arrows, 3) along
    /// `vectors` (arrows, 3), times `scale`.
    #[pyo3(signature = (name, origins, vectors, color = None, scale = 1.0, visible = true))]
    fn add_arrows(
        &mut self,
        name: String,
        origins: PyReadonlyArrayDyn<f32>,
        vectors: PyReadonlyArrayDyn<f32>,
        color: Option<Vec<f32>>,
        scale: f32,
        visible: bool,
    ) -> PyResult<()> {
        let (origins, vectors) = (points(&origins, "origins")?, points(&vectors, &name)?);
        if origins.len() != vectors.len() {
            return Err(value_error(format!("{name}: {} origins for {} vectors", origins.len(), vectors.len())));
        }
        self.add_field(name, vectors, Some(origins), color, scale, visible)
    }

    /// Fixed spheres at `positions` (points, 3), with `colors` (points, 3|4) or one `color`.
    #[pyo3(signature = (name, positions, colors = None, color = None, radius = 0.05, visible = true))]
    fn add_points(
        &mut self,
        name: String,
        positions: PyReadonlyArrayDyn<f32>,
        colors: Option<PyReadonlyArrayDyn<f32>>,
        color: Option<Vec<f32>>,
        radius: f32,
        visible: bool,
    ) -> PyResult<()> {
        let positions = points(&positions, &name)?;
        let colors = match (colors, color) {
            (Some(colors), _) => self::colors(&colors, &name)?,
            (None, color) => vec![color.as_deref().map(rgba).transpose()?.unwrap_or(self.scene.theme.vector); positions.len()],
        };
        if colors.len() != positions.len() {
            return Err(value_error(format!("{name}: {} colors for {} points", colors.len(), positions.len())));
        }
        self.scene.points.insert(name, Points { positions, colors, radius, visible });
        self.changed(FRAME);
        Ok(())
    }

    fn set_vector_scale(&mut self, name: &str, scale: f32) -> PyResult<()> {
        let field = self.scene.vectors.get_mut(name).ok_or_else(|| PyKeyError::new_err(name.to_string()))?;
        field.scale = scale;
        self.changed(FRAME);
        Ok(())
    }

    /// Per-atom colors (frames, atoms, 3|4) or (atoms, 3|4), shown with `coloring = name`.
    fn add_coloring(&mut self, name: String, colors: PyReadonlyArrayDyn<f32>) -> PyResult<()> {
        let colors = self::colors(&colors, &name)?;
        self.check_per_atom(colors.len(), &name)?;
        self.scene.colorings.insert(name, colors);
        self.changed(FRAME);
        Ok(())
    }

    #[getter]
    fn coloring_names(&self) -> Vec<String> {
        self.scene.colorings.keys().cloned().collect()
    }

    /// The coloring in use; None: element colors.
    #[getter]
    fn coloring(&self) -> Option<String> {
        self.scene.coloring.clone()
    }

    #[setter]
    fn set_coloring(&mut self, name: Option<String>) -> PyResult<()> {
        if let Some(name) = &name
            && !self.scene.colorings.contains_key(name)
        {
            return Err(PyKeyError::new_err(name.clone()));
        }
        self.scene.coloring = name;
        self.changed(FRAME);
        Ok(())
    }

    /// A translucent copy of the structure at `positions` (atoms, 3), in the same
    /// style; colors move towards the background by `fade`. None removes it.
    #[pyo3(signature = (positions, opacity = 0.35, fade = 0.6))]
    fn set_ghost(&mut self, positions: Option<PyReadonlyArrayDyn<f32>>, opacity: f32, fade: f32) -> PyResult<()> {
        self.scene.ghost = match positions {
            Some(positions) => {
                let positions = points(&positions, "positions")?;
                if positions.len() != self.structure()?.num_atoms {
                    return Err(value_error("ghost: one position per atom of the structure"));
                }
                Some(Ghost { positions, visible: true, opacity, fade })
            }
            None => None,
        };
        self.changed(LOOK);
        Ok(())
    }

    /// The molecular surface (show it with set_visible("surface", True)): a smooth
    /// envelope of the van der Waals spheres of the heavy atoms (protein style) or
    /// all atoms. opacity < 1 draws only its front layer, over the rest; colors:
    /// "uniform" (the theme's) or "atoms" (the nearest atom's); spacing: grid (Å);
    /// sharpness: higher hugs the spheres, lower is smoother; scale multiplies the radii.
    #[pyo3(signature = (opacity = None, colors = None, spacing = None, sharpness = None, scale = None))]
    fn set_surface(
        &mut self,
        opacity: Option<f32>,
        colors: Option<&str>,
        spacing: Option<f32>,
        sharpness: Option<f32>,
        scale: Option<f32>,
    ) -> PyResult<()> {
        let surface = &mut self.scene.surface;
        if let Some(name) = colors {
            surface.colors = parsed(SurfaceColors::parse(name), "surface colors", name, SurfaceColors::NAMES)?;
        }
        surface.opacity = opacity.unwrap_or(surface.opacity);
        let settings = &mut surface.settings;
        settings.spacing = spacing.unwrap_or(settings.spacing);
        settings.sharpness = sharpness.unwrap_or(settings.sharpness);
        settings.scale = scale.unwrap_or(settings.scale);
        self.changed(FRAME);
        Ok(())
    }

    /// Van der Waals overlap (Å) above which two atoms clash.
    #[getter]
    fn clash_overlap(&self) -> f32 {
        self.scene.clash_overlap
    }

    #[setter]
    fn set_clash_overlap(&mut self, overlap: f32) {
        self.scene.clash_overlap = overlap;
        self.changed(FRAME);
    }

    /// Clashing atom pairs (i, j, overlap) in the current frame, among the atoms of the style.
    fn clashes(&self) -> PyResult<Vec<(usize, usize, f32)>> {
        let s = self.structure()?;
        Ok(self.scene.clashes(s, self.scene.frame_positions(s, self.frame).0))
    }

    // --- camera ------------------------------------------------------------------

    /// Rotate by a mouse drag of (dx, dy) pixels.
    fn orbit(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
    }

    /// Translate by a mouse drag of (dx, dy) pixels, in a view `height` pixels high.
    fn pan(&mut self, dx: f32, dy: f32, height: f32) {
        self.camera.pan(dx, dy, height);
    }

    /// Multiply the camera distance (< 1 moves closer).
    fn zoom(&mut self, factor: f32) {
        self.camera.zoom(factor);
    }

    fn reset_camera(&mut self) {
        self.camera = self.home;
    }

    /// The camera as a dict (target, distance, azimuth, elevation in degrees, fov in degrees).
    #[getter]
    fn camera<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        self.camera.to_dict(py)
    }

    /// Set (some of) the camera, with the keys the `camera` getter returns.
    #[setter]
    fn set_camera(&mut self, values: Bound<'_, PyDict>) -> PyResult<()> {
        self.camera.update(&values)
    }

    /// The whole view as a dict: everything the explorer's panel can change, the camera
    /// included (see llandau.recording.State::KEYS). Set it with a full or partial dict:
    /// entries left out keep their value. The explorer copies it, records it, a deck
    /// applies and plays it.
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        recording::State::capture(self).to_dict(py)
    }

    #[setter]
    fn set_state(&mut self, values: Bound<'_, PyDict>) -> PyResult<()> {
        let current = recording::State::capture(self);
        recording::State::from_dict(&values, &current)?.apply(self, &current);
        Ok(())
    }
}

/// Bonds (edges, 2) between atoms closer than `tolerance` times the sum of their
/// covalent radii: positions (atoms, 3), radii (atoms,).
#[pyfunction]
#[pyo3(signature = (positions, radii, tolerance = bonds::TOLERANCE))]
fn infer_bonds<'py>(
    py: Python<'py>,
    positions: PyReadonlyArrayDyn<f32>,
    radii: PyReadonlyArrayDyn<f32>,
    tolerance: f32,
) -> PyResult<Bound<'py, PyArray2<i64>>> {
    let positions = points(&positions, "positions")?;
    let radii = radii.as_slice()?;
    if radii.len() != positions.len() {
        return Err(value_error(format!("radii: {} entries for {} atoms", radii.len(), positions.len())));
    }
    let edges: Vec<i64> = bonds::infer(&positions, radii, tolerance).iter().flatten().map(|&a| a as i64).collect();
    let count = edges.len() / 2;
    Ok(PyArray1::from_vec(py, edges).reshape([count, 2])?)
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Viewer>()?;
    module.add_function(wrap_pyfunction!(infer_bonds, module)?)?;
    module.add_class::<slides::Slides>()?;
    module.add_function(wrap_pyfunction!(slides::parse_svg, module)?)?;
    module.add_function(wrap_pyfunction!(slides::image_size, module)?)?;
    Ok(())
}
