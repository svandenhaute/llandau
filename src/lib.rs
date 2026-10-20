//! riemann_viewer: a small viewer for atomic trajectories that draws into a native
//! window handle (e.g. a Qt widget over the manim-slides player) or offscreen.

pub mod camera;
pub mod clash;
pub mod protein;
pub mod render;
pub mod scene;
pub mod surface;
pub mod theme;
pub mod window;

use std::ffi::c_void;
use std::ptr::NonNull;

use glam::Vec3;
use numpy::{PyArray1, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use camera::Camera;
use render::{Gpu, Offscreen, Renderer};
use scene::{Coloring, Ghost, Points, Scene, Sidechains, Structure, Style, SurfaceColors, VectorField};
use theme::{Rgba, Theme};

pub(crate) fn runtime_error(message: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(message.to_string())
}

fn value_error(message: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(message.to_string())
}

/// Display and window handles from a native window id, as Qt's `winId()` returns it.
fn raw_handles(handle: usize) -> PyResult<(RawDisplayHandle, RawWindowHandle)> {
    let pointer = NonNull::new(handle as *mut c_void).ok_or_else(|| value_error("null window handle"))?;
    #[cfg(target_os = "macos")]
    {
        use raw_window_handle::{AppKitDisplayHandle, AppKitWindowHandle};
        Ok((
            RawDisplayHandle::AppKit(AppKitDisplayHandle::new()),
            RawWindowHandle::AppKit(AppKitWindowHandle::new(pointer)),
        ))
    }
    #[cfg(target_os = "windows")]
    {
        use raw_window_handle::{Win32WindowHandle, WindowsDisplayHandle};
        let hwnd = std::num::NonZeroIsize::new(pointer.as_ptr() as isize).unwrap();
        Ok((
            RawDisplayHandle::Windows(WindowsDisplayHandle::new()),
            RawWindowHandle::Win32(Win32WindowHandle::new(hwnd)),
        ))
    }
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    {
        use raw_window_handle::{XlibDisplayHandle, XlibWindowHandle};
        Ok((
            RawDisplayHandle::Xlib(XlibDisplayHandle::new(None, 0)),
            RawWindowHandle::Xlib(XlibWindowHandle::new(pointer.as_ptr() as _)),
        ))
    }
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

struct Window {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
}

/// A trajectory viewer. Draw into a native window with `attach` + `render`,
/// or offscreen with `render_image`.
#[pyclass(unsendable, subclass)]
struct Viewer {
    gpu: Gpu,
    renderer: Renderer,
    camera: Camera,
    home: Camera,
    scene: Scene,
    frame: usize,
    /// whether the frame geometry and the trails need uploading
    stale: (bool, bool),
    window: Option<Window>,
    offscreen: Option<Offscreen>,
}

impl Viewer {
    fn upload(&mut self) {
        let (geometry, trails) = self.stale;
        if geometry {
            let geometry = self.scene.geometry(self.frame);
            self.renderer.upload(&self.gpu.device, &self.gpu.queue, &geometry);
        }
        if trails {
            let trails = self.scene.trails();
            self.renderer.upload_trails(&self.gpu.device, &self.gpu.queue, &trails);
        }
        self.renderer.background = self.scene.theme.background;
        self.stale = (false, false);
    }

    /// Mark everything for upload.
    fn changed(&mut self) {
        self.stale = (true, true);
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

    fn height(&self) -> f32 {
        self.window.as_ref().map_or(1000.0, |w| w.config.height as f32)
    }
}

#[pymethods]
impl Viewer {
    /// Arguments are for subclasses' __init__ (e.g. riemann_viewer.Viewer's theme).
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(_args: &Bound<'_, pyo3::types::PyTuple>, _kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let gpu = Gpu::new().map_err(runtime_error)?;
        let renderer = Renderer::new(&gpu.device);
        Ok(Self {
            gpu,
            renderer,
            camera: Camera::default(),
            home: Camera::default(),
            scene: Scene::default(),
            frame: 0,
            stale: (true, true),
            window: None,
            offscreen: None,
        })
    }

    // --- output ----------------------------------------------------------------

    /// The GPU in use: name, backend (Metal, Vulkan, Gl) and type (a CPU type means software rendering).
    #[getter]
    fn adapter(&self) -> String {
        let info = self.gpu.adapter.get_info();
        format!("{} ({:?}, {:?})", info.name, info.backend, info.device_type)
    }

    /// Draw into the native window `handle` (an NSView on macOS, HWND on Windows,
    /// X11 window otherwise) of `width` x `height` physical pixels.
    fn attach(&mut self, handle: usize, width: u32, height: u32) -> PyResult<()> {
        let (raw_display_handle, raw_window_handle) = raw_handles(handle)?;
        // Safety: the caller keeps the window alive while this viewer draws into it
        // (the Qt widget owns both and detaches before it is destroyed).
        let surface = unsafe {
            self.gpu.instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::RawHandle {
                raw_display_handle: Some(raw_display_handle),
                raw_window_handle,
            })
        }
        .map_err(runtime_error)?;
        let capabilities = surface.get_capabilities(&self.gpu.adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or_else(|| capabilities.formats.first().copied())
            .ok_or_else(|| runtime_error("the window surface supports no format"))?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&self.gpu.device, &config);
        self.window = Some(Window { surface, config });
        Ok(())
    }

    /// Stop drawing into the window (call before it is destroyed).
    fn detach(&mut self) {
        self.window = None;
    }

    fn resize(&mut self, width: u32, height: u32) {
        if let Some(window) = &mut self.window {
            window.config.width = width.max(1);
            window.config.height = height.max(1);
            window.surface.configure(&self.gpu.device, &window.config);
        }
    }

    /// Draw one frame into the attached window.
    fn render(&mut self) -> PyResult<()> {
        self.upload();
        let Some(window) = &self.window else {
            return Err(runtime_error("no window attached"));
        };
        let texture = match window.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                window.surface.configure(&self.gpu.device, &window.config);
                return Ok(());
            }
            _ => return Ok(()), // timeout or occluded: skip this frame
        };
        let view = texture.texture.create_view(&Default::default());
        let size = (window.config.width, window.config.height);
        self.renderer.draw(&self.gpu.device, &self.gpu.queue, &view, window.config.format, size, None, &self.camera);
        self.gpu.queue.present(texture);
        Ok(())
    }

    /// Draw offscreen; returns an (height, width, 4) uint8 RGBA array.
    fn render_image<'py>(&mut self, py: Python<'py>, width: u32, height: u32) -> PyResult<Bound<'py, PyAny>> {
        self.upload();
        let pixels =
            render::render_image(&self.gpu, &mut self.renderer, &mut self.offscreen, (width, height), &self.camera);
        let array = PyArray1::from_vec(py, pixels).reshape([height.max(1) as usize, width.max(1) as usize, 4])?;
        Ok(array.into_any())
    }

    /// Open the explorer window and return when it closes. plots: (name, xs,
    /// values) on the scale of t, the parameter of each path frame; buttons:
    /// (label, function of the camera dict returning text to copy); screenshot:
    /// write one frame of the window to this PNG file and close.
    #[pyo3(signature = (plots = None, t = None, info = None, buttons = None, title = None, size = (1300.0, 800.0), screenshot = None))]
    #[allow(clippy::too_many_arguments)]
    fn explore(
        &mut self,
        py: Python<'_>,
        plots: Option<Vec<window::Curve>>,
        t: Option<Vec<f64>>,
        info: Option<String>,
        buttons: Option<Vec<(String, Py<PyAny>)>>,
        title: Option<String>,
        size: (f64, f64),
        screenshot: Option<String>,
    ) -> PyResult<()> {
        let options = window::Options {
            title: title.unwrap_or_else(|| "riemann_viewer".into()),
            size,
            plots: plots.unwrap_or_default(),
            t: t.unwrap_or_default(),
            info,
            buttons: buttons.unwrap_or_default(),
            screenshot,
        };
        window::explore(self, py, options)
    }

    // --- structure ---------------------------------------------------------------

    /// Set the trajectory: positions (frames, atoms, 3), radii and vdw_radii
    /// (atoms,), colors (atoms, 3 or 4), bonds (edges, 2) and optionally protein
    /// roles (atoms,): 0 hydrogen, 1 side chain, 2 backbone N C O, 3 CA. Clears
    /// the layers and frames the camera on all frames.
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
        self.changed();
        Ok(())
    }

    /// Discrete states of the same atoms, (states, atoms, 3), to scrub through
    /// instead of the path frames with `scrub_discrete`. None removes them.
    #[pyo3(signature = (positions = None))]
    fn set_discrete_frames(&mut self, positions: Option<PyReadonlyArrayDyn<f32>>) -> PyResult<()> {
        let num_atoms = self.structure()?.num_atoms;
        let discrete = positions.map(|p| points(&p, "positions")).transpose()?;
        if let Some(d) = &discrete
            && (d.is_empty() || d.len() % num_atoms != 0)
        {
            return Err(value_error(format!("discrete frames: expected (states, {num_atoms}, 3)")));
        }
        let structure = self.scene.structure.as_mut().unwrap();
        structure.discrete = discrete;
        if structure.discrete.is_none() {
            self.scene.scrub_discrete = false;
        }
        self.frame = 0;
        self.changed();
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
        self.changed();
        Ok(())
    }

    /// Forget the structure and every layer and setting (the theme stays).
    fn reset(&mut self) {
        let theme = self.scene.theme;
        self.scene = Scene { theme, ..Scene::default() };
        self.frame = 0;
        self.changed();
    }

    /// Replace the element colors (atoms, 3 or 4).
    fn set_colors(&mut self, colors: PyReadonlyArrayDyn<f32>) -> PyResult<()> {
        let colors = self::colors(&colors, "colors")?;
        if colors.len() != self.structure()?.num_atoms {
            return Err(value_error("colors: one per atom"));
        }
        self.scene.structure.as_mut().unwrap().colors = colors;
        self.changed();
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
        self.changed();
        Ok(())
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

    #[getter]
    fn num_atoms(&self) -> usize {
        self.scene.structure.as_ref().map_or(0, |s| s.num_atoms)
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
            self.stale.0 = true;
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
        self.scene.theme = Theme::named(name).ok_or_else(|| value_error(format!("unknown theme {name:?}")))?;
        self.changed();
        Ok(())
    }

    /// "atoms" (ball and stick) or "protein" (backbone tube and side chains; needs roles).
    #[getter]
    fn style(&self) -> &'static str {
        match self.scene.style {
            Style::Atoms => "atoms",
            Style::Protein => "protein",
        }
    }

    #[setter]
    fn set_style(&mut self, style: &str) -> PyResult<()> {
        self.scene.style = match style {
            "atoms" => Style::Atoms,
            "protein" => Style::Protein,
            _ => return Err(value_error(format!("unknown style {style:?}; atoms or protein"))),
        };
        self.changed();
        Ok(())
    }

    /// Side chains in the protein style: "sticks" or "spheres" (van der Waals).
    #[getter]
    fn sidechains(&self) -> &'static str {
        match self.scene.sidechains {
            Sidechains::Sticks => "sticks",
            Sidechains::Spheres => "spheres",
        }
    }

    #[setter]
    fn set_sidechains(&mut self, sidechains: &str) -> PyResult<()> {
        self.scene.sidechains = match sidechains {
            "sticks" => Sidechains::Sticks,
            "spheres" => Sidechains::Spheres,
            _ => return Err(value_error(format!("unknown side chains {sidechains:?}; sticks or spheres"))),
        };
        self.changed();
        Ok(())
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
        self.changed();
        Ok(())
    }

    #[getter]
    fn sizes<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = &self.scene;
        let dict = PyDict::new(py);
        for (key, value) in [
            ("radius_scale", s.radius_scale),
            ("bond_radius", s.bond_radius),
            ("stick_radius", s.stick_radius),
            ("tube_radius", s.tube_radius),
            ("arrow_radius", s.arrow_radius),
            ("trail_radius", s.trails.radius),
        ] {
            dict.set_item(key, value)?;
        }
        Ok(dict)
    }

    /// Bond color in the ball and stick style; None: the theme's.
    #[setter]
    fn set_bond_color(&mut self, color: Option<Vec<f32>>) -> PyResult<()> {
        self.scene.bond_color = color.as_deref().map(rgba).transpose()?;
        self.changed();
        Ok(())
    }

    /// Visibility of a layer: "atoms", "trails", "ghost", "clashes", "surface", or the name of
    /// a vector field or a point set.
    fn set_visible(&mut self, layer: &str, visible: bool) -> PyResult<()> {
        let s = &mut self.scene;
        match layer {
            "atoms" => s.show_atoms = visible,
            "trails" => s.trails.visible = visible,
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
        self.changed();
        Ok(())
    }

    fn is_visible(&self, layer: &str) -> PyResult<bool> {
        let s = &self.scene;
        Ok(match layer {
            "atoms" => s.show_atoms,
            "trails" => s.trails.visible,
            "clashes" => s.show_clashes,
            "surface" => s.surface.visible,
            "ghost" => s.ghost.as_ref().is_some_and(|g| g.visible),
            name => match (s.vectors.get(name), s.points.get(name)) {
                (Some(field), _) => field.visible,
                (None, Some(points)) => points.visible,
                (None, None) => return Err(PyKeyError::new_err(name.to_string())),
            },
        })
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
        self.changed();
        Ok(())
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
        let color = color.as_deref().map(rgba).transpose()?;
        self.scene.vectors.insert(name, VectorField { vectors, origins: None, color, scale, visible });
        self.changed();
        Ok(())
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
        let color = color.as_deref().map(rgba).transpose()?;
        self.scene.vectors.insert(name, VectorField { vectors, origins: Some(origins), color, scale, visible });
        self.changed();
        Ok(())
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
        self.changed();
        Ok(())
    }

    #[getter]
    fn point_names(&self) -> Vec<String> {
        self.scene.points.keys().cloned().collect()
    }

    fn set_vector_scale(&mut self, name: &str, scale: f32) -> PyResult<()> {
        let field = self.scene.vectors.get_mut(name).ok_or_else(|| PyKeyError::new_err(name.to_string()))?;
        field.scale = scale;
        self.changed();
        Ok(())
    }

    fn vector_scale(&self, name: &str) -> PyResult<f32> {
        Ok(self.scene.vectors.get(name).ok_or_else(|| PyKeyError::new_err(name.to_string()))?.scale)
    }

    #[getter]
    fn vector_names(&self) -> Vec<String> {
        self.scene.vectors.keys().cloned().collect()
    }

    /// Per-atom colors (frames, atoms, 3|4) or (atoms, 3|4), shown with `coloring = name`.
    fn add_coloring(&mut self, name: String, colors: PyReadonlyArrayDyn<f32>) -> PyResult<()> {
        let colors = self::colors(&colors, &name)?;
        self.check_per_atom(colors.len(), &name)?;
        self.scene.colorings.insert(name, Coloring { colors });
        self.changed();
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
        self.changed();
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
        self.changed();
        Ok(())
    }

    #[setter]
    fn set_ghost_opacity(&mut self, opacity: f32) {
        if let Some(ghost) = &mut self.scene.ghost {
            ghost.opacity = opacity;
            self.changed();
        }
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
        if let Some(colors) = colors {
            surface.colors = match colors {
                "uniform" => SurfaceColors::Uniform,
                "atoms" => SurfaceColors::Atoms,
                _ => return Err(value_error(format!("unknown surface colors {colors:?}; uniform or atoms"))),
            };
        }
        surface.opacity = opacity.unwrap_or(surface.opacity);
        let settings = &mut surface.settings;
        settings.spacing = spacing.unwrap_or(settings.spacing);
        settings.sharpness = sharpness.unwrap_or(settings.sharpness);
        settings.scale = scale.unwrap_or(settings.scale);
        self.changed();
        Ok(())
    }

    #[getter]
    fn surface<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let surface = &self.scene.surface;
        let dict = PyDict::new(py);
        dict.set_item("visible", surface.visible)?;
        dict.set_item("opacity", surface.opacity)?;
        dict.set_item("colors", if surface.colors == SurfaceColors::Atoms { "atoms" } else { "uniform" })?;
        dict.set_item("spacing", surface.settings.spacing)?;
        dict.set_item("sharpness", surface.settings.sharpness)?;
        dict.set_item("scale", surface.settings.scale)?;
        Ok(dict)
    }

    /// Van der Waals overlap (Å) above which two atoms clash.
    #[getter]
    fn clash_overlap(&self) -> f32 {
        self.scene.clash_overlap
    }

    #[setter]
    fn set_clash_overlap(&mut self, overlap: f32) {
        self.scene.clash_overlap = overlap;
        self.changed();
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

    /// Translate by a mouse drag of (dx, dy) pixels, in a view `height` pixels high
    /// (default: the attached window's).
    #[pyo3(signature = (dx, dy, height = None))]
    fn pan(&mut self, dx: f32, dy: f32, height: Option<f32>) {
        let height = height.unwrap_or_else(|| self.height());
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
        let c = &self.camera;
        let dict = PyDict::new(py);
        dict.set_item("target", c.target.to_array())?;
        dict.set_item("distance", c.distance)?;
        dict.set_item("azimuth", c.azimuth.to_degrees())?;
        dict.set_item("elevation", c.elevation.to_degrees())?;
        dict.set_item("fov", c.fov_y.to_degrees())?;
        Ok(dict)
    }

    /// Set (some of) the camera, with the keys the `camera` getter returns.
    #[setter]
    fn set_camera(&mut self, values: Bound<'_, PyDict>) -> PyResult<()> {
        let c = &mut self.camera;
        if let Some(v) = values.get_item("target")? {
            c.target = Vec3::from_array(v.extract()?);
        }
        if let Some(v) = values.get_item("distance")? {
            c.distance = v.extract()?;
        }
        if let Some(v) = values.get_item("azimuth")? {
            c.azimuth = v.extract::<f32>()?.to_radians();
        }
        if let Some(v) = values.get_item("elevation")? {
            c.elevation = v.extract::<f32>()?.to_radians();
        }
        if let Some(v) = values.get_item("fov")? {
            c.fov_y = v.extract::<f32>()?.to_radians();
        }
        Ok(())
    }
}

#[pymodule]
fn _core(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Viewer>()?;
    Ok(())
}

// --- the explorer window -------------------------------------------------------

impl window::Explorable for Viewer {
    fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    fn prepare(&mut self) {
        self.upload();
    }

    fn draw_into(&mut self, view: &wgpu::TextureView, format: wgpu::TextureFormat, size: (u32, u32), viewport: [f32; 4]) {
        self.renderer.draw(&self.gpu.device, &self.gpu.queue, view, format, size, Some(viewport), &self.camera);
    }

    fn controls(&mut self) -> &mut dyn window::Controls {
        self
    }
}

impl window::Controls for Viewer {
    fn frame(&self) -> usize {
        self.frame
    }
    fn set_frame(&mut self, frame: usize) {
        Viewer::set_frame(self, frame)
    }
    fn num_frames(&self) -> usize {
        Viewer::num_frames(self)
    }
    fn num_discrete_frames(&self) -> usize {
        Viewer::num_discrete_frames(self)
    }
    fn scrub_discrete(&self) -> bool {
        self.scene.scrub_discrete
    }
    fn set_scrub_discrete(&mut self, on: bool) {
        let _ = Viewer::set_scrub_discrete(self, on);
    }
    fn theme(&self) -> &'static str {
        Viewer::theme(self)
    }
    fn set_theme(&mut self, name: &str) {
        let _ = Viewer::set_theme(self, name);
    }
    fn has_roles(&self) -> bool {
        Viewer::has_roles(self)
    }
    fn style(&self) -> &'static str {
        Viewer::style(self)
    }
    fn set_style(&mut self, style: &str) {
        let _ = Viewer::set_style(self, style);
    }
    fn sidechains(&self) -> &'static str {
        Viewer::sidechains(self)
    }
    fn set_sidechains(&mut self, sidechains: &str) {
        let _ = Viewer::set_sidechains(self, sidechains);
    }
    fn coloring_names(&self) -> Vec<String> {
        self.scene.colorings.keys().cloned().collect()
    }
    fn coloring(&self) -> Option<String> {
        self.scene.coloring.clone()
    }
    fn set_coloring(&mut self, name: Option<String>) {
        let _ = Viewer::set_coloring(self, name);
    }
    fn layers(&self) -> Vec<(String, bool)> {
        let s = &self.scene;
        let mut layers = vec![("atoms".into(), s.show_atoms), ("surface".into(), s.surface.visible)];
        if self.num_path_frames() > 1 {
            layers.push(("trails".into(), s.trails.visible));
        }
        if let Some(ghost) = &s.ghost {
            layers.push(("ghost".into(), ghost.visible));
        }
        layers.push(("clashes".into(), s.show_clashes));
        layers.extend(s.vectors.iter().map(|(name, field)| (name.clone(), field.visible)));
        layers.extend(s.points.iter().map(|(name, points)| (name.clone(), points.visible)));
        layers
    }
    fn set_visible(&mut self, layer: &str, visible: bool) {
        let _ = Viewer::set_visible(self, layer, visible);
    }
    fn radius_scale(&self) -> f32 {
        self.scene.radius_scale
    }
    fn set_radius_scale(&mut self, scale: f32) {
        self.scene.radius_scale = scale;
        self.changed();
    }
    fn surface_opacity(&self) -> f32 {
        self.scene.surface.opacity
    }
    fn set_surface_opacity(&mut self, opacity: f32) {
        self.scene.surface.opacity = opacity;
        self.changed();
    }
    fn surface_colors_by_atom(&self) -> bool {
        self.scene.surface.colors == SurfaceColors::Atoms
    }
    fn set_surface_colors_by_atom(&mut self, by_atom: bool) {
        self.scene.surface.colors = if by_atom { SurfaceColors::Atoms } else { SurfaceColors::Uniform };
        self.changed();
    }
    fn vector_scales(&self) -> Vec<(String, f32)> {
        self.scene.vectors.iter().map(|(name, field)| (name.clone(), field.scale)).collect()
    }
    fn set_vector_scale(&mut self, name: &str, scale: f32) {
        let _ = Viewer::set_vector_scale(self, name, scale);
    }
    fn orbit(&mut self, dx: f32, dy: f32) {
        self.camera.orbit(dx, dy);
    }
    fn pan(&mut self, dx: f32, dy: f32, height: f32) {
        self.camera.pan(dx, dy, height);
    }
    fn zoom(&mut self, factor: f32) {
        self.camera.zoom(factor);
    }
    fn reset_camera(&mut self) {
        self.camera = self.home;
    }
    fn camera_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        Viewer::camera(self, py)
    }
}
