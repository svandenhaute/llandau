//! The explorer: a native window (winit) with the 3D view and a panel of
//! controls and plots (egui), drawn with the viewer's own GPU device.
//!
//! Every control has a key (see KEYS; ? shows them, the panel hints them):
//! for presenting, H hides the panel, G shows the plots large under the view
//! and F goes full screen. Scrolling scrubs, faster when scrolling fast and for
//! long trajectories; with shift, frame by frame.
//!
//! Recording keeps the view's state at every redraw; stopping replays it
//! offscreen at a fixed frame rate into a video (a few frames per redraw, so
//! the window stays responsive).

use std::cell::RefCell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{Color32, Key, PointerButton, RichText, Sense};
use numpy::{PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Fullscreen, Window, WindowId};

use crate::camera::Camera;
use crate::plots::{self, Plots};
use crate::recording::{Recording, State};
use crate::render::{Gpu, Offscreen};

const PANEL_WIDTH: f32 = 300.0;
const PLAY_INTERVAL: Duration = Duration::from_millis(33);
/// Wheel or trackpad points per frame when scrubbing slowly, for up to this many frames.
const SCROLL_PER_FRAME: f32 = 12.0;
const SHORT_TRAJECTORY: f32 = 200.0;
/// Scrolling faster than this (points per second) speeds up the scrubbing ...
const SCROLL_SPEED: f32 = 400.0;
/// ... up to this factor.
const MAX_ACCELERATION: f32 = 10.0;
/// Time per redraw spent rendering a video's frames.
const RENDER_BUDGET: Duration = Duration::from_millis(100);
const VIDEO_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// The large plots' share of the window height, and their text size (points).
const PLOTS_HEIGHT: f32 = 0.4;
const PLOTS_TEXT: f32 = 18.0;
/// Zoom per + or - key press.
const ZOOM_STEP: f32 = 1.25;
const COPY: &str = if cfg!(target_os = "macos") { "Cmd C" } else { "Ctrl C" };
/// The keys hinted in the panel (in monospace: egui's text font lacks the arrows),
/// for the test that egui can draw them.
#[cfg(test)]
pub(crate) const HINTS: [&str; 13] = ["← →", "Space", "D", "R", "T", "L", "S", "⇧S", "C", "1", "9", COPY, "?"];

/// The explorer's keys, as its help shows them ("" keys head groups).
pub(crate) const KEYS: &[(&str, &str)] = &[
    ("", "Trajectory"),
    ("Space  P", "play / pause"),
    ("← →  , .", "previous / next frame (Shift: 5%)"),
    ("Home  End", "first / last frame"),
    ("D", "discrete states"),
    ("", "View"),
    ("drag", "rotate (right or Shift: pan)"),
    ("wheel", "scrub frames (pinch or Ctrl: zoom)"),
    ("+  −", "zoom in / out"),
    ("0", "reset the camera (or double click)"),
    ("", "Look"),
    ("T", "light / dark"),
    ("L", "lit / flat shading"),
    ("S", "atoms / protein (Shift: side chains)"),
    ("C", "next coloring"),
    ("1 – 9", "layers, in the panel's order"),
    ("", "Window"),
    ("G", "large plots"),
    ("H", "hide the panel"),
    ("F", "full screen"),
    ("R", "record a video"),
    (COPY, "copy the camera"),
    ("?", "these keys"),
    ("Esc", "close the keys, full screen, the window"),
];

thread_local! {
    // winit allows one event loop per process; each explore() reuses it.
    static EVENT_LOOP: RefCell<Option<EventLoop<()>>> = const { RefCell::new(None) };
}

pub struct Options {
    pub title: String,
    pub size: (f64, f64),
    pub info: Option<String>,
    /// label -> function of the camera dict returning text to copy
    pub buttons: Vec<(String, Py<PyAny>)>,
    /// render one frame of the window to this PNG path and close
    pub screenshot: Option<String>,
    /// function (width, height) -> video writer (write(frame), close() -> path,
    /// abort()) for the record button; None: recording is unavailable
    pub recorder: Option<Py<PyAny>>,
    /// video size in pixels (default: the 3D view's) and frames per second
    pub record_size: Option<(u32, u32)>,
    pub record_fps: f64,
}

/// What the explorer needs from the viewer, implemented in lib.rs.
pub trait Explorable {
    fn gpu(&self) -> &Gpu;
    fn prepare(&mut self);
    fn draw_into(&mut self, view: &wgpu::TextureView, format: wgpu::TextureFormat, size: (u32, u32), viewport: [f32; 4]);
    fn controls(&mut self) -> &mut dyn Controls;
}

/// The settings the panel edits.
pub trait Controls {
    fn frame(&self) -> usize;
    fn set_frame(&mut self, frame: usize);
    fn num_frames(&self) -> usize;
    fn num_discrete_frames(&self) -> usize;
    fn scrub_discrete(&self) -> bool;
    fn set_scrub_discrete(&mut self, on: bool);
    fn theme(&self) -> &'static str;
    fn set_theme(&mut self, name: &str);
    fn has_roles(&self) -> bool;
    fn shading(&self) -> &'static str;
    fn set_shading(&mut self, shading: &str);
    fn style(&self) -> &'static str;
    fn set_style(&mut self, style: &str);
    fn sidechains(&self) -> &'static str;
    fn set_sidechains(&mut self, sidechains: &str);
    fn coloring_names(&self) -> Vec<String>;
    fn coloring(&self) -> Option<String>;
    fn set_coloring(&mut self, name: Option<String>);
    /// toggleable layers: (name, visible)
    fn layers(&self) -> Vec<(String, bool)>;
    fn set_visible(&mut self, layer: &str, visible: bool);
    fn color_gamma(&self) -> f32;
    fn set_color_gamma(&mut self, gamma: f32);
    fn fog(&self) -> f32;
    fn set_fog(&mut self, fog: f32);
    fn occlusion(&self) -> f32;
    fn set_occlusion(&mut self, occlusion: f32);
    fn radius_scale(&self) -> f32;
    fn set_radius_scale(&mut self, scale: f32);
    fn surface_opacity(&self) -> f32;
    fn set_surface_opacity(&mut self, opacity: f32);
    fn surface_colors_by_atom(&self) -> bool;
    fn set_surface_colors_by_atom(&mut self, by_atom: bool);
    fn vector_scales(&self) -> Vec<(String, f32)>;
    fn set_vector_scale(&mut self, name: &str, scale: f32);
    fn orbit(&mut self, dx: f32, dy: f32);
    fn pan(&mut self, dx: f32, dy: f32, height: f32);
    fn zoom(&mut self, factor: f32);
    fn reset_camera(&mut self);
    fn camera(&self) -> Camera;
    fn set_camera(&mut self, camera: Camera);
    fn plots(&self) -> &Plots;
    /// whether the plots are shown large
    fn show_plots(&self) -> bool;
    fn set_show_plots(&mut self, on: bool);
    fn camera_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>>;
}

struct Surface {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

struct App<'a, V: Explorable> {
    viewer: &'a mut V,
    py: Python<'a>,
    options: Options,
    egui: egui::Context,
    surface: Option<Surface>,
    /// the 3D view's rectangle in points, from the last layout
    viewport: egui::Rect,
    playing: bool,
    last_step: Instant,
    scroll: f32,
    last_scroll: Instant,
    initial_scales: Vec<(String, f32)>,
    status: String,
    frames_drawn: u32,
    close_requested: bool,
    panel_hidden: bool,
    help: bool,
    toggle_fullscreen: bool,
    /// Escape: leave full screen, else close (applied with the window at hand)
    escape: bool,
    record: Record,
    error: Option<PyErr>,
}

/// The record button: off, recording, or rendering the video.
enum Record {
    Off,
    On { start: Instant, size: (u32, u32), recording: Recording },
    Rendering { recording: Recording, size: (u32, u32), writer: Py<PyAny>, target: Box<Offscreen>, next: usize, total: usize },
}

/// Run `f` with the process's event loop (created on first use).
pub fn with_event_loop<T>(f: impl FnOnce(&mut EventLoop<()>) -> PyResult<T>) -> PyResult<T> {
    EVENT_LOOP.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let event_loop = EventLoop::new().map_err(|e| crate::runtime_error(format!("cannot open a window: {e}")))?;
            *slot = Some(event_loop);
        }
        f(slot.as_mut().unwrap())
    })
}

pub fn explore<V: Explorable>(viewer: &mut V, py: Python<'_>, options: Options) -> PyResult<()> {
    with_event_loop(|event_loop| {
        let initial_scales = viewer.controls().vector_scales();
        let mut app = App {
            viewer,
            py,
            options,
            egui: egui::Context::default(),
            surface: None,
            viewport: egui::Rect::NOTHING,
            playing: false,
            last_step: Instant::now(),
            scroll: 0.0,
            last_scroll: Instant::now(),
            initial_scales,
            status: String::new(),
            frames_drawn: 0,
            close_requested: false,
            panel_hidden: false,
            help: false,
            toggle_fullscreen: false,
            escape: false,
            record: Record::Off,
            error: None,
        };
        event_loop.run_app_on_demand(&mut app).map_err(|e| crate::runtime_error(format!("window: {e}")))?;
        app.surface = None; // close the window before returning
        app.finish_recording();
        match app.error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })
}

impl<V: Explorable> App<'_, V> {
    fn create_surface(&mut self, event_loop: &ActiveEventLoop) -> Result<Surface, String> {
        let attributes = Window::default_attributes()
            .with_title(self.options.title.clone())
            .with_inner_size(LogicalSize::new(self.options.size.0, self.options.size.1));
        let window = Arc::new(event_loop.create_window(attributes).map_err(|e| e.to_string())?);
        let gpu = self.viewer.gpu();
        let surface = gpu.instance.create_surface(window.clone()).map_err(|e| e.to_string())?;
        let capabilities = surface.get_capabilities(&gpu.adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or_else(|| capabilities.formats.first().copied())
            .ok_or("the window supports no format")?;
        let size = window.inner_size();
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: capabilities.alpha_modes[0],
            view_formats: vec![],
            color_space: wgpu::SurfaceColorSpace::Auto,
        };
        surface.configure(&gpu.device, &config);
        let egui_state = egui_winit::State::new(
            self.egui.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_wgpu::Renderer::new(
            &gpu.device,
            format,
            egui_wgpu::RendererOptions { msaa_samples: 1, depth_stencil_format: None, dithering: true, ..Default::default() },
        );
        Ok(Surface { window, surface, config, egui_state, egui_renderer })
    }

    fn redraw(&mut self) {
        if self.playing && self.last_step.elapsed() >= PLAY_INTERVAL {
            self.last_step = Instant::now();
            let controls = self.viewer.controls();
            let count = controls.num_frames().max(1);
            controls.set_frame((controls.frame() + 1) % count);
        }
        let Some(mut target) = self.surface.take() else { return };
        let raw_input = target.egui_state.take_egui_input(&target.window);
        let egui = self.egui.clone();
        let output = egui.run_ui(raw_input, |ui| self.ui(ui));
        self.record_step();
        let fullscreen = target.window.fullscreen().is_some();
        if std::mem::take(&mut self.toggle_fullscreen) {
            target.window.set_fullscreen(if fullscreen { None } else { Some(Fullscreen::Borderless(None)) });
        }
        if std::mem::take(&mut self.escape) {
            if fullscreen {
                target.window.set_fullscreen(None);
            } else {
                self.close_requested = true;
            }
        }
        target.egui_state.handle_platform_output(&target.window, output.platform_output);
        let paint_jobs = self.egui.tessellate(output.shapes, output.pixels_per_point);
        let ppp = output.pixels_per_point;
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [target.config.width, target.config.height],
            pixels_per_point: ppp,
        };
        let viewport = [
            self.viewport.min.x * ppp,
            self.viewport.min.y * ppp,
            self.viewport.width() * ppp,
            self.viewport.height() * ppp,
        ];

        // the screenshot frame goes to an offscreen copy of the window instead
        let screenshot = self.options.screenshot.clone().filter(|_| self.frames_drawn >= 2);
        let size = (target.config.width, target.config.height);
        let format = target.config.format;
        let (device, queue) = (self.viewer.gpu().device.clone(), self.viewer.gpu().queue.clone());
        for (id, deltas) in &output.textures_delta.set {
            for delta in deltas {
                target.egui_renderer.update_texture(&device, &queue, *id, delta);
            }
        }
        let mut offscreen = None;
        let surface_texture = if screenshot.is_some() {
            offscreen = Some(Offscreen::with_format(&device, size, format));
            None
        } else {
            match target.surface.get_current_texture() {
                wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
                wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                    target.surface.configure(&device, &target.config);
                    self.surface = Some(target);
                    return;
                }
                _ => {
                    self.surface = Some(target);
                    return;
                }
            }
        };
        let view = match (&surface_texture, &offscreen) {
            (Some(texture), _) => texture.texture.create_view(&Default::default()),
            (None, Some(offscreen)) => offscreen.view().clone(),
            _ => unreachable!(),
        };

        self.viewer.prepare();
        self.viewer.draw_into(&view, format, size, viewport);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("egui") });
        target.egui_renderer.update_buffers(&device, &queue, &mut encoder, &paint_jobs, &screen);
        {
            let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("egui"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            target.egui_renderer.render(&mut pass.forget_lifetime(), &paint_jobs, &screen);
        }
        queue.submit([encoder.finish()]);
        for id in &output.textures_delta.free {
            target.egui_renderer.free_texture(id);
        }
        match (surface_texture, screenshot, offscreen) {
            (Some(texture), _, _) => queue.present(texture),
            (None, Some(path), Some(offscreen)) => {
                let pixels = offscreen.read(self.viewer.gpu());
                if let Err(error) = save_png(&path, size, format, &pixels) {
                    self.error = Some(crate::runtime_error(error));
                }
                self.options.screenshot = None;
                self.surface = Some(target);
                self.close_requested = true;
                return;
            }
            _ => {}
        }
        self.frames_drawn += 1;
        let again = self.playing
            || !matches!(self.record, Record::Off)
            || self.options.screenshot.is_some()
            || output.viewport_output.values().any(|v| v.repaint_delay < Duration::from_millis(100));
        if again {
            target.window.request_redraw();
        }
        self.surface = Some(target);
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        let dark = self.viewer.controls().theme() == "dark";
        ui.ctx().set_visuals(if dark { egui::Visuals::dark() } else { egui::Visuals::light() });
        self.keys(ui);
        if !self.panel_hidden {
            egui::Panel::right("controls")
                .exact_size(PANEL_WIDTH)
                .resizable(false)
                .show_separator_line(false)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                        ui.set_max_width(PANEL_WIDTH - 24.0);
                        self.panel(ui)
                    });
                });
        }
        let x = self.plot_x();
        let controls = self.viewer.controls();
        if controls.show_plots() && !controls.plots().is_empty() {
            egui::Panel::bottom("plots")
                .exact_size((PLOTS_HEIGHT * ui.available_height()).round())
                .resizable(false)
                .show(ui, |ui| controls.plots().large(ui, x, PLOTS_TEXT));
        }
        // the 3D view shows through the central panel, which only takes input
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let rect = ui.max_rect();
            let response = ui.allocate_rect(rect, Sense::click_and_drag());
            self.viewport = rect;
            self.pointer(ui, &response);
        });
        if self.help {
            plots::help(ui, KEYS, 15.0);
        }
    }

    fn keys(&mut self, ui: &egui::Ui) {
        if ui.ctx().egui_wants_keyboard_input() {
            return;
        }
        let pressed = |key| ui.input(|i| i.key_pressed(key));
        let shift = ui.input(|i| i.modifiers.shift);
        let controls = self.viewer.controls();
        if pressed(Key::Space) || pressed(Key::P) {
            self.playing = !self.playing;
        }
        // shift: jump 5% of the trajectory (at least 10 frames)
        let count = controls.num_frames().max(1) as i64;
        let jump = if shift { (count / 20).max(10) } else { 1 };
        for (key, step) in [
            (Key::ArrowLeft, -jump),
            (Key::Comma, -jump),
            (Key::ArrowRight, jump),
            (Key::Period, jump),
            (Key::Home, -count),
            (Key::End, count),
        ] {
            if pressed(key) {
                step_frames(controls, step);
            }
        }
        if pressed(Key::D) {
            toggle_discrete(controls);
        }
        if pressed(Key::Num0) {
            controls.reset_camera();
        }
        if pressed(Key::Plus) || pressed(Key::Equals) {
            controls.zoom(1.0 / ZOOM_STEP);
        }
        if pressed(Key::Minus) {
            controls.zoom(ZOOM_STEP);
        }
        if pressed(Key::T) {
            let next = if controls.theme() == "dark" { "light" } else { "dark" };
            controls.set_theme(next);
        }
        if pressed(Key::L) {
            toggle_shading(controls);
        }
        if pressed(Key::S) {
            if shift { toggle_sidechains(controls) } else { toggle_style(controls) }
        }
        if pressed(Key::C) {
            next_coloring(controls);
        }
        for (index, key) in NUMBER_KEYS.into_iter().enumerate() {
            if pressed(key) {
                toggle_layer(controls, index);
            }
        }
        if pressed(Key::G) {
            let on = !controls.show_plots();
            controls.set_show_plots(on);
        }
        if pressed(Key::H) {
            self.panel_hidden = !self.panel_hidden;
        }
        if pressed(Key::F) {
            self.toggle_fullscreen = true;
        }
        if pressed(Key::Questionmark) || (pressed(Key::Slash) && shift) {
            self.help = !self.help;
        }
        if pressed(Key::Escape) {
            if self.help { self.help = false } else { self.escape = true }
        }
        if pressed(Key::R) {
            self.toggle_recording(ui.ctx().pixels_per_point());
        }
        if ui.input(|i| i.events.iter().any(|e| matches!(e, egui::Event::Copy))) {
            self.copy(ui.ctx(), None);
        }
    }

    fn pointer(&mut self, ui: &egui::Ui, response: &egui::Response) {
        let ppp = ui.ctx().pixels_per_point();
        let height = response.rect.height() * ppp;
        let delta = response.drag_delta() * ppp;
        let shift = ui.input(|i| i.modifiers.shift);
        let (scroll, zoom) = if response.hovered() {
            ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()))
        } else {
            (0.0, 1.0)
        };
        let num_frames = self.viewer.controls().num_frames().max(1);
        if scroll != 0.0 {
            self.scroll -= scroll / SCROLL_PER_FRAME * self.scroll_gain(scroll, shift, num_frames);
        }
        let steps = self.scroll.trunc();
        self.scroll -= steps;

        let controls = self.viewer.controls();
        if response.dragged_by(PointerButton::Primary) && !shift {
            controls.orbit(delta.x, delta.y);
        } else if response.dragged_by(PointerButton::Secondary) || (response.dragged_by(PointerButton::Primary) && shift) {
            controls.pan(delta.x, delta.y, height);
        }
        if response.double_clicked() {
            controls.reset_camera();
        }
        if zoom != 1.0 {
            controls.zoom(1.0 / zoom);
        }
        if steps != 0.0 {
            let frame = (controls.frame() as i64 + steps as i64).clamp(0, num_frames as i64 - 1);
            controls.set_frame(frame as usize);
        }
    }

    /// Frames per scrolled point, relative to SCROLL_PER_FRAME: more for long
    /// trajectories and for fast scrolling; shift scrolls frame by frame.
    fn scroll_gain(&mut self, scroll: f32, precise: bool, num_frames: usize) -> f32 {
        let elapsed = self.last_scroll.elapsed().as_secs_f32();
        self.last_scroll = Instant::now();
        scroll_gain(scroll, elapsed, precise, num_frames)
    }

    fn panel(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().slider_width = 100.0;
        if let Some(info) = &self.options.info {
            ui.label(info.clone());
            ui.separator();
        }
        let controls = self.viewer.controls();

        // frames
        let count = controls.num_frames().max(1);
        let mut frame = controls.frame();
        ui.horizontal(|ui| {
            ui.label("frame");
            if ui.add(egui::Slider::new(&mut frame, 0..=count - 1)).changed() {
                controls.set_frame(frame);
            }
            hint(ui, "← →");
        });
        ui.horizontal(|ui| {
            if ui.button(if self.playing { "Pause" } else { "Play" }).clicked() {
                self.playing = !self.playing;
            }
            hint(ui, "Space");
            if controls.num_discrete_frames() > 0 {
                let mut on = controls.scrub_discrete();
                if ui.checkbox(&mut on, "scrub discrete").changed() {
                    controls.set_scrub_discrete(on);
                }
                hint(ui, "D");
            }
        });
        self.record_controls(ui);
        let controls = self.viewer.controls();
        ui.separator();

        // appearance
        egui::Grid::new("appearance").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            choice(ui, "theme", "T", &["light", "dark"], controls.theme(), |name| controls.set_theme(name));
            ui.end_row();
            choice(ui, "shading", "L", &["lit", "flat"], controls.shading(), |s| controls.set_shading(s));
            ui.end_row();
            if controls.has_roles() {
                choice(ui, "style", "S", &["atoms", "protein"], controls.style(), |s| controls.set_style(s));
                ui.end_row();
                choice(ui, "side chains", "⇧S", &["sticks", "spheres"], controls.sidechains(), |s| {
                    controls.set_sidechains(s)
                });
                ui.end_row();
            }
            let colorings = controls.coloring_names();
            if !colorings.is_empty() {
                let names: Vec<&str> = std::iter::once("element").chain(colorings.iter().map(String::as_str)).collect();
                let current = controls.coloring().unwrap_or_else(|| "element".into());
                let mut picked = None;
                choice(ui, "color", "C", &names, &current, |name| picked = Some(name.to_string()));
                if let Some(name) = picked {
                    controls.set_coloring(Some(name).filter(|n| n != "element"));
                }
                ui.end_row();
            }
        });
        ui.separator();

        // layers
        for (index, (layer, visible)) in controls.layers().into_iter().enumerate() {
            let mut on = visible;
            ui.horizontal(|ui| {
                if ui.checkbox(&mut on, layer.as_str()).changed() {
                    controls.set_visible(&layer, on);
                }
                if index < NUMBER_KEYS.len() {
                    hint(ui, &(index + 1).to_string());
                }
            });
        }
        ui.separator();

        // sizes
        egui::Grid::new("sizes").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            let mut gamma = controls.color_gamma();
            ui.label("color depth");
            if ui.add(egui::Slider::new(&mut gamma, 1.0..=2.4)).changed() {
                controls.set_color_gamma(gamma);
            }
            ui.end_row();
            let mut fog = controls.fog();
            ui.label("fog");
            if ui.add(egui::Slider::new(&mut fog, 0.0..=1.0)).changed() {
                controls.set_fog(fog);
            }
            ui.end_row();
            let mut occlusion = controls.occlusion();
            ui.label("occlusion");
            if ui.add(egui::Slider::new(&mut occlusion, 0.0..=1.0)).changed() {
                controls.set_occlusion(occlusion);
            }
            ui.end_row();
            let mut radius = controls.radius_scale();
            ui.label("atom radius");
            if ui.add(egui::Slider::new(&mut radius, 0.1..=2.0)).changed() {
                controls.set_radius_scale(radius);
            }
            ui.end_row();
            let mut opacity = controls.surface_opacity();
            ui.label("surface opacity");
            if ui.add(egui::Slider::new(&mut opacity, 0.05..=1.0)).changed() {
                controls.set_surface_opacity(opacity);
            }
            ui.end_row();
            let by_atom = controls.surface_colors_by_atom();
            choice(ui, "surface color", "", &["uniform", "atoms"], if by_atom { "atoms" } else { "uniform" }, |c| {
                controls.set_surface_colors_by_atom(c == "atoms")
            });
            ui.end_row();
            for (name, scale) in controls.vector_scales() {
                let initial = self.initial_scales.iter().find(|(n, _)| *n == name).map_or(scale, |(_, s)| *s);
                let mut value = scale;
                ui.label(format!("{name} length"));
                if ui.add(egui::Slider::new(&mut value, 0.0..=(4.0 * initial).max(1e-3))).changed() {
                    controls.set_vector_scale(&name, value);
                }
                ui.end_row();
            }
        });
        ui.separator();

        // copy buttons
        let mut copy = None;
        ui.horizontal_wrapped(|ui| {
            if ui.button("copy camera").clicked() {
                copy = Some(None);
            }
            hint(ui, COPY);
            for (index, (label, _)) in self.options.buttons.iter().enumerate() {
                if ui.button(label).clicked() {
                    copy = Some(Some(index));
                }
            }
        });
        if let Some(which) = copy {
            self.copy(ui.ctx(), which);
        }
        if !self.status.is_empty() {
            ui.small(&self.status);
        }

        // plots, unless shown large
        let x = self.plot_x();
        let controls = self.viewer.controls();
        if !controls.show_plots() {
            controls.plots().small(ui, x);
        }
        ui.add_space(8.0);
        let large = if controls.plots().is_empty() { "" } else { "G large plots · " };
        ui.label(RichText::new(format!("? keys · {large}H hide panel · F full screen")).small().weak());
    }

    fn record_controls(&mut self, ui: &mut egui::Ui) {
        let mut toggle = false;
        let mut cancel = false;
        ui.horizontal(|ui| match &self.record {
            Record::Off => {
                let available = self.options.recorder.is_some();
                let button = ui.add_enabled(available, egui::Button::new("Record"));
                if available {
                    toggle = button.on_hover_text("the video is rendered when you stop").clicked();
                    hint(ui, "R");
                } else {
                    button.on_disabled_hover_text("recording needs ffmpeg: pip install 'llandau[video]'");
                }
            }
            Record::On { start, size, .. } => {
                toggle = ui.button("Stop").clicked();
                let seconds = start.elapsed().as_secs();
                let time = format!("● {}:{:02}", seconds / 60, seconds % 60);
                ui.label(RichText::new(time).monospace().color(Color32::from_rgb(220, 50, 50)));
                ui.small(format!("{}×{}", size.0, size.1));
            }
            Record::Rendering { next, total, .. } => {
                cancel = ui.button("Cancel").clicked();
                let progress = *next as f32 / *total as f32;
                ui.add(egui::ProgressBar::new(progress).text(format!("rendering {next}/{total}")));
            }
        });
        if toggle {
            self.toggle_recording(ui.ctx().pixels_per_point());
        }
        if cancel {
            self.stop_rendering("cancelled".into());
        }
    }

    /// Start recording, or stop it and start rendering the video.
    fn toggle_recording(&mut self, pixels_per_point: f32) {
        match std::mem::replace(&mut self.record, Record::Off) {
            Record::Off if self.options.recorder.is_some() => {
                let view = self.viewport.size() * pixels_per_point;
                let (width, height) = self.options.record_size.unwrap_or((view.x as u32, view.y as u32));
                // even sizes: what H.264 players take
                let size = (width.max(2) & !1, height.max(2) & !1);
                self.record = Record::On { start: Instant::now(), size, recording: Recording::default() };
                self.status.clear();
            }
            Record::Off => {}
            Record::On { recording, size, .. } => {
                let total = ((recording.duration() * self.options.record_fps).round() as usize).max(1);
                let recorder = self.options.recorder.as_ref().expect("recording without a recorder");
                match recorder.call1(self.py, size) {
                    Ok(writer) => {
                        let target = Box::new(Offscreen::with_format(&self.viewer.gpu().device, size, VIDEO_FORMAT));
                        self.record = Record::Rendering { recording, size, writer, target, next: 0, total };
                    }
                    Err(error) => self.status = format!("{error}"),
                }
            }
            rendering @ Record::Rendering { .. } => self.record = rendering,
        }
    }

    /// While recording, keep the view's state; while rendering, render the
    /// video's next frames (the window then draws the live view again).
    fn record_step(&mut self) {
        match &mut self.record {
            Record::Off => {}
            Record::On { start, recording, .. } => {
                recording.push(start.elapsed().as_secs_f64(), State::capture(self.viewer.controls()));
            }
            Record::Rendering { .. } => {
                if let Err(error) = self.render_frames() {
                    self.stop_rendering(format!("recording failed: {error}"));
                }
            }
        }
    }

    fn render_frames(&mut self) -> PyResult<()> {
        let Record::Rendering { recording, size, writer, target, next, total } = &mut self.record else {
            return Ok(());
        };
        let live = State::capture(self.viewer.controls());
        let mut current = live.clone();
        let started = Instant::now();
        let mut result = Ok(());
        while *next < *total && started.elapsed() < RENDER_BUDGET {
            let Some(state) = recording.at(*next as f64 / self.options.record_fps) else { break };
            state.apply(self.viewer.controls(), &current);
            current = state;
            self.viewer.prepare();
            let (width, height) = *size;
            self.viewer.draw_into(target.view(), VIDEO_FORMAT, *size, [0.0, 0.0, width as f32, height as f32]);
            let pixels = target.read(self.viewer.gpu());
            result = PyArray1::from_vec(self.py, pixels)
                .reshape([height as usize, width as usize, 4])
                .and_then(|frame| writer.call_method1(self.py, "write", (frame,)))
                .map(|_| ());
            if result.is_err() {
                break;
            }
            *next += 1;
        }
        live.apply(self.viewer.controls(), &current);
        result?;
        if *next >= *total {
            let path: String = writer.call_method0(self.py, "close")?.extract(self.py)?;
            self.record = Record::Off;
            self.status = format!("saved {path}");
        }
        Ok(())
    }

    /// Closing the window still renders the video being recorded.
    fn finish_recording(&mut self) {
        if matches!(self.record, Record::Off) {
            return;
        }
        if matches!(self.record, Record::On { .. }) {
            self.toggle_recording(1.0);
        }
        while matches!(self.record, Record::Rendering { .. }) {
            if let Err(error) = self.py.check_signals() {
                self.stop_rendering("cancelled".into());
                self.error = Some(error);
                return;
            }
            self.record_step();
        }
        println!("{}", self.status);
    }

    /// Stop rendering without a video.
    fn stop_rendering(&mut self, status: String) {
        if let Record::Rendering { writer, .. } = std::mem::replace(&mut self.record, Record::Off) {
            let _ = writer.call_method0(self.py, "abort");
        }
        self.status = status;
    }

    /// Copy the camera (or a button's text of it) to the clipboard, and print it.
    fn copy(&mut self, ctx: &egui::Context, which: Option<usize>) {
        match self.copy_text(which) {
            Ok(text) => {
                println!("{text}");
                ctx.copy_text(text);
                self.status = "copied".into();
            }
            Err(error) => self.status = format!("{error}"),
        }
    }

    fn copy_text(&mut self, which: Option<usize>) -> PyResult<String> {
        let camera = self.viewer.controls().camera_dict(self.py)?;
        match which {
            None => {
                let json = self.py.import("json")?;
                json.call_method1("dumps", (camera,))?.extract()
            }
            Some(index) => self.options.buttons[index].1.call1(self.py, (camera,))?.extract(self.py),
        }
    }

    /// Where the plots' cursor is: the path parameter of the current frame.
    fn plot_x(&mut self) -> f64 {
        let c = self.viewer.controls();
        c.plots().x(c.frame(), c.num_frames(), c.scrub_discrete())
    }
}

const NUMBER_KEYS: [Key; 9] =
    [Key::Num1, Key::Num2, Key::Num3, Key::Num4, Key::Num5, Key::Num6, Key::Num7, Key::Num8, Key::Num9];

// what the explorer's and the live views' keys do

pub(crate) fn step_frames(c: &mut dyn Controls, step: i64) {
    let count = c.num_frames().max(1) as i64;
    c.set_frame((c.frame() as i64 + step).clamp(0, count - 1) as usize);
}

pub(crate) fn toggle_discrete(c: &mut dyn Controls) {
    if c.num_discrete_frames() > 0 {
        let on = !c.scrub_discrete();
        c.set_scrub_discrete(on);
    }
}

pub(crate) fn toggle_shading(c: &mut dyn Controls) {
    let next = if c.shading() == "lit" { "flat" } else { "lit" };
    c.set_shading(next);
}

pub(crate) fn toggle_style(c: &mut dyn Controls) {
    if c.has_roles() {
        let next = if c.style() == "atoms" { "protein" } else { "atoms" };
        c.set_style(next);
    }
}

pub(crate) fn toggle_sidechains(c: &mut dyn Controls) {
    if c.has_roles() {
        let next = if c.sidechains() == "sticks" { "spheres" } else { "sticks" };
        c.set_sidechains(next);
    }
}

/// Element colors, then each named coloring in turn.
pub(crate) fn next_coloring(c: &mut dyn Controls) {
    let names = c.coloring_names();
    let next = match c.coloring() {
        None => names.first().cloned(),
        Some(current) => names.iter().position(|n| *n == current).and_then(|i| names.get(i + 1)).cloned(),
    };
    c.set_coloring(next);
}

/// The `index`th layer (as the panel lists them) on or off.
pub(crate) fn toggle_layer(c: &mut dyn Controls, index: usize) {
    if let Some((name, visible)) = c.layers().into_iter().nth(index) {
        c.set_visible(&name, !visible);
    }
}

/// Frames per scrolled point relative to SCROLL_PER_FRAME, for `scroll` points
/// `elapsed` seconds after the previous scroll event.
fn scroll_gain(scroll: f32, elapsed: f32, precise: bool, num_frames: usize) -> f32 {
    if precise {
        return 1.0;
    }
    let length = (num_frames as f32 / SHORT_TRAJECTORY).max(1.0).sqrt();
    let speed = scroll.abs() / elapsed.max(1.0 / 240.0);
    length * (speed / SCROLL_SPEED).clamp(1.0, MAX_ACCELERATION)
}

/// A key's hint beside its control: small and dim.
fn hint(ui: &mut egui::Ui, key: &str) {
    ui.label(RichText::new(key).monospace().weak());
}

/// A labeled drop-down (its key hinted after the label, if any).
fn choice(ui: &mut egui::Ui, label: &str, key: &str, options: &[&str], current: &str, mut apply: impl FnMut(&str)) {
    ui.horizontal(|ui| {
        ui.label(label);
        if !key.is_empty() {
            hint(ui, key);
        }
    });
    egui::ComboBox::from_id_salt(label).selected_text(current).show_ui(ui, |ui| {
        for option in options {
            if ui.selectable_label(*option == current, *option).clicked() && *option != current {
                apply(option);
            }
        }
    });
}

pub(crate) fn save_png(path: &str, size: (u32, u32), format: wgpu::TextureFormat, pixels: &[u8]) -> Result<(), String> {
    let mut rgba = pixels.to_vec();
    if matches!(format, wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb) {
        rgba.chunks_exact_mut(4).for_each(|p| p.swap(0, 2));
    }
    let file = std::fs::File::create(path).map_err(|e| e.to_string())?;
    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), size.0, size.1);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header().and_then(|mut w| w.write_image_data(&rgba)).map_err(|e| e.to_string())
}

impl<V: Explorable> ApplicationHandler for App<'_, V> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.surface.is_some() {
            return;
        }
        match self.create_surface(event_loop) {
            Ok(surface) => {
                surface.window.request_redraw();
                self.surface = Some(surface);
            }
            Err(error) => {
                self.error = Some(crate::runtime_error(format!("cannot open a window: {error}")));
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(target) = self.surface.as_mut() else { return };
        let response = target.egui_state.on_window_event(&target.window, &event);
        match event {
            WindowEvent::CloseRequested => self.close_requested = true,
            WindowEvent::Resized(size) => {
                target.config.width = size.width.max(1);
                target.config.height = size.height.max(1);
                target.surface.configure(&self.viewer.gpu().device, &target.config);
                target.window.request_redraw();
            }
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {
                if response.repaint {
                    target.window.request_redraw();
                }
            }
        }
        if self.close_requested {
            self.surface = None;
            event_loop.exit();
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // let Ctrl-C in the terminal stop the window
        if let Err(error) = self.py.check_signals() {
            self.error = Some(error);
            self.surface = None;
            event_loop.exit();
            return;
        }
        if self.playing
            && let Some(target) = &self.surface
        {
            target.window.request_redraw();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrolling_accelerates() {
        // slow scrolling on a short trajectory: SCROLL_PER_FRAME points per frame
        assert_eq!(scroll_gain(5.0, 0.1, false, 50), 1.0);
        // fast scrolling: up to MAX_ACCELERATION times more frames per point
        assert_eq!(scroll_gain(40.0, 0.004, false, 50), MAX_ACCELERATION);
        assert!((scroll_gain(40.0, 0.016, false, 50) - 6.25).abs() < 1e-4);
        assert!((scroll_gain(8.0, 0.01, false, 50) - 2.0).abs() < 1e-4);
        // long trajectories move faster, shift goes frame by frame
        assert!((scroll_gain(5.0, 0.1, false, 3200) - 4.0).abs() < 1e-4);
        assert_eq!(scroll_gain(40.0, 0.016, true, 3200), 1.0);
    }

    /// Run frames of a combo box made by `choice`, with simulated pointer events.
    #[test]
    fn choice_applies_the_clicked_option() {
        let ctx = egui::Context::default();
        let mut current = "uniform".to_string();
        let mut option_rects: Vec<(String, egui::Rect)> = Vec::new();
        let mut combo_rect = egui::Rect::NOTHING;
        let frame = |events: Vec<egui::Event>, current: &mut String, option_rects: &mut Vec<(String, egui::Rect)>, combo_rect: &mut egui::Rect| {
            let input = egui::RawInput {
                screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 400.0))),
                events,
                ..Default::default()
            };
            let mut output = ctx.run_ui(input, |ui| {
                egui::CentralPanel::default().show(ui, |ui| {
                    let selected = current.clone();
                    ui.label("surface color");
                    let response = egui::ComboBox::from_id_salt("surface color").selected_text(selected.as_str()).show_ui(ui, |ui| {
                        for option in ["uniform", "atoms"] {
                            let r = ui.selectable_label(option == selected, option);
                            option_rects.push((option.to_string(), r.rect));
                            if r.clicked() && option != selected {
                                *current = option.to_string();
                            }
                        }
                    });
                    *combo_rect = response.response.rect;
                });
            });
            output.textures_delta.clear();
        };
        let click = |pos: egui::Pos2| {
            vec![
                egui::Event::PointerMoved(pos),
                egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed: true, modifiers: Default::default() },
                egui::Event::PointerButton { pos, button: PointerButton::Primary, pressed: false, modifiers: Default::default() },
            ]
        };
        frame(vec![], &mut current, &mut option_rects, &mut combo_rect);
        frame(click(combo_rect.center()), &mut current, &mut option_rects, &mut combo_rect);
        for _ in 0..3 {
            frame(vec![], &mut current, &mut option_rects, &mut combo_rect);
        }
        let atoms = option_rects.iter().rev().find(|(name, _)| name == "atoms").expect("the popup opened").1;
        frame(click(atoms.center()), &mut current, &mut option_rects, &mut combo_rect);
        frame(vec![], &mut current, &mut option_rects, &mut combo_rect);
        assert_eq!(current, "atoms");
    }
}
