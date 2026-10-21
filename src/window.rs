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
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Fullscreen, Window, WindowId};

use crate::Viewer;
use crate::plots;
use crate::recording::{Recording, State};
use crate::render::{Gpu, Offscreen};

pub(crate) const PANEL_WIDTH: f32 = 300.0;
pub(crate) const PLAY_INTERVAL: Duration = Duration::from_millis(33);
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
    (COPY, "copy the view's state"),
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
    /// abort()) for the record button, which then renders a video
    pub recorder: Option<Py<PyAny>>,
    /// function (samples: list of (time, state dict)) -> path, for the record
    /// button, which then saves the keyframes; neither: recording is unavailable
    pub keyframes: Option<Py<PyAny>>,
    /// video size in pixels (default: the 3D view's) and frames per second
    pub record_size: Option<(u32, u32)>,
    pub record_fps: f64,
}

/// A window and the wgpu surface drawn into it (the explorer's and the deck player's).
pub(crate) struct Surface {
    pub window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
}

impl Surface {
    /// A window of `size` logical pixels; srgb picks an sRGB format when there is one
    /// (else a plain one: the deck writes sRGB values as they are).
    pub fn new(event_loop: &ActiveEventLoop, gpu: &Gpu, title: &str, size: (f64, f64), srgb: bool, fullscreen: bool) -> Result<Self, String> {
        let mut attributes = Window::default_attributes().with_title(title).with_inner_size(LogicalSize::new(size.0, size.1));
        if fullscreen {
            attributes = attributes.with_fullscreen(Some(Fullscreen::Borderless(None)));
        }
        let window = Arc::new(event_loop.create_window(attributes).map_err(|e| e.to_string())?);
        let surface = gpu.instance.create_surface(window.clone()).map_err(|e| e.to_string())?;
        let capabilities = surface.get_capabilities(&gpu.adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb() == srgb)
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
        Ok(Self { window, surface, config })
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }

    pub fn resize(&mut self, device: &wgpu::Device, size: PhysicalSize<u32>) {
        self.config.width = size.width.max(1);
        self.config.height = size.height.max(1);
        self.surface.configure(device, &self.config);
        self.window.request_redraw();
    }

    /// The frame to draw into; None when there is none right now (the surface
    /// was lost and is configured again, or the window is occluded).
    pub fn acquire(&self, device: &wgpu::Device) -> Option<wgpu::SurfaceTexture> {
        match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => Some(t),
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                self.surface.configure(device, &self.config);
                None
            }
            _ => None,
        }
    }
}

/// Paint one egui frame over `view` (`size` pixels); whether egui wants another frame soon.
pub(crate) fn paint_egui(
    renderer: &mut egui_wgpu::Renderer,
    gpu: &Gpu,
    ctx: &egui::Context,
    output: egui::FullOutput,
    view: &wgpu::TextureView,
    size: (u32, u32),
) -> bool {
    let (device, queue) = (&gpu.device, &gpu.queue);
    let jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
    let screen = egui_wgpu::ScreenDescriptor { size_in_pixels: [size.0, size.1], pixels_per_point: output.pixels_per_point };
    for (id, deltas) in &output.textures_delta.set {
        for delta in deltas {
            renderer.update_texture(device, queue, *id, delta);
        }
    }
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("egui") });
    renderer.update_buffers(device, queue, &mut encoder, &jobs, &screen);
    {
        let pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations { load: wgpu::LoadOp::Load, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        renderer.render(&mut pass.forget_lifetime(), &jobs, &screen);
    }
    queue.submit([encoder.finish()]);
    for id in &output.textures_delta.free {
        renderer.free_texture(id);
    }
    output.viewport_output.values().any(|v| v.repaint_delay < Duration::from_millis(100))
}

pub(crate) fn egui_renderer(gpu: &Gpu, format: wgpu::TextureFormat) -> egui_wgpu::Renderer {
    let options = egui_wgpu::RendererOptions { msaa_samples: 1, depth_stencil_format: None, dithering: true, ..Default::default() };
    egui_wgpu::Renderer::new(&gpu.device, format, options)
}

/// Wheel points turned into whole frames, faster when scrolling fast and for long
/// trajectories; `precise` (shift) scrolls frame by frame.
#[derive(Default)]
pub(crate) struct Scrub {
    accumulated: f32,
    last: Option<Instant>,
}

impl Scrub {
    pub fn frames(&mut self, points: f32, precise: bool, num_frames: usize) -> i64 {
        let elapsed = self.last.map_or(1.0, |t| t.elapsed().as_secs_f32());
        self.last = Some(Instant::now());
        self.accumulated -= points / SCROLL_PER_FRAME * scroll_gain(points, elapsed, precise, num_frames);
        let steps = self.accumulated.trunc();
        self.accumulated -= steps;
        steps as i64
    }
}

/// The 3D view's egui-independent keys, shared with live views: frames, discrete
/// states, zoom, the camera, the look, layers, the plots. True when `key` is one.
pub(crate) fn view_key(v: &mut Viewer, key: char, shift: bool) -> bool {
    // shift: jump 5% of the trajectory (at least 10 frames)
    let count = v.num_frames().max(1) as i64;
    let jump = if shift { (count / 20).max(10) } else { 1 };
    match key {
        ',' | '<' => step_frames(v, -jump),
        '.' | '>' => step_frames(v, jump),
        'd' => drop(v.set_scrub_discrete(!v.scene.scrub_discrete)),
        '0' => v.reset_camera(),
        '+' | '=' => v.zoom(1.0 / ZOOM_STEP),
        '-' => v.zoom(ZOOM_STEP),
        'l' => drop(v.set_shading(if v.shading() == "lit" { "flat" } else { "lit" })),
        's' if shift => drop(v.set_sidechains(if v.sidechains() == "sticks" { "spheres" } else { "sticks" })),
        's' => drop(v.set_style(if v.style() == "atoms" { "protein" } else { "atoms" })),
        'c' => next_coloring(v),
        'g' => v.show_plots = !v.show_plots,
        '1'..='9' => toggle_layer(v, key as usize - '1' as usize),
        _ => return false,
    }
    true
}

/// The egui keys of `view_key`, as characters.
const VIEW_KEYS: [(Key, char); 20] = [
    (Key::Comma, ','),
    (Key::Period, '.'),
    (Key::D, 'd'),
    (Key::Num0, '0'),
    (Key::Plus, '+'),
    (Key::Equals, '='),
    (Key::Minus, '-'),
    (Key::L, 'l'),
    (Key::S, 's'),
    (Key::C, 'c'),
    (Key::G, 'g'),
    (Key::Num1, '1'),
    (Key::Num2, '2'),
    (Key::Num3, '3'),
    (Key::Num4, '4'),
    (Key::Num5, '5'),
    (Key::Num6, '6'),
    (Key::Num7, '7'),
    (Key::Num8, '8'),
    (Key::Num9, '9'),
];

/// The explorer's window with egui on it.
struct Target {
    surface: Surface,
    egui_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
}

struct App<'a> {
    viewer: &'a mut Viewer,
    py: Python<'a>,
    options: Options,
    egui: egui::Context,
    surface: Option<Target>,
    /// the 3D view's rectangle in points, from the last layout
    viewport: egui::Rect,
    playing: bool,
    last_step: Instant,
    scrub: Scrub,
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

pub fn explore(viewer: &mut Viewer, py: Python<'_>, options: Options) -> PyResult<()> {
    with_event_loop(|event_loop| {
        let initial_scales = viewer.vector_scales();
        let mut app = App {
            viewer,
            py,
            options,
            egui: egui::Context::default(),
            surface: None,
            viewport: egui::Rect::NOTHING,
            playing: false,
            last_step: Instant::now(),
            scrub: Scrub::default(),
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

impl App<'_> {
    fn create_surface(&mut self, event_loop: &ActiveEventLoop) -> Result<Target, String> {
        let gpu = &self.viewer.gpu;
        let surface = Surface::new(event_loop, gpu, &self.options.title, self.options.size, true, false)?;
        let egui_state = egui_winit::State::new(
            self.egui.clone(),
            egui::ViewportId::ROOT,
            &surface.window,
            Some(surface.window.scale_factor() as f32),
            None,
            Some(gpu.device.limits().max_texture_dimension_2d as usize),
        );
        let egui_renderer = egui_renderer(gpu, surface.config.format);
        Ok(Target { surface, egui_state, egui_renderer })
    }

    fn redraw(&mut self) {
        if self.playing && self.last_step.elapsed() >= PLAY_INTERVAL {
            self.last_step = Instant::now();
            let count = self.viewer.num_frames().max(1);
            let frame = (self.viewer.frame + 1) % count;
            self.viewer.set_frame(frame);
        }
        let Some(mut target) = self.surface.take() else { return };
        let window = target.surface.window.clone();
        let size = target.surface.size();
        let format = target.surface.config.format;
        let gpu = self.viewer.gpu.clone();
        // the frame to draw, before egui runs: its output (the font atlas, the first
        // time) must reach the renderer, so a frame without a surface is drawn later
        let screenshot = if self.frames_drawn >= 2 { self.options.screenshot.take() } else { None };
        let mut offscreen = None;
        let surface_texture = match &screenshot {
            // the screenshot frame goes to an offscreen copy of the window instead
            Some(_) => {
                offscreen = Some(Offscreen::with_format(&gpu.device, size, format));
                None
            }
            None => match target.surface.acquire(&gpu.device) {
                Some(texture) => Some(texture),
                None => {
                    window.request_redraw();
                    self.surface = Some(target);
                    return;
                }
            },
        };
        let view = match (&surface_texture, &offscreen) {
            (Some(texture), _) => texture.texture.create_view(&Default::default()),
            (None, Some(offscreen)) => offscreen.view().clone(),
            _ => unreachable!(),
        };

        let raw_input = target.egui_state.take_egui_input(&window);
        let egui = self.egui.clone();
        let output = egui.run_ui(raw_input, |ui| self.ui(ui));
        self.record_step();
        let fullscreen = window.fullscreen().is_some();
        if std::mem::take(&mut self.toggle_fullscreen) {
            window.set_fullscreen(if fullscreen { None } else { Some(Fullscreen::Borderless(None)) });
        }
        if std::mem::take(&mut self.escape) {
            if fullscreen {
                window.set_fullscreen(None);
            } else {
                self.close_requested = true;
            }
        }
        target.egui_state.handle_platform_output(&window, output.platform_output.clone());
        let ppp = output.pixels_per_point;
        let viewport = [
            self.viewport.min.x * ppp,
            self.viewport.min.y * ppp,
            self.viewport.width() * ppp,
            self.viewport.height() * ppp,
        ];

        self.viewer.draw_into(&view, format, size, Some(viewport));
        let settling = paint_egui(&mut target.egui_renderer, &gpu, &egui, output, &view, size);
        match (surface_texture, screenshot, offscreen) {
            (Some(texture), _, _) => gpu.queue.present(texture),
            (None, Some(path), Some(offscreen)) => {
                let pixels = offscreen.read(&gpu);
                if let Err(error) = save_png(&path, size, format, &pixels) {
                    self.error = Some(crate::runtime_error(error));
                }
                self.surface = Some(target);
                self.close_requested = true;
                return;
            }
            _ => {}
        }
        self.frames_drawn += 1;
        if self.playing || !matches!(self.record, Record::Off) || self.options.screenshot.is_some() || settling {
            window.request_redraw();
        }
        self.surface = Some(target);
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        let dark = self.viewer.theme() == "dark";
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
        let v = &*self.viewer;
        if v.show_plots {
            v.plots.bottom_panel(ui, x);
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
        let v = &mut *self.viewer;
        for (key, c) in VIEW_KEYS {
            if pressed(key) {
                view_key(v, c, shift);
            }
        }
        if pressed(Key::Space) || pressed(Key::P) {
            self.playing = !self.playing;
        }
        let count = v.num_frames().max(1) as i64;
        for (key, c) in [(Key::ArrowLeft, ','), (Key::ArrowRight, '.')] {
            if pressed(key) {
                view_key(v, c, shift);
            }
        }
        for (key, step) in [(Key::Home, -count), (Key::End, count)] {
            if pressed(key) {
                step_frames(v, step);
            }
        }
        if pressed(Key::T) {
            let _ = v.set_theme(if v.theme() == "dark" { "light" } else { "dark" });
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
        let num_frames = self.viewer.num_frames().max(1);
        let steps = if scroll != 0.0 { self.scrub.frames(scroll, shift, num_frames) } else { 0 };

        let v = &mut *self.viewer;
        if response.dragged_by(PointerButton::Primary) && !shift {
            v.orbit(delta.x, delta.y);
        } else if response.dragged_by(PointerButton::Secondary) || (response.dragged_by(PointerButton::Primary) && shift) {
            v.pan(delta.x, delta.y, height);
        }
        if response.double_clicked() {
            v.reset_camera();
        }
        if zoom != 1.0 {
            v.zoom(1.0 / zoom);
        }
        if steps != 0 {
            step_frames(v, steps);
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui) {
        ui.spacing_mut().slider_width = 100.0;
        if let Some(info) = &self.options.info {
            ui.label(info.clone());
            ui.separator();
        }
        frame_controls(ui, self.viewer, &mut self.playing, "Space");
        self.record_controls(ui);
        ui.separator();
        view_controls(ui, self.viewer, &self.initial_scales, true);
        ui.separator();

        // copy buttons
        let mut copy = None;
        ui.horizontal_wrapped(|ui| {
            if ui.button("copy state").on_hover_text("the view as a dict: for live(view=...) in a deck").clicked() {
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
        let v = &*self.viewer;
        if !v.show_plots {
            v.plots.small(ui, x);
        }
        ui.add_space(8.0);
        let large = if v.plots.is_empty() { "" } else { "G large plots · " };
        ui.label(RichText::new(format!("? keys · {large}H hide panel · F full screen")).small().weak());
    }
}

/// The frame row of the panel: the slider, play / pause (its key hinted), discrete states.
pub(crate) fn frame_controls(ui: &mut egui::Ui, v: &mut Viewer, playing: &mut bool, play_key: &str) {
    let count = v.num_frames().max(1);
    let mut frame = v.frame;
    ui.horizontal(|ui| {
        ui.label("frame");
        if ui.add(egui::Slider::new(&mut frame, 0..=count - 1)).changed() {
            v.set_frame(frame);
        }
        hint(ui, ", .");
    });
    ui.horizontal(|ui| {
        if ui.button(if *playing { "Pause" } else { "Play" }).clicked() {
            *playing = !*playing;
        }
        hint(ui, play_key);
        if v.num_discrete_frames() > 0 {
            let mut on = v.scene.scrub_discrete;
            if ui.checkbox(&mut on, "scrub discrete").changed() {
                let _ = v.set_scrub_discrete(on);
            }
            hint(ui, "D");
        }
    });
}

/// The view's settings: appearance, layers and sizes, every one with its key hinted.
/// `initial_scales`: the vector lengths as given, the top of their sliders being 4 x that;
/// `theme`: with the theme choice (not in a deck, whose theme is the deck's).
pub(crate) fn view_controls(ui: &mut egui::Ui, v: &mut Viewer, initial_scales: &[(String, f32)], theme: bool) {
    ui.spacing_mut().slider_width = 100.0;
    egui::Grid::new("appearance").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
        if theme {
            choice(ui, "theme", "T", &["light", "dark"], v.theme(), |name| drop(v.set_theme(name)));
            ui.end_row();
        }
        choice(ui, "shading", "L", &["lit", "flat"], v.shading(), |s| drop(v.set_shading(s)));
            ui.end_row();
            if v.has_roles() {
                choice(ui, "style", "S", &["atoms", "protein"], v.style(), |s| drop(v.set_style(s)));
                ui.end_row();
                choice(ui, "side chains", "⇧S", &["sticks", "spheres"], v.sidechains(), |s| drop(v.set_sidechains(s)));
                ui.end_row();
            }
            let colorings = v.coloring_names();
            if !colorings.is_empty() {
                let names: Vec<&str> = std::iter::once("element").chain(colorings.iter().map(String::as_str)).collect();
                let current = v.scene.coloring.clone().unwrap_or_else(|| "element".into());
                let mut picked = None;
                choice(ui, "color", "C", &names, &current, |name| picked = Some(name.to_string()));
                if let Some(name) = picked {
                    let _ = v.set_coloring(Some(name).filter(|n| n != "element"));
                }
                ui.end_row();
            }
        });
        ui.separator();

        // layers
        for (index, (layer, visible)) in v.layers().into_iter().enumerate() {
            let mut on = visible;
            ui.horizontal(|ui| {
                if ui.checkbox(&mut on, layer.as_str()).changed() {
                    let _ = v.set_visible(&layer, on);
                }
                if index < 9 {
                    hint(ui, &(index + 1).to_string());
                }
            });
        }
        ui.separator();

        // sizes
        egui::Grid::new("sizes").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            let mut gamma = v.color_gamma();
            ui.label("color depth");
            if ui.add(egui::Slider::new(&mut gamma, 1.0..=2.4)).changed() {
                v.set_color_gamma(gamma);
            }
            ui.end_row();
            let mut fog = v.fog();
            ui.label("fog");
            if ui.add(egui::Slider::new(&mut fog, 0.0..=1.0)).changed() {
                v.set_fog(fog);
            }
            ui.end_row();
            let mut occlusion = v.scene.occlusion;
            ui.label("occlusion");
            if ui.add(egui::Slider::new(&mut occlusion, 0.0..=1.0)).changed() {
                v.set_occlusion(occlusion);
            }
            ui.end_row();
            let mut radius = v.scene.radius_scale;
            ui.label("atom radius");
            if ui.add(egui::Slider::new(&mut radius, 0.1..=2.0)).changed() {
                v.set_radius_scale(radius);
            }
            ui.end_row();
            let frames = v.num_path_frames();
            if frames > 2 {
                // the slider's top is the whole trails
                let mut window = v.trail_window().unwrap_or(frames - 1).min(frames - 1);
                ui.label("trail window");
                if ui.add(egui::Slider::new(&mut window, 0..=frames - 1)).changed() {
                    v.set_trail_window((window < frames - 1).then_some(window));
                }
                ui.end_row();
            }
            let mut opacity = v.scene.surface.opacity;
            ui.label("surface opacity");
            if ui.add(egui::Slider::new(&mut opacity, 0.05..=1.0)).changed() {
                v.set_surface_opacity(opacity);
            }
            ui.end_row();
            let by_atom = v.scene.surface.colors == crate::scene::SurfaceColors::Atoms;
            choice(ui, "surface color", "", &["uniform", "atoms"], if by_atom { "atoms" } else { "uniform" }, |c| {
                v.set_surface_colors_by_atom(c == "atoms")
            });
            ui.end_row();
            for (name, scale) in v.vector_scales() {
                let initial = initial_scales.iter().find(|(n, _)| *n == name).map_or(scale, |(_, s)| *s);
                let mut value = scale;
                ui.label(format!("{name} length"));
                if ui.add(egui::Slider::new(&mut value, 0.0..=(4.0 * initial).max(1e-3))).changed() {
                    let _ = v.set_vector_scale(&name, value);
                }
                ui.end_row();
            }
        });
}

impl App<'_> {
    fn record_controls(&mut self, ui: &mut egui::Ui) {
        let mut toggle = false;
        let mut cancel = false;
        ui.horizontal(|ui| match &self.record {
            Record::Off => {
                let available = self.options.recorder.is_some() || self.options.keyframes.is_some();
                let button = ui.add_enabled(available, egui::Button::new("Record"));
                if available {
                    let what = if self.options.recorder.is_some() {
                        "the video is rendered when you stop"
                    } else {
                        "what you do, as keyframes in a JSON file a deck can play"
                    };
                    toggle = button.on_hover_text(what).clicked();
                    hint(ui, "R");
                } else {
                    button.on_disabled_hover_text("a video needs ffmpeg: pip install 'llandau[video]'");
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

    /// Start recording, or stop it: save the keyframes, or start rendering the video.
    fn toggle_recording(&mut self, pixels_per_point: f32) {
        match std::mem::replace(&mut self.record, Record::Off) {
            Record::Off if self.options.recorder.is_some() || self.options.keyframes.is_some() => {
                let view = self.viewport.size() * pixels_per_point;
                let (width, height) = self.options.record_size.unwrap_or((view.x as u32, view.y as u32));
                // even sizes: what H.264 players take
                let size = (width.max(2) & !1, height.max(2) & !1);
                self.record = Record::On { start: Instant::now(), size, recording: Recording::default() };
                self.status.clear();
            }
            Record::Off => {}
            Record::On { recording, .. } if self.options.recorder.is_none() => {
                self.status = match self.save_keyframes(&recording) {
                    Ok(path) => format!("saved {path}"),
                    Err(error) => format!("{error}"),
                };
            }
            Record::On { recording, size, .. } => {
                let total = ((recording.duration() * self.options.record_fps).round() as usize).max(1);
                let recorder = self.options.recorder.as_ref().expect("recording without a recorder");
                match recorder.call1(self.py, size) {
                    Ok(writer) => {
                        let target = Box::new(Offscreen::with_format(&self.viewer.gpu.device, size, VIDEO_FORMAT));
                        self.record = Record::Rendering { recording, size, writer, target, next: 0, total };
                    }
                    Err(error) => self.status = format!("{error}"),
                }
            }
            rendering @ Record::Rendering { .. } => self.record = rendering,
        }
    }

    /// The recording's keyframes to the `keyframes` function: a list of (time, state dict).
    fn save_keyframes(&self, recording: &Recording) -> PyResult<String> {
        let py = self.py;
        let save = self.options.keyframes.as_ref().expect("keyframes without a writer");
        let samples: Vec<(f64, Py<pyo3::types::PyDict>)> =
            recording.samples().iter().map(|(t, s)| s.to_dict(py).map(|d| (*t, d.unbind()))).collect::<PyResult<_>>()?;
        save.call1(py, (samples,))?.extract(py)
    }

    /// While recording, keep the view's state whenever it changes; while
    /// rendering, render the video's next frames (the window then draws the
    /// live view again).
    fn record_step(&mut self) {
        match &mut self.record {
            Record::Off => {}
            Record::On { start, recording, .. } => {
                recording.push_change(start.elapsed().as_secs_f64(), State::capture(self.viewer));
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
        let live = State::capture(self.viewer);
        let mut current = live.clone();
        let started = Instant::now();
        let mut result = Ok(());
        while *next < *total && started.elapsed() < RENDER_BUDGET {
            let Some(state) = recording.at(*next as f64 / self.options.record_fps) else { break };
            state.apply(self.viewer, &current);
            current = state;
            let (width, height) = *size;
            self.viewer.draw_into(target.view(), VIDEO_FORMAT, *size, None);
            let pixels = target.read(&self.viewer.gpu);
            result = PyArray1::from_vec(self.py, pixels)
                .reshape([height as usize, width as usize, 4])
                .and_then(|frame| writer.call_method1(self.py, "write", (frame,)))
                .map(|_| ());
            if result.is_err() {
                break;
            }
            *next += 1;
        }
        live.apply(self.viewer, &current);
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

    /// Copy the view's state (or a button's text of the camera) to the clipboard, and print it.
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
        match which {
            None => {
                let state = State::capture(self.viewer).to_dict(self.py)?;
                let json = self.py.import("json")?;
                json.call_method1("dumps", (state,))?.extract()
            }
            Some(index) => {
                let camera = self.viewer.camera.to_dict(self.py)?;
                self.options.buttons[index].1.call1(self.py, (camera,))?.extract(self.py)
            }
        }
    }

    /// Where the plots' cursor is: the path parameter of the current frame.
    fn plot_x(&mut self) -> f64 {
        self.viewer.plot_x()
    }
}

pub(crate) fn step_frames(v: &mut Viewer, step: i64) {
    let count = v.num_frames().max(1) as i64;
    v.set_frame((v.frame as i64 + step).clamp(0, count - 1) as usize);
}

/// Element colors, then each named coloring in turn.
fn next_coloring(v: &mut Viewer) {
    let names = v.coloring_names();
    let next = match &v.scene.coloring {
        None => names.first().cloned(),
        Some(current) => names.iter().position(|n| n == current).and_then(|i| names.get(i + 1)).cloned(),
    };
    let _ = v.set_coloring(next);
}

/// The `index`th layer (as the panel lists them) on or off.
fn toggle_layer(v: &mut Viewer, index: usize) {
    if let Some((name, visible)) = v.layers().into_iter().nth(index) {
        let _ = v.set_visible(&name, !visible);
    }
}

/// Frames per scrolled point relative to SCROLL_PER_FRAME, for `scroll` points
/// `elapsed` seconds after the previous scroll event.
pub(crate) fn scroll_gain(scroll: f32, elapsed: f32, precise: bool, num_frames: usize) -> f32 {
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
    image::save_buffer(path, &rgba, size.0, size.1, image::ColorType::Rgba8).map_err(|e| e.to_string())
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.surface.is_some() {
            return;
        }
        match self.create_surface(event_loop) {
            Ok(target) => {
                target.surface.window.request_redraw();
                self.surface = Some(target);
            }
            Err(error) => {
                self.error = Some(crate::runtime_error(format!("cannot open a window: {error}")));
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(target) = self.surface.as_mut() else { return };
        let response = target.egui_state.on_window_event(&target.surface.window, &event);
        match event {
            WindowEvent::CloseRequested => self.close_requested = true,
            WindowEvent::Resized(size) => target.surface.resize(&self.viewer.gpu.device, size),
            WindowEvent::RedrawRequested => self.redraw(),
            _ => {
                if response.repaint {
                    target.surface.window.request_redraw();
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
            target.surface.window.request_redraw();
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
