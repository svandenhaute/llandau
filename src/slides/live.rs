//! Live 3D views in a deck: llandau viewers drawn into their regions of
//! the slide every frame, on the same GPU, and controlled with the mouse and
//! keys while their step is shown.
//!
//! In a live region: drag rotates, right (or shift) drag pans, the wheel scrubs
//! frames, pinch or ctrl + wheel zooms, a double click resets the camera. The
//! shown view also takes keys (KEYS, which ? shows over it): G shows the
//! viewer's plots large under its 3D view, readable on a projector.

use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::paint::{Painter, letterbox};
use super::timeline::Timeline;
use crate::Viewer;
use crate::camera::Camera;
use crate::plots::{self, PLOTS_HEIGHT};
use crate::render::Gpu;
use crate::window::{self, PLAY_INTERVAL, Scrub};

/// How long the camera takes to move to a step's view.
const CAMERA_MOVE: Duration = Duration::from_millis(900);
const DOUBLE_CLICK: Duration = Duration::from_millis(350);
/// The picture's format: drawn sRGB-encoded, sampled as written.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
/// The overlay's scale: a live view is this many points high (text sizes are in points).
const OVERLAY_HEIGHT: f32 = 600.0;

/// The keys of a shown live view, as ? shows them ("" keys head groups), and the
/// deck's own.
pub(crate) const KEYS: &[(&str, &str)] = &[
    ("", "Live view"),
    ("P", "play / pause"),
    (", .", "previous / next frame (Shift: 5%)"),
    ("D", "discrete states"),
    ("drag", "rotate (right or Shift: pan)"),
    ("wheel", "scrub frames (pinch or Ctrl: zoom)"),
    ("+  −", "zoom in / out"),
    ("0", "reset the camera (or double click)"),
    ("L", "lit / flat shading"),
    ("S", "atoms / protein (Shift: side chains)"),
    ("C", "next coloring"),
    ("1 – 9", "layers"),
    ("G", "large plots"),
    ("?", "these keys"),
    ("", "Slides"),
    ("→  Space", "next step"),
    ("←", "previous step"),
    ("Home  End", "first / last step"),
    ("R", "replay the step"),
    ("F", "full screen"),
    ("Esc  Q", "leave full screen, quit"),
];

/// What a live span does when its step starts.
pub struct Entry {
    /// called with the viewer: applies the step's view settings
    pub enter: Option<Py<PyAny>>,
    /// the camera to move to (a camera dict)
    pub camera: Option<Py<PyDict>>,
}

struct Target {
    size: (u32, u32),
    /// drawn into (sRGB)
    view: wgpu::TextureView,
}

#[derive(Default)]
struct State {
    /// the live span entered last
    entered: Option<usize>,
    camera_move: Option<(Instant, Camera, Camera)>,
    playing: bool,
    last_step: Option<Instant>,
}

#[derive(Default)]
pub struct Lives {
    pub viewers: Vec<Py<Viewer>>,
    /// one per timeline live span
    pub entries: Vec<Entry>,
    /// per viewer
    targets: Vec<Option<Target>>,
    states: Vec<State>,
    /// (viewer, button) being dragged, and the last cursor position
    drag: Option<(usize, bool)>,
    cursor: (f64, f64),
    last_click: Option<Instant>,
    scrub: Scrub,
    /// the keys shown over the focused view
    help: bool,
    overlay: Option<Overlay>,
}

/// egui over a live view's picture: its plots and the keys.
struct Overlay {
    ctx: egui::Context,
    renderer: egui_wgpu::Renderer,
}

impl Overlay {
    fn new(gpu: &Gpu) -> Self {
        Self { ctx: egui::Context::default(), renderer: window::egui_renderer(gpu, FORMAT) }
    }

    /// Draw `contents` over `view` (`size` pixels); whether egui wants another frame.
    fn draw(&mut self, gpu: &Gpu, view: &wgpu::TextureView, size: (u32, u32), dark: bool, contents: impl FnMut(&mut egui::Ui)) -> bool {
        let ppp = size.1 as f32 / OVERLAY_HEIGHT;
        let points = egui::vec2(size.0 as f32, size.1 as f32) / ppp;
        let mut input = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, points)), ..Default::default() };
        input.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp);
        self.ctx.set_visuals(if dark { egui::Visuals::dark() } else { egui::Visuals::light() });
        let output = self.ctx.run_ui(input, contents);
        window::paint_egui(&mut self.renderer, gpu, &self.ctx, output, view, size)
    }
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

impl Lives {
    /// The index of `viewer`, added if new.
    pub fn viewer_index(&mut self, viewer: Py<Viewer>) -> usize {
        if let Some(i) = self.viewers.iter().position(|v| v.as_ptr() == viewer.as_ptr()) {
            return i;
        }
        self.viewers.push(viewer);
        self.targets.push(None);
        self.states.push(State::default());
        self.viewers.len() - 1
    }

    /// Enter the live spans shown at `time` that were not entered yet: apply
    /// their views; the camera moves there unless the view just appeared.
    pub fn enter(&mut self, py: Python<'_>, timeline: &Timeline, time: f64) -> PyResult<()> {
        let active: Vec<usize> = timeline.live_at(time).collect();
        let shown: Vec<usize> = active.iter().map(|&i| timeline.lives[i].viewer).collect();
        // views no longer shown start afresh next time
        for (viewer, state) in self.states.iter_mut().enumerate() {
            if !shown.contains(&viewer) {
                state.entered = None;
                state.playing = false;
            }
        }
        for span in active {
            let viewer_index = timeline.lives[span].viewer;
            let state = &mut self.states[viewer_index];
            if state.entered == Some(span) {
                continue;
            }
            let appearing = state.entered.is_none();
            state.entered = Some(span);
            let viewer = self.viewers[viewer_index].bind(py);
            let entry = &self.entries[span];
            if let Some(enter) = &entry.enter {
                enter.call1(py, (viewer,))?;
            }
            if let Some(camera) = &entry.camera {
                let mut v = viewer.borrow_mut();
                let start = v.camera;
                v.set_camera(camera.bind(py).clone())?;
                let target = v.camera;
                if appearing {
                    state.camera_move = None;
                } else {
                    v.camera = start;
                    state.camera_move = Some((Instant::now(), start, target));
                }
            }
        }
        Ok(())
    }

    /// Advance camera moves and playing trajectories; whether anything moves.
    pub fn animate(&mut self, py: Python<'_>) -> bool {
        let mut moving = false;
        for (index, state) in self.states.iter_mut().enumerate() {
            if state.entered.is_none() {
                continue;
            }
            let mut v = self.viewers[index].bind(py).borrow_mut();
            if let Some((started, from, to)) = state.camera_move {
                let t = (started.elapsed().as_secs_f32() / CAMERA_MOVE.as_secs_f32()).min(1.0);
                v.camera = from.lerp(&to, smooth(t));
                if t >= 1.0 {
                    state.camera_move = None;
                }
                moving = true;
            }
            if state.playing {
                if state.last_step.is_none_or(|t| t.elapsed() >= PLAY_INTERVAL) {
                    state.last_step = Some(Instant::now());
                    let count = v.num_frames().max(1);
                    let frame = (v.frame + 1) % count;
                    v.set_frame(frame);
                }
                moving = true;
            }
        }
        moving
    }

    /// Draw the live views shown at `time` into their pictures, at the pixel
    /// size their regions have in a target of `size`; whether to draw again soon
    /// (an overlay settling).
    pub fn draw(&mut self, py: Python<'_>, gpu: &Gpu, painter: &mut Painter, timeline: &Timeline, time: f64, size: (u32, u32)) -> bool {
        let focused = self.focused(timeline, time);
        let mut again = false;
        for span in timeline.live_at(time).collect::<Vec<_>>() {
            let live = timeline.lives[span];
            let rect = pixel_rect(timeline, size, live.rect);
            let pixels = (rect[2].round().max(1.0) as u32, rect[3].round().max(1.0) as u32);
            if !matches!(&self.targets[live.viewer], Some(t) if t.size == pixels) {
                let texture = gpu.device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("live view"),
                    size: wgpu::Extent3d { width: pixels.0, height: pixels.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: FORMAT,
                    usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[wgpu::TextureFormat::Rgba8Unorm],
                });
                let view = texture.create_view(&Default::default());
                // the deck blends sRGB values as they are: sample without decoding
                let sampled = texture.create_view(&wgpu::TextureViewDescriptor {
                    format: Some(wgpu::TextureFormat::Rgba8Unorm),
                    ..Default::default()
                });
                painter.set_live(&gpu.device, live.viewer, &sampled);
                self.targets[live.viewer] = Some(Target { size: pixels, view });
            }
            let target = self.targets[live.viewer].as_ref().unwrap();
            let mut v = self.viewers[live.viewer].bind(py).borrow_mut();
            let plots = v.show_plots && !v.plots.is_empty();
            let help = self.help && focused == Some(live.viewer);
            // the plots go under the 3D view, which keeps the top
            let viewport = plots.then(|| [0.0, 0.0, pixels.0 as f32, ((1.0 - PLOTS_HEIGHT) * pixels.1 as f32).round()]);
            v.draw_into(&target.view, FORMAT, pixels, viewport);
            if plots || help {
                let x = v.plots.x(v.frame, v.num_frames(), v.scene.scrub_discrete);
                let dark = v.theme() == "dark";
                let overlay = self.overlay.get_or_insert_with(|| Overlay::new(gpu));
                again |= overlay.draw(gpu, &target.view, pixels, dark, |ui| {
                    if plots {
                        v.plots.bottom_panel(ui, x);
                    }
                    if help {
                        plots::help(ui, KEYS, 15.0);
                    }
                });
            }
        }
        again
    }

    /// The live view under the cursor at `time` (the topmost), and its pixel rectangle.
    fn under(&self, timeline: &Timeline, time: f64, size: (u32, u32), at: (f64, f64)) -> Option<(usize, [f32; 4])> {
        timeline.live_at(time).rev().find_map(|span| {
            let live = timeline.lives[span];
            let r = pixel_rect(timeline, size, live.rect);
            let (x, y) = (at.0 as f32, at.1 as f32);
            (x >= r[0] && x <= r[0] + r[2] && y >= r[1] && y <= r[1] + r[3]).then_some((live.viewer, r))
        })
    }

    /// The live view keys go to: the topmost one shown.
    fn focused(&self, timeline: &Timeline, time: f64) -> Option<usize> {
        timeline.live_at(time).last().map(|span| timeline.lives[span].viewer)
    }

    pub fn cursor_moved(&mut self, py: Python<'_>, timeline: &Timeline, size: (u32, u32), at: (f64, f64)) -> bool {
        let (dx, dy) = ((at.0 - self.cursor.0) as f32, (at.1 - self.cursor.1) as f32);
        self.cursor = at;
        let Some((viewer, pan)) = self.drag else { return false };
        let height = self.targets[viewer]
            .as_ref()
            .map(|t| t.size.1 as f32)
            .unwrap_or_else(|| pixel_rect(timeline, size, [0.0, 0.0, 1.0, 1.0])[3]);
        let mut v = self.viewers[viewer].bind(py).borrow_mut();
        self.states[viewer].camera_move = None;
        if pan {
            v.pan(dx, dy, height);
        } else {
            v.orbit(dx, dy);
        }
        true
    }

    /// A mouse button: true when a live view took it.
    pub fn button(
        &mut self,
        py: Python<'_>,
        timeline: &Timeline,
        time: f64,
        size: (u32, u32),
        pressed: bool,
        right: bool,
        shift: bool,
    ) -> bool {
        if !pressed {
            return self.drag.take().is_some();
        }
        let Some((viewer, _)) = self.under(timeline, time, size, self.cursor) else { return false };
        if !right {
            let now = Instant::now();
            if self.last_click.is_some_and(|t| now - t < DOUBLE_CLICK) {
                self.viewers[viewer].bind(py).borrow_mut().reset_camera();
                self.last_click = None;
                return true;
            }
            self.last_click = Some(now);
        }
        self.drag = Some((viewer, right || shift));
        true
    }

    /// Scrolling over a live view: frames (or zoom, with ctrl). True when taken.
    pub fn scroll(&mut self, py: Python<'_>, timeline: &Timeline, time: f64, size: (u32, u32), points: f32, zoom: bool) -> bool {
        let Some((viewer, _)) = self.under(timeline, time, size, self.cursor) else { return false };
        let mut v = self.viewers[viewer].bind(py).borrow_mut();
        if zoom {
            v.zoom((-points * 0.004).exp());
            return true;
        }
        let steps = self.scrub.frames(points, false, v.num_frames());
        if steps != 0 {
            window::step_frames(&mut v, steps);
        }
        true
    }

    /// Pinching over a live view zooms. True when taken.
    pub fn pinch(&mut self, py: Python<'_>, timeline: &Timeline, time: f64, size: (u32, u32), delta: f64) -> bool {
        let Some((viewer, _)) = self.under(timeline, time, size, self.cursor) else { return false };
        self.viewers[viewer].bind(py).borrow_mut().zoom(1.0 / (1.0 + delta as f32).max(0.1));
        true
    }

    /// A key for the shown live view: true when taken.
    pub fn key(&mut self, py: Python<'_>, timeline: &Timeline, time: f64, key: &str, shift: bool) -> bool {
        let Some(viewer) = self.focused(timeline, time) else { return false };
        let mut v = self.viewers[viewer].bind(py).borrow_mut();
        let state = &mut self.states[viewer];
        match key {
            "p" => state.playing = !state.playing,
            "?" => self.help = !self.help,
            _ => {
                let Some(c) = key.chars().next().filter(|_| key.chars().count() == 1) else { return false };
                if c == '0' {
                    state.camera_move = None;
                }
                return window::view_key(&mut v, c, shift);
            }
        }
        true
    }
}

/// A frame rectangle (center, width, height) in pixels of a target of `size`:
/// left, top, width, height.
pub fn pixel_rect(timeline: &Timeline, size: (u32, u32), rect: [f32; 4]) -> [f32; 4] {
    let [vx, vy, vw, vh] = letterbox(timeline.frame, size);
    let [fw, fh] = timeline.frame;
    let [cx, cy, w, h] = rect;
    let left = vx + (cx - w / 2.0 + fw / 2.0) / fw * vw;
    let top = vy + (fh / 2.0 - (cy + h / 2.0)) / fh * vh;
    [left, top, w / fw * vw, h / fh * vh]
}
