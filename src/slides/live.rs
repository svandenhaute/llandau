//! Live 3D views in a deck: llandau viewers drawn into their regions of
//! the slide every frame, on the same GPU, and controlled with the mouse and
//! keys while their step is shown.
//!
//! In a live region: drag rotates, right (or shift) drag pans, the wheel scrubs
//! frames, pinch or ctrl + wheel zooms, a double click resets the camera. The
//! shown view also takes keys (KEYS, which ? shows over it): G shows the
//! viewer's plots large under its 3D view, readable on a projector.
//!
//! A view with an `opens` rectangle (a cell of a grid) opens up to it on a
//! click, drawn over the others and taking the keys, and closes on the next
//! click (a drag still turns it; the double click is then not special).
//!
//! Keys, the panel (H) and the keys shown (?) go to the focused view: the open
//! one, else the one the cursor was over last, else the topmost. The mouse over
//! the panel goes to the panel.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::paint::Painter;
use super::timeline::Timeline;
use super::zoom::Zoom;
use crate::Viewer;
use crate::plots::{self, PLOTS_HEIGHT};
use crate::recording::{Recording, State as ViewState};
use crate::render::Gpu;
use crate::window::{self, PANEL_WIDTH, PLAY_INTERVAL, Scrub, frame_controls, view_controls};

/// How long the camera takes to move to a step's camera, on the step's clock
/// (llandau.live.CAMERA_MOVE makes the step last that long).
const CAMERA_MOVE: f64 = 0.9;
const DOUBLE_CLICK: Duration = Duration::from_millis(350);
/// How long a view takes to open up or close.
const OPEN_MOVE: Duration = Duration::from_millis(300);
/// A press and release this close (pixels) is a click, not a drag.
const CLICK_SLOP: f64 = 4.0;
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
    ("0", "the step's view again (double click: the camera)"),
    ("click", "open up a grid's view, close it"),
    ("L", "lit / flat shading"),
    ("S", "atoms / protein (Shift: side chains)"),
    ("C", "next coloring"),
    ("1 – 9", "layers"), // replaced by the view's layer names
    ("G", "large plots"),
    ("H", "the panel of controls"),
    ("?", "these keys"),
    ("", "Slides"),
    ("→  Space", "next step"),
    ("←", "previous step"),
    ("Home  End", "first / last step"),
    ("R", "replay the step"),
    ("Cmd wheel", "zoom the slide (pinch too; over a 3D view: its camera)"),
    ("Cmd =  −  0", "zoom the slide in / out / whole; drag the slide to pan"),
    ("F", "full screen"),
    ("Esc  Q", "leave full screen, quit"),
];

/// What a live span does when its step starts.
pub struct Entry {
    /// called with the viewer: applies the step's view settings
    pub enter: Option<Py<PyAny>>,
    /// the camera to move to (a camera dict)
    pub camera: Option<Py<PyDict>>,
    /// keyframes (seconds from the step's start, a state dict) played on the step's clock
    pub play: Option<Vec<(f64, Py<PyDict>)>>,
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
    /// what the step plays on its clock (the recording, from the span's start)
    play: Option<(f64, Recording)>,
    /// the timeline time the recording was last applied at: at the same time
    /// again (paused), the presenter's turns are kept
    applied: Option<f64>,
    /// the step's view at its end: what 0 goes back to
    home: Option<ViewState>,
    playing: bool,
    last_step: Option<Instant>,
    /// the vector lengths as the scene set them: the panel's sliders go to 4 x that
    initial_scales: Vec<(String, f32)>,
}

/// A live span opening up to its `opens` rectangle, or closing again.
struct Open {
    span: usize,
    since: Instant,
    opening: bool,
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
    /// the span pressed with the left button, and where: a release nearby is a click
    press: Option<(usize, (f64, f64))>,
    open: Option<Open>,
    last_click: Option<Instant>,
    scrub: Scrub,
    /// the keys shown over the focused view
    help: bool,
    /// the panel of controls shown over the focused view (H)
    panel: bool,
    /// mouse events for the panel, in the overlay's points, until the next draw
    events: Vec<egui::Event>,
    /// a press went to the panel: its drag and release do too
    egui_pressed: bool,
    /// the span the cursor was over last: keys go to it while nothing is open
    hovered: Option<usize>,
    /// the window's size at the last mouse event, to place the cursor
    size: (u32, u32),
    /// the slide's zoom (the player's): the views are placed through it
    pub zoom: Zoom,
    /// egui over each viewer's picture: one context each, so that a press handled over
    /// one view is not taken up by the next view's pass (as a drag of its plot)
    overlays: HashMap<usize, Overlay>,
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

    /// Draw `contents` over `view` (`size` pixels), given `events`; whether egui wants another frame.
    fn draw(
        &mut self,
        gpu: &Gpu,
        view: &wgpu::TextureView,
        size: (u32, u32),
        dark: bool,
        events: Vec<egui::Event>,
        contents: impl FnMut(&mut egui::Ui),
    ) -> bool {
        let ppp = size.1 as f32 / OVERLAY_HEIGHT;
        let points = egui::vec2(size.0 as f32, size.1 as f32) / ppp;
        let mut input =
            egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, points)), events, ..Default::default() };
        input.viewports.entry(egui::ViewportId::ROOT).or_default().native_pixels_per_point = Some(ppp);
        self.ctx.set_visuals(if dark { egui::Visuals::dark() } else { egui::Visuals::light() });
        let output = self.ctx.run_ui(input, contents);
        window::paint_egui(&mut self.renderer, gpu, &self.ctx, output, view, size)
    }
}

fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

fn lerp_rect(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    std::array::from_fn(|i| a[i] + t * (b[i] - a[i]))
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
        if self.open.as_ref().is_some_and(|o| !active.contains(&o.span)) {
            self.open = None; // its step is over, and the panel and keys with it
            self.panel = false;
            self.help = false;
        }
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
            let mut v = viewer.borrow_mut();
            if state.initial_scales.is_empty() {
                state.initial_scales = v.vector_scales();
            }
            let start = ViewState::capture(&v);
            // what the step plays: its recording, or the move to its camera
            let mut recording = None;
            if let Some(frames) = &entry.play {
                let mut rec = Recording::default();
                let mut base = start.clone();
                for (t, dict) in frames {
                    let s = ViewState::from_dict(dict.bind(py), &base)?;
                    rec.push(*t, s.clone());
                    base = s;
                }
                recording = Some(rec);
            } else if let Some(camera) = &entry.camera {
                let mut target = start.clone();
                target.camera.update(camera.bind(py))?;
                if appearing {
                    target.apply(&mut v, &start);
                } else {
                    let mut rec = Recording::default();
                    rec.push(0.0, start.clone());
                    rec.push(CAMERA_MOVE, target);
                    recording = Some(rec);
                }
            }
            state.home = Some(match &recording {
                Some(rec) => rec.at(rec.duration()).unwrap_or_else(|| start.clone()),
                None => ViewState::capture(&v),
            });
            state.play = recording.map(|rec| (timeline.lives[span].span[0], rec));
            state.applied = None;
        }
        Ok(())
    }

    /// Advance playing trajectories; whether anything moves.
    pub fn animate(&mut self, py: Python<'_>) -> bool {
        let mut moving = false;
        for (index, state) in self.states.iter_mut().enumerate() {
            if state.entered.is_none() {
                continue;
            }
            let mut v = self.viewers[index].bind(py).borrow_mut();
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

    /// Where a span is drawn now, the rectangle it is heading for, and whether
    /// it is still moving.
    fn placement(&self, timeline: &Timeline, span: usize) -> ([f32; 4], [f32; 4], bool) {
        let live = timeline.lives[span];
        match (&self.open, live.opens) {
            (Some(open), Some(big)) if open.span == span => {
                let t = smooth((open.since.elapsed().as_secs_f32() / OPEN_MOVE.as_secs_f32()).min(1.0));
                let (from, to) = if open.opening { (live.rect, big) } else { (big, live.rect) };
                (lerp_rect(from, to, t), to, t < 1.0)
            }
            _ => (live.rect, live.rect, false),
        }
    }

    /// Open span `span` up, or close it when it is open (the panel and the keys
    /// go with it: a cell shows the molecule, not the controls).
    fn toggle(&mut self, span: usize) {
        let now = Instant::now();
        self.open = match self.open.take() {
            Some(open) if open.span == span => {
                // reverse from where it is
                let elapsed = open.since.elapsed().min(OPEN_MOVE);
                if open.opening {
                    self.panel = false;
                    self.help = false;
                }
                Some(Open { span, since: now - (OPEN_MOVE - elapsed), opening: !open.opening })
            }
            _ => Some(Open { span, since: now, opening: true }),
        };
    }

    /// Draw the live views shown at `time` into their pictures, at the pixel
    /// size their regions have in a target of `size`; whether to draw again soon
    /// (an overlay settling, a view opening up).
    pub fn draw(&mut self, py: Python<'_>, gpu: &Gpu, painter: &mut Painter, timeline: &Timeline, time: f64, size: (u32, u32)) -> bool {
        let focused = self.focused(timeline, time);
        let mut again = false;
        painter.live_top = None;
        for span in timeline.live_at(time).collect::<Vec<_>>() {
            let live = timeline.lives[span];
            let (now, _, moving) = self.placement(timeline, span);
            again |= moving;
            if now == live.rect {
                painter.live_rects.remove(&live.viewer);
            } else {
                painter.live_rects.insert(live.viewer, now);
            }
            if self.open.as_ref().is_some_and(|o| o.span == span) {
                if moving || self.open.as_ref().unwrap().opening {
                    painter.live_top = Some(live.viewer);
                } else {
                    self.open = None; // closed
                }
            }
            // drawn at the size it has now: a picture of another aspect ratio would stretch
            // the step's recording, at this moment of the timeline (once per moment: paused at
            // its mark, the presenter's turns are kept)
            let state = &mut self.states[live.viewer];
            if let Some((start, recording)) = &state.play
                && state.applied != Some(time)
            {
                let mut v = self.viewers[live.viewer].bind(py).borrow_mut();
                let current = ViewState::capture(&v);
                if let Some(wanted) = recording.at(time - start) {
                    wanted.apply(&mut v, &current);
                }
                state.applied = Some(time);
            }
            let rect = self.pixel_rect(timeline, size, now);
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
            let is_focused = focused == Some(live.viewer);
            let help = self.help && is_focused;
            let panel = self.panel && is_focused;
            // the plots go under the 3D view, which keeps the top
            let viewport = plots.then(|| [0.0, 0.0, pixels.0 as f32, ((1.0 - PLOTS_HEIGHT) * pixels.1 as f32).round()]);
            v.draw_into(&target.view, FORMAT, pixels, viewport);
            if plots || help || panel {
                let x = v.plot_x();
                let dark = v.theme() == "dark";
                let overlay = self.overlays.entry(live.viewer).or_insert_with(|| Overlay::new(gpu));
                let rows = help_rows(&v);
                let keys: Vec<(&str, &str)> = rows.iter().map(|(k, a)| (k.as_str(), a.as_str())).collect();
                let events = if is_focused { std::mem::take(&mut self.events) } else { Vec::new() };
                let state = &mut self.states[live.viewer];
                let mut playing = state.playing;
                let initial_scales = state.initial_scales.clone();
                again |= overlay.draw(gpu, &target.view, pixels, dark, events, |ui| {
                    if panel {
                        egui::Panel::right("live controls")
                            .exact_size(PANEL_WIDTH)
                            .resizable(false)
                            .show_separator_line(false)
                            .show(ui, |ui| {
                                egui::ScrollArea::vertical().auto_shrink(false).show(ui, |ui| {
                                    ui.set_max_width(PANEL_WIDTH - 24.0);
                                    frame_controls(ui, &mut v, &mut playing, "P");
                                    ui.separator();
                                    view_controls(ui, &mut v, &initial_scales, false);
                                    if !v.show_plots {
                                        ui.separator();
                                        v.plots.small(ui, x);
                                    }
                                    ui.add_space(8.0);
                                    ui.label(egui::RichText::new("? keys · H hide panel · 0 the step's view").small().weak());
                                });
                            });
                    }
                    if plots {
                        v.plots.bottom_panel(ui, x);
                    }
                    if help {
                        plots::help(ui, &keys, 15.0);
                    }
                });
                state.playing = playing;
            }
        }
        self.events.clear();
        again
    }

    /// Where the cursor is in the focused view's panel, in the overlay's points
    /// (None: no panel there, or the cursor is not over it); `anywhere`: the point
    /// over the whole view, for a drag that began in the panel.
    fn panel_point(&self, timeline: &Timeline, time: f64, size: (u32, u32), anywhere: bool) -> Option<egui::Pos2> {
        if !self.panel {
            return None;
        }
        let span = self.focused_span(timeline, time)?;
        let (now, _, _) = self.placement(timeline, span);
        let r = self.pixel_rect(timeline, size, now);
        let ppp = r[3] / OVERLAY_HEIGHT;
        let point = egui::pos2((self.cursor.0 as f32 - r[0]) / ppp, (self.cursor.1 as f32 - r[1]) / ppp);
        let inside = point.x >= 0.0 && point.y >= 0.0 && point.x <= r[2] / ppp && point.y <= OVERLAY_HEIGHT;
        let in_panel = inside && point.x >= r[2] / ppp - PANEL_WIDTH;
        (anywhere || in_panel).then_some(point)
    }

    /// The live span under the cursor at `time` (the open one first, then the
    /// topmost), its viewer and its pixel rectangle.
    fn under(&self, timeline: &Timeline, time: f64, size: (u32, u32), at: (f64, f64)) -> Option<(usize, usize, [f32; 4])> {
        let open = self.open.as_ref().map(|o| o.span).filter(|&s| timeline.live_at(time).any(|a| a == s));
        let rest = timeline.live_at(time).rev().filter(|&s| Some(s) != open);
        open.into_iter().chain(rest).find_map(|span| {
            let (now, _, _) = self.placement(timeline, span);
            let r = self.pixel_rect(timeline, size, now);
            let (x, y) = (at.0 as f32, at.1 as f32);
            (x >= r[0] && x <= r[0] + r[2] && y >= r[1] && y <= r[1] + r[3]).then_some((span, timeline.lives[span].viewer, r))
        })
    }

    /// The live span keys and the panel go to: the open one, else the one the cursor
    /// was over last (while it is shown), else the topmost one shown.
    fn focused_span(&self, timeline: &Timeline, time: f64) -> Option<usize> {
        let shown = |span: usize| timeline.live_at(time).any(|s| s == span);
        if let Some(open) = &self.open
            && shown(open.span)
        {
            return Some(open.span);
        }
        if let Some(span) = self.hovered
            && shown(span)
        {
            return Some(span);
        }
        timeline.live_at(time).last()
    }

    /// Note the span under the cursor, for the keys.
    fn hover(&mut self, timeline: &Timeline, time: f64, size: (u32, u32)) {
        self.size = size;
        if let Some((span, _, _)) = self.under(timeline, time, size, self.cursor) {
            self.hovered = Some(span);
        }
    }

    fn focused(&self, timeline: &Timeline, time: f64) -> Option<usize> {
        self.focused_span(timeline, time).map(|span| timeline.lives[span].viewer)
    }

    pub fn cursor_moved(&mut self, py: Python<'_>, timeline: &Timeline, time: f64, size: (u32, u32), at: (f64, f64)) -> bool {
        let (dx, dy) = ((at.0 - self.cursor.0) as f32, (at.1 - self.cursor.1) as f32);
        self.cursor = at;
        if self.drag.is_none() && !self.egui_pressed {
            self.hover(timeline, time, size);
        }
        if let Some(point) = self.panel_point(timeline, time, size, self.egui_pressed) {
            self.events.push(egui::Event::PointerMoved(point));
            return true;
        }
        let Some((viewer, pan)) = self.drag else { return false };
        let height = self.targets[viewer]
            .as_ref()
            .map(|t| t.size.1 as f32)
            .unwrap_or_else(|| self.pixel_rect(timeline, size, [0.0, 0.0, 1.0, 1.0])[3]);
        let mut v = self.viewers[viewer].bind(py).borrow_mut();
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
        if !pressed && self.egui_pressed {
            self.egui_pressed = false;
            if let Some(pos) = self.panel_point(timeline, time, size, true) {
                let modifiers = egui::Modifiers { shift, ..Default::default() };
                self.events.push(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: false, modifiers });
            }
            return true;
        }
        if pressed && !right
            && let Some(pos) = self.panel_point(timeline, time, size, false)
        {
            let modifiers = egui::Modifiers { shift, ..Default::default() };
            self.events.push(egui::Event::PointerButton { pos, button: egui::PointerButton::Primary, pressed: true, modifiers });
            self.egui_pressed = true;
            return true;
        }
        if !pressed {
            let dragged = self.drag.take().is_some();
            if let Some((span, at)) = self.press.take() {
                let moved = (self.cursor.0 - at.0).abs().max((self.cursor.1 - at.1).abs()) > CLICK_SLOP;
                if !moved {
                    self.toggle(span);
                    return true;
                }
            }
            return dragged;
        }
        let Some((span, viewer, _)) = self.under(timeline, time, size, self.cursor) else { return false };
        let opens = timeline.lives[span].opens.is_some();
        if !right && opens {
            self.press = Some((span, self.cursor));
        } else if !right {
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

    /// Scrolling over a live view: frames (or zoom, with ctrl); over its panel, the panel. True when taken.
    pub fn scroll(&mut self, py: Python<'_>, timeline: &Timeline, time: f64, size: (u32, u32), points: f32, zoom: bool) -> bool {
        if self.panel_point(timeline, time, size, false).is_some() {
            let delta = egui::vec2(0.0, points);
            self.events.push(egui::Event::MouseWheel {
                unit: egui::MouseWheelUnit::Point,
                delta,
                phase: egui::TouchPhase::Move,
                source: egui::MouseWheelSource::Unknown,
                modifiers: Default::default(),
            });
            return true;
        }
        let Some((_, viewer, _)) = self.under(timeline, time, size, self.cursor) else { return false };
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
        let Some((_, viewer, _)) = self.under(timeline, time, size, self.cursor) else { return false };
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
            "h" => self.panel = !self.panel,
            "0" => match &state.home {
                // the step's view again, the camera included
                Some(home) => {
                    let current = ViewState::capture(&v);
                    home.apply(&mut v, &current);
                }
                None => v.reset_camera(),
            },
            _ => {
                let Some(c) = key.chars().next().filter(|_| key.chars().count() == 1) else { return false };
                return window::view_key(&mut v, c, shift);
            }
        }
        true
    }
}

/// The keys shown over a live view: KEYS, with the view's layers by name.
fn help_rows(v: &Viewer) -> Vec<(String, String)> {
    let layers: Vec<String> = v.layers().into_iter().map(|(name, _)| name).collect();
    KEYS.iter()
        .filter_map(|(key, action)| match *key {
            "1 – 9" if layers.is_empty() => None,
            "1 – 9" => {
                let n = layers.len().min(9);
                let key = if n == 1 { "1".to_string() } else { format!("1 – {n}") };
                Some((key, format!("layers: {}", layers[..n].join(", "))))
            }
            _ => Some((key.to_string(), action.to_string())),
        })
        .collect()
}

impl Lives {
    /// A frame rectangle (center, width, height) in pixels of a target of `size`,
    /// through the zoom: left, top, width, height.
    fn pixel_rect(&self, timeline: &Timeline, size: (u32, u32), rect: [f32; 4]) -> [f32; 4] {
        let [cx, cy, w, h] = rect;
        let [left, top] = self.zoom.pixel_of(timeline.frame, size, [cx - w / 2.0, cy + h / 2.0]);
        let [right, bottom] = self.zoom.pixel_of(timeline.frame, size, [cx + w / 2.0, cy - h / 2.0]);
        [left, top, right - left, bottom - top]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slides::timeline::LiveSpan;

    fn grid() -> Timeline {
        let mut timeline = Timeline { frame: [16.0, 9.0], ..Default::default() };
        for (k, cx) in [-4.0, 0.0, 4.0].into_iter().enumerate() {
            timeline.lives.push(LiveSpan { viewer: k, rect: [cx, 0.0, 3.0, 2.0], span: [0.0, 10.0], opens: Some([0.0, 0.0, 12.0, 8.0]) });
        }
        timeline
    }

    #[test]
    fn a_click_opens_a_cell_and_the_next_closes_it() {
        let timeline = grid();
        let mut lives = Lives::default();
        let size = (1600, 900);
        // the middle cell is under the frame's center; the left one is not
        assert_eq!(lives.under(&timeline, 1.0, size, (800.0, 450.0)).map(|u| u.0), Some(1));
        assert_eq!(lives.under(&timeline, 1.0, size, (400.0, 450.0)).map(|u| u.0), Some(0));
        lives.toggle(0);
        let (now, heading, moving) = lives.placement(&timeline, 0);
        assert!(moving && heading == [0.0, 0.0, 12.0, 8.0]);
        assert!(now[2] >= 3.0 && now[2] <= 12.0);
        // the open cell is found first, even where it overlaps the others
        lives.open.as_mut().unwrap().since = Instant::now() - OPEN_MOVE;
        assert_eq!(lives.under(&timeline, 1.0, size, (800.0, 450.0)).map(|u| u.0), Some(0));
        assert_eq!(lives.focused(&timeline, 1.0), Some(0));
        let (now, _, moving) = lives.placement(&timeline, 0);
        assert!(!moving && now == [0.0, 0.0, 12.0, 8.0]);
        lives.toggle(0);
        assert!(!lives.open.as_ref().unwrap().opening);
        lives.open.as_mut().unwrap().since = Instant::now() - OPEN_MOVE;
        let (now, heading, moving) = lives.placement(&timeline, 0);
        assert!(!moving && now == [-4.0, 0.0, 3.0, 2.0] && heading == now);
        // a view whose step ended is closed
        lives.toggle(2);
        let mut later = grid();
        later.lives[2].span = [0.0, 0.5];
        assert!(lives.open.is_some());
        let active: Vec<usize> = later.live_at(1.0).collect();
        assert!(!active.contains(&2));
    }

    #[test]
    fn the_panel_takes_the_right_of_the_focused_view() {
        let timeline = grid();
        let mut lives = Lives::default();
        let size = (1600, 900);
        lives.toggle(0);
        lives.open.as_mut().unwrap().since = Instant::now() - OPEN_MOVE;
        // open to 12 x 8 of the 16 x 9 frame: pixels 200..1400 wide, 50..850 tall; 800 px = 600 points
        lives.cursor = (1350.0, 450.0);
        assert!(lives.panel_point(&timeline, 1.0, size, false).is_none(), "no panel yet");
        lives.panel = true;
        let point = lives.panel_point(&timeline, 1.0, size, false).expect("over the panel");
        assert!((point.x - 862.5).abs() < 1e-3 && (point.y - 300.0).abs() < 1e-3);
        lives.cursor = (300.0, 450.0);
        assert!(lives.panel_point(&timeline, 1.0, size, false).is_none(), "the view, not its panel");
        assert!(lives.panel_point(&timeline, 1.0, size, true).is_some(), "a drag from the panel follows");
        lives.help = true;
        lives.toggle(0); // closing hides the panel and the keys
        assert!(!lives.panel && !lives.help);
    }

    #[test]
    fn keys_go_to_the_view_under_the_cursor_unless_one_is_open() {
        let timeline = grid();
        let mut lives = Lives::default();
        let size = (1600, 900);
        assert_eq!(lives.focused_span(&timeline, 1.0), Some(2), "nothing hovered: the topmost");
        lives.cursor = (400.0, 450.0);
        lives.hover(&timeline, 1.0, size);
        assert_eq!(lives.focused_span(&timeline, 1.0), Some(0));
        lives.cursor = (10.0, 10.0); // off every view: the last one hovered keeps the keys
        lives.hover(&timeline, 1.0, size);
        assert_eq!(lives.focused_span(&timeline, 1.0), Some(0));
        lives.toggle(2);
        assert_eq!(lives.focused_span(&timeline, 1.0), Some(2), "the open view");
    }

    #[test]
    fn rect_interpolation() {
        assert_eq!(lerp_rect([0.0, 0.0, 2.0, 2.0], [4.0, 4.0, 6.0, 6.0], 0.5), [2.0, 2.0, 4.0, 4.0]);
    }
}
