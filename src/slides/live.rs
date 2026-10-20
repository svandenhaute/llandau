//! Live 3D views in a deck: riemann_viewer viewers drawn into their regions of
//! the slide every frame, on the same GPU, and controlled with the mouse and
//! keys while their step is shown.
//!
//! In a live region: drag rotates, right (or shift) drag pans, the wheel scrubs
//! frames, pinch or ctrl + wheel zooms, a double click resets the camera.
//! Keys: , and . step frames (with shift: 5% of them), P plays the trajectory,
//! D toggles the discrete states, 0 resets the camera.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::types::PyDict;

use super::paint::{Painter, letterbox};
use super::timeline::Timeline;
use crate::Viewer;
use crate::camera::Camera;
use crate::render::Gpu;
use crate::window::Controls;

/// How long the camera takes to move to a step's view.
const CAMERA_MOVE: Duration = Duration::from_millis(900);
const PLAY_INTERVAL: Duration = Duration::from_millis(33);
/// Wheel points per frame when scrubbing slowly.
const SCROLL_PER_FRAME: f32 = 12.0;
const DOUBLE_CLICK: Duration = Duration::from_millis(350);
/// The picture's format: drawn sRGB-encoded, sampled as written.
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

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
    scroll: f32,
}

#[derive(Default)]
pub struct Lives {
    pub viewers: Vec<Py<Viewer>>,
    /// one per timeline live span
    pub entries: Vec<Entry>,
    targets: HashMap<usize, Target>,
    states: HashMap<usize, State>,
    /// (viewer, button) being dragged, and the last cursor position
    drag: Option<(usize, bool)>,
    cursor: (f64, f64),
    last_click: Option<Instant>,
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
        self.viewers.len() - 1
    }

    /// Enter the live spans shown at `time` that were not entered yet: apply
    /// their views; the camera moves there unless the view just appeared.
    pub fn enter(&mut self, py: Python<'_>, timeline: &Timeline, time: f64) -> PyResult<()> {
        let active: Vec<usize> = timeline.live_at(time).collect();
        let shown: Vec<usize> = active.iter().map(|&i| timeline.lives[i].viewer).collect();
        // views no longer shown start afresh next time
        for (viewer, state) in self.states.iter_mut() {
            if !shown.contains(viewer) {
                state.entered = None;
                state.playing = false;
            }
        }
        for span in active {
            let viewer_index = timeline.lives[span].viewer;
            let state = self.states.entry(viewer_index).or_default();
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
        for (&index, state) in self.states.iter_mut() {
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
                    let count = Controls::num_frames(&*v).max(1);
                    let frame = (Controls::frame(&*v) + 1) % count;
                    Controls::set_frame(&mut *v, frame);
                }
                moving = true;
            }
        }
        moving
    }

    /// Draw the live views shown at `time` into their pictures, at the pixel
    /// size their regions have in a target of `size`.
    pub fn draw(&mut self, py: Python<'_>, gpu: &Gpu, painter: &mut Painter, timeline: &Timeline, time: f64, size: (u32, u32)) {
        for span in timeline.live_at(time).collect::<Vec<_>>() {
            let live = timeline.lives[span];
            let rect = pixel_rect(timeline, size, live.rect);
            let pixels = (rect[2].round().max(1.0) as u32, rect[3].round().max(1.0) as u32);
            if !matches!(self.targets.get(&live.viewer), Some(t) if t.size == pixels) {
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
                self.targets.insert(live.viewer, Target { size: pixels, view });
            }
            let target = &self.targets[&live.viewer];
            self.viewers[live.viewer].bind(py).borrow_mut().draw_into_view(&target.view, FORMAT, pixels);
        }
    }

    /// The live view under the cursor at `time` (the topmost), and its pixel rectangle.
    fn under(&self, timeline: &Timeline, time: f64, size: (u32, u32), at: (f64, f64)) -> Option<(usize, [f32; 4])> {
        timeline.live_at(time).collect::<Vec<_>>().into_iter().rev().find_map(|span| {
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
        let height = self
            .targets
            .get(&viewer)
            .map(|t| t.size.1 as f32)
            .unwrap_or_else(|| pixel_rect(timeline, size, [0.0, 0.0, 1.0, 1.0])[3]);
        let mut v = self.viewers[viewer].bind(py).borrow_mut();
        if let Some(state) = self.states.get_mut(&viewer) {
            state.camera_move = None;
        }
        if pan {
            Controls::pan(&mut *v, dx, dy, height);
        } else {
            Controls::orbit(&mut *v, dx, dy);
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
                Controls::reset_camera(&mut *self.viewers[viewer].bind(py).borrow_mut());
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
            Controls::zoom(&mut *v, (-points * 0.004).exp());
            return true;
        }
        let state = self.states.entry(viewer).or_default();
        let count = Controls::num_frames(&*v).max(1);
        let gain = (count as f32 / 200.0).max(1.0).sqrt();
        state.scroll -= points / SCROLL_PER_FRAME * gain;
        let steps = state.scroll.trunc();
        state.scroll -= steps;
        if steps != 0.0 {
            let frame = (Controls::frame(&*v) as i64 + steps as i64).clamp(0, count as i64 - 1);
            Controls::set_frame(&mut *v, frame as usize);
        }
        true
    }

    /// Pinching over a live view zooms. True when taken.
    pub fn pinch(&mut self, py: Python<'_>, timeline: &Timeline, time: f64, size: (u32, u32), delta: f64) -> bool {
        let Some((viewer, _)) = self.under(timeline, time, size, self.cursor) else { return false };
        Controls::zoom(&mut *self.viewers[viewer].bind(py).borrow_mut(), 1.0 / (1.0 + delta as f32).max(0.1));
        true
    }

    /// A key for the shown live view: true when taken.
    pub fn key(&mut self, py: Python<'_>, timeline: &Timeline, time: f64, key: &str, shift: bool) -> bool {
        let Some(viewer) = self.focused(timeline, time) else { return false };
        let mut v = self.viewers[viewer].bind(py).borrow_mut();
        let count = Controls::num_frames(&*v).max(1);
        let state = self.states.entry(viewer).or_default();
        let step = if shift { (count / 20).max(10) } else { 1 } as i64;
        match key {
            "," | "<" => {
                let frame = (Controls::frame(&*v) as i64 - step).max(0);
                Controls::set_frame(&mut *v, frame as usize);
            }
            "." | ">" => {
                let frame = (Controls::frame(&*v) as i64 + step).min(count as i64 - 1);
                Controls::set_frame(&mut *v, frame as usize);
            }
            "p" => state.playing = !state.playing,
            "d" => {
                let on = Controls::scrub_discrete(&*v);
                Controls::set_scrub_discrete(&mut *v, !on);
            }
            "0" => {
                state.camera_move = None;
                Controls::reset_camera(&mut *v);
            }
            _ => return false,
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
