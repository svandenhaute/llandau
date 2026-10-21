//! Presenting a timeline in a native window.
//!
//! Keys: → space PageDown ↓ play the next step (or finish the current one),
//! ← PageUp ↑ go back one step, Home and End go to the first and last step,
//! R replays the step, F toggles full screen, Escape leaves full screen or
//! closes, Q closes. Scrolling scrubs through time, faster when scrolling fast.
//! Over a live 3D view, the mouse and some keys go to it (see `live`; ? shows them).

use std::time::Instant;

use pyo3::prelude::*;
use winit::application::ApplicationHandler;
use winit::event::{ElementState, KeyEvent, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::ActiveEventLoop;
use winit::event_loop::ControlFlow;
use winit::keyboard::{Key, ModifiersState, NamedKey};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Fullscreen, WindowId};

use super::live::Lives;
use super::paint::{Image, Painter};
use super::timeline::Timeline;
use crate::render::{Gpu, Offscreen};
use crate::window::{self, Surface};

/// Seconds of the timeline per scrolled point (line: 20 points), before acceleration.
const SCRUB_PER_POINT: f64 = 0.004;
/// A paused step shows the deck this long before its mark.
const REST_BEFORE_MARK: f64 = 1e-6;

pub struct Options {
    pub title: String,
    pub size: (f64, f64),
    pub fullscreen: bool,
    /// the step to start at
    pub start: usize,
    /// draw the end of the start step into this PNG file and close
    pub screenshot: Option<String>,
}

/// Scripted input, for tests: LLANDAU_DRIVE names a file of commands, one per line, run in
/// order once the window is up: `move X Y` (pixels), `press`, `release`, `rpress`, `rrelease`,
/// `wheel DY`, `key K` (a character, or right, left, space, home, end, escape), `wait MS`,
/// `shot PATH` (a PNG of the window now), `quit`.
struct Driver {
    commands: Vec<String>,
    next: usize,
    due: Instant,
}

struct App<'a> {
    py: Python<'a>,
    gpu: &'a Gpu,
    painter: &'a mut Painter,
    timeline: &'a Timeline,
    images: &'a [Image],
    lives: &'a mut Lives,
    options: Options,
    modifiers: ModifiersState,
    surface: Option<Surface>,
    /// the current step: the presentation pauses at its mark
    step: usize,
    time: f64,
    /// playing towards this time, from (instant, time) when it started
    playing: Option<(Instant, f64, f64)>,
    last_scroll: Instant,
    closing: bool,
    error: Option<PyErr>,
    driver: Option<Driver>,
}

pub fn play(
    py: Python<'_>,
    gpu: &Gpu,
    painter: &mut Painter,
    timeline: &Timeline,
    images: &[Image],
    lives: &mut Lives,
    options: Options,
) -> PyResult<()> {
    crate::window::with_event_loop(|event_loop| {
        let start = options.start.min(timeline.marks.len().saturating_sub(1));
        let mut app = App {
            py,
            gpu,
            painter,
            timeline,
            images,
            lives,
            options,
            modifiers: ModifiersState::empty(),
            surface: None,
            step: start,
            time: 0.0,
            playing: None,
            last_scroll: Instant::now(),
            closing: false,
            error: None,
            driver: std::env::var("LLANDAU_DRIVE").ok().and_then(|path| {
                let text = std::fs::read_to_string(&path).ok()?;
                Some(Driver { commands: text.lines().map(String::from).collect(), next: 0, due: Instant::now() })
            }),
        };
        if start == 0 && app.options.screenshot.is_none() {
            app.play_to(app.mark(0));
        } else {
            app.time = app.mark(start);
        }
        event_loop.run_app_on_demand(&mut app).map_err(|e| crate::runtime_error(format!("window: {e}")))?;
        app.surface = None;
        match app.error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    })
}

impl App<'_> {
    /// Where step `step` rests: just before its mark, since at the mark itself
    /// the next slide may already have begun.
    fn mark(&self, step: usize) -> f64 {
        let time = self.timeline.marks.get(step).map(|m| m.time).unwrap_or_else(|| self.timeline.duration());
        time - REST_BEFORE_MARK
    }

    fn last_step(&self) -> usize {
        self.timeline.marks.len().saturating_sub(1)
    }

    fn play_to(&mut self, target: f64) {
        self.playing = Some((Instant::now(), self.time, target));
    }

    fn forward(&mut self) {
        if let Some((_, _, target)) = self.playing.take() {
            self.time = target; // finish the step at once
        } else if self.time < self.mark(self.step) - 1e-9 {
            self.play_to(self.mark(self.step)); // after scrubbing back
        } else if self.step < self.last_step() {
            self.step += 1;
            self.play_to(self.mark(self.step));
        }
    }

    fn back(&mut self) {
        self.playing = None;
        self.step = self.step.saturating_sub(1);
        self.time = self.mark(self.step);
    }

    fn jump(&mut self, step: usize) {
        self.playing = None;
        self.step = step.min(self.last_step());
        self.time = self.mark(self.step);
    }

    fn replay(&mut self) {
        self.time = if self.step == 0 { 0.0 } else { self.mark(self.step - 1) };
        self.play_to(self.mark(self.step));
    }

    fn scrub(&mut self, points: f64) {
        self.playing = None;
        let elapsed = self.last_scroll.elapsed().as_secs_f32();
        self.last_scroll = Instant::now();
        let gain = window::scroll_gain(points as f32, elapsed, false, 0) as f64;
        self.time = (self.time + points * SCRUB_PER_POINT * gain).clamp(0.0, self.timeline.duration());
        // the step whose mark is the first at or after the time
        self.step = self.timeline.marks.iter().position(|m| m.time - REST_BEFORE_MARK >= self.time - 1e-9).unwrap_or(self.last_step());
    }

    fn redraw(&mut self) {
        if let Some((started, from, target)) = self.playing {
            self.time = (from + started.elapsed().as_secs_f64()).min(target);
            if self.time >= target {
                self.playing = None;
            }
        }
        if let Err(error) = self.lives.enter(self.py, self.timeline, self.time) {
            self.error = Some(error);
        }
        let live_moving = self.lives.animate(self.py);
        let Some(target) = self.surface.as_mut() else { return };
        let size = target.size();
        let overlay_settling = self.lives.draw(self.py, self.gpu, self.painter, self.timeline, self.time, size);
        if let Some(path) = self.options.screenshot.take() {
            let offscreen = Offscreen::with_format(&self.gpu.device, size, target.config.format);
            self.painter.draw(
                &self.gpu.device,
                &self.gpu.queue,
                self.timeline,
                self.images,
                self.time,
                offscreen.view(),
                target.config.format,
                size,
            );
            let pixels = offscreen.read(self.gpu);
            if let Err(error) = crate::window::save_png(&path, size, target.config.format, &pixels) {
                self.error = Some(crate::runtime_error(error));
            }
            self.surface = None;
            self.closing = true;
            return;
        }
        let Some(texture) = target.acquire(&self.gpu.device) else {
            target.window.request_redraw();
            return;
        };
        let view = texture.texture.create_view(&Default::default());
        self.painter.draw(
            &self.gpu.device,
            &self.gpu.queue,
            self.timeline,
            self.images,
            self.time,
            &view,
            target.config.format,
            size,
        );
        self.gpu.queue.present(texture);
        if self.playing.is_some() || live_moving || overlay_settling {
            target.window.request_redraw();
        }
    }

    /// A PNG of the window as it is now.
    fn snapshot(&mut self, path: &str) {
        if let Err(error) = self.lives.enter(self.py, self.timeline, self.time) {
            self.error = Some(error);
        }
        let Some(target) = self.surface.as_ref() else { return };
        let size = target.size();
        let format = target.config.format;
        self.lives.draw(self.py, self.gpu, self.painter, self.timeline, self.time, size);
        let offscreen = Offscreen::with_format(&self.gpu.device, size, format);
        self.painter.draw(&self.gpu.device, &self.gpu.queue, self.timeline, self.images, self.time, offscreen.view(), format, size);
        let pixels = offscreen.read(self.gpu);
        if let Err(error) = crate::window::save_png(path, size, format, &pixels) {
            self.error = Some(crate::runtime_error(error));
        }
    }

    /// Run the next scripted command when it is due (see Driver).
    fn drive(&mut self, event_loop: &ActiveEventLoop) {
        let Some(driver) = &mut self.driver else { return };
        if driver.next >= driver.commands.len() || self.surface.is_none() {
            return;
        }
        if Instant::now() < driver.due {
            event_loop.set_control_flow(ControlFlow::WaitUntil(driver.due));
            return;
        }
        let command = driver.commands[driver.next].clone();
        driver.next += 1;
        let words: Vec<&str> = command.split_whitespace().collect();
        let size = self.size();
        let number = |i: usize| words.get(i).and_then(|w| w.parse::<f64>().ok()).unwrap_or(0.0);
        match words.first().copied() {
            Some("move") => {
                self.lives.cursor_moved(self.py, self.timeline, self.time, size, (number(1), number(2)));
            }
            Some(b @ ("press" | "release" | "rpress" | "rrelease")) => {
                self.lives.button(self.py, self.timeline, self.time, size, b.ends_with("press"), b.starts_with('r'), false);
            }
            Some("wheel") => {
                let points = number(1);
                if !self.lives.scroll(self.py, self.timeline, self.time, size, points as f32, false) {
                    self.scrub(-points);
                }
            }
            Some("key") => {
                let key = match words.get(1).copied().unwrap_or("") {
                    "right" => Key::Named(NamedKey::ArrowRight),
                    "left" => Key::Named(NamedKey::ArrowLeft),
                    "space" => Key::Named(NamedKey::Space),
                    "home" => Key::Named(NamedKey::Home),
                    "end" => Key::Named(NamedKey::End),
                    "escape" => Key::Named(NamedKey::Escape),
                    c => Key::Character(c),
                };
                self.on_key(event_loop, key);
            }
            Some("wait") => {
                if let Some(driver) = &mut self.driver {
                    driver.due = Instant::now() + std::time::Duration::from_millis(number(1) as u64);
                }
            }
            Some("shot") => self.snapshot(words.get(1).copied().unwrap_or("shot.png")),
            Some("quit") => self.closing = true,
            _ => {}
        }
        self.request_redraw();
        event_loop.set_control_flow(ControlFlow::Poll);
    }

    fn key(&mut self, event_loop: &ActiveEventLoop, event: KeyEvent) {
        if event.state != ElementState::Pressed {
            return;
        }
        self.on_key(event_loop, event.logical_key.as_ref());
    }

    fn on_key(&mut self, event_loop: &ActiveEventLoop, key: Key<&str>) {
        let Some(target) = self.surface.as_ref() else { return };
        let window = target.window.clone();
        if let Key::Character(c) = key {
            let shift = self.modifiers.shift_key();
            if self.lives.key(self.py, self.timeline, self.time, &c.to_lowercase(), shift) {
                window.request_redraw();
                return;
            }
        }
        match key {
            Key::Named(NamedKey::ArrowRight | NamedKey::Space | NamedKey::PageDown | NamedKey::ArrowDown) => {
                self.forward()
            }
            Key::Named(NamedKey::ArrowLeft | NamedKey::PageUp | NamedKey::ArrowUp | NamedKey::Backspace) => {
                self.back()
            }
            Key::Named(NamedKey::Home) => self.jump(0),
            Key::Named(NamedKey::End) => self.jump(self.last_step()),
            Key::Named(NamedKey::Escape) => {
                if window.fullscreen().is_some() {
                    window.set_fullscreen(None);
                } else {
                    self.close(event_loop);
                    return;
                }
            }
            Key::Character(c) => match c.to_lowercase().as_str() {
                "f" => window.set_fullscreen(match window.fullscreen() {
                    Some(_) => None,
                    None => Some(Fullscreen::Borderless(None)),
                }),
                "r" => self.replay(),
                "q" => {
                    self.close(event_loop);
                    return;
                }
                _ => return,
            },
            _ => return,
        }
        window.request_redraw();
    }

    fn size(&self) -> (u32, u32) {
        self.surface.as_ref().map_or((1, 1), Surface::size)
    }

    fn request_redraw(&self) {
        if let Some(target) = &self.surface {
            target.window.request_redraw();
        }
    }

    fn close(&mut self, event_loop: &ActiveEventLoop) {
        self.surface = None;
        event_loop.exit();
    }
}

impl ApplicationHandler for App<'_> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.surface.is_some() {
            return;
        }
        let options = &self.options;
        match Surface::new(event_loop, self.gpu, &options.title, options.size, false, options.fullscreen) {
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
        match event {
            WindowEvent::CloseRequested => self.close(event_loop),
            WindowEvent::Resized(size) => {
                if let Some(target) = self.surface.as_mut() {
                    target.resize(&self.gpu.device, size);
                }
            }
            WindowEvent::RedrawRequested => self.redraw(),
            WindowEvent::KeyboardInput { event, .. } => self.key(event_loop, event),
            WindowEvent::ModifiersChanged(modifiers) => self.modifiers = modifiers.state(),
            WindowEvent::CursorMoved { position, .. } => {
                if self.lives.cursor_moved(self.py, self.timeline, self.time, self.size(), (position.x, position.y)) {
                    self.request_redraw();
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let pressed = state == ElementState::Pressed;
                let right = button == MouseButton::Right;
                let shift = self.modifiers.shift_key();
                if self.lives.button(self.py, self.timeline, self.time, self.size(), pressed, right, shift) {
                    self.request_redraw();
                }
            }
            WindowEvent::PinchGesture { delta, .. } => {
                if self.lives.pinch(self.py, self.timeline, self.time, self.size(), delta) {
                    self.request_redraw();
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let points = match delta {
                    MouseScrollDelta::LineDelta(_, y) => 20.0 * y as f64,
                    MouseScrollDelta::PixelDelta(p) => p.y,
                };
                let zoom = self.modifiers.control_key() || self.modifiers.super_key();
                let size = self.size();
                if !self.lives.scroll(self.py, self.timeline, self.time, size, points as f32, zoom) {
                    // scrolling down (negative) goes forward in time
                    self.scrub(-points);
                }
                self.request_redraw();
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        if self.closing {
            self.close(event_loop);
            return;
        }
        self.drive(event_loop);
        // let Ctrl-C in the terminal stop the window
        if let Err(error) = self.py.check_signals() {
            self.error = Some(error);
            self.close(event_loop);
            return;
        }
        if self.playing.is_some()
            && let Some(target) = &self.surface
        {
            target.window.request_redraw();
        }
    }
}
