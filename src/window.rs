//! The explorer: a native window (winit) with the 3D view and a panel of
//! controls and plots (egui), drawn with the viewer's own GPU device.
//!
//! Keys: space plays, , and . step frames, D toggles the discrete states,
//! 0 resets the camera, Escape closes.

use std::cell::RefCell;
use std::sync::Arc;
use std::time::{Duration, Instant};

use egui::{Color32, Key, PointerButton, Sense};
use egui_plot::{Line, Plot, PlotPoints, Points, VLine};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::platform::run_on_demand::EventLoopExtRunOnDemand;
use winit::window::{Window, WindowId};

use crate::render::{Gpu, Offscreen};

const PANEL_WIDTH: f32 = 300.0;
const PLAY_INTERVAL: Duration = Duration::from_millis(33);
/// Wheel or trackpad points per frame when scrubbing.
const SCROLL_PER_FRAME: f32 = 12.0;

thread_local! {
    // winit allows one event loop per process; each explore() reuses it.
    static EVENT_LOOP: RefCell<Option<EventLoop<()>>> = const { RefCell::new(None) };
}

/// A scalar curve: (name, xs, values).
pub type Curve = (String, Vec<f64>, Vec<f64>);

pub struct Options {
    pub title: String,
    pub size: (f64, f64),
    pub plots: Vec<Curve>,
    /// parameter of each path frame (for the plot cursors)
    pub t: Vec<f64>,
    pub info: Option<String>,
    /// label -> function of the camera dict returning text to copy
    pub buttons: Vec<(String, Py<PyAny>)>,
    /// render one frame of the window to this PNG path and close
    pub screenshot: Option<String>,
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
    initial_scales: Vec<(String, f32)>,
    status: String,
    frames_drawn: u32,
    close_requested: bool,
    error: Option<PyErr>,
}

pub fn explore<V: Explorable>(viewer: &mut V, py: Python<'_>, options: Options) -> PyResult<()> {
    EVENT_LOOP.with(|cell| {
        let mut slot = cell.borrow_mut();
        if slot.is_none() {
            let event_loop = EventLoop::new().map_err(|e| crate::runtime_error(format!("cannot open a window: {e}")))?;
            *slot = Some(event_loop);
        }
        let event_loop = slot.as_mut().unwrap();
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
            initial_scales,
            status: String::new(),
            frames_drawn: 0,
            close_requested: false,
            error: None,
        };
        event_loop.run_app_on_demand(&mut app).map_err(|e| crate::runtime_error(format!("window: {e}")))?;
        app.surface = None; // close the window before returning
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
        // the 3D view shows through the central panel, which only takes input
        egui::CentralPanel::default().frame(egui::Frame::NONE).show(ui, |ui| {
            let rect = ui.max_rect();
            let response = ui.allocate_rect(rect, Sense::click_and_drag());
            self.viewport = rect;
            self.pointer(ui, &response);
        });
    }

    fn keys(&mut self, ui: &egui::Ui) {
        if ui.ctx().egui_wants_keyboard_input() {
            return;
        }
        let pressed = |key| ui.input(|i| i.key_pressed(key));
        let controls = self.viewer.controls();
        if pressed(Key::Space) {
            self.playing = !self.playing;
        }
        for (key, step) in [(Key::Comma, -1i64), (Key::Period, 1)] {
            if pressed(key) {
                let frame = (controls.frame() as i64 + step).clamp(0, controls.num_frames() as i64 - 1);
                controls.set_frame(frame.max(0) as usize);
            }
        }
        if pressed(Key::D) && controls.num_discrete_frames() > 0 {
            let on = !controls.scrub_discrete();
            controls.set_scrub_discrete(on);
        }
        if pressed(Key::Num0) {
            controls.reset_camera();
        }
        if pressed(Key::Escape) {
            self.close_requested = true;
        }
    }

    fn pointer(&mut self, ui: &egui::Ui, response: &egui::Response) {
        let ppp = ui.ctx().pixels_per_point();
        let height = response.rect.height() * ppp;
        let controls = self.viewer.controls();
        let delta = response.drag_delta() * ppp;
        let shift = ui.input(|i| i.modifiers.shift);
        if response.dragged_by(PointerButton::Primary) && !shift {
            controls.orbit(delta.x, delta.y);
        } else if response.dragged_by(PointerButton::Secondary) || (response.dragged_by(PointerButton::Primary) && shift) {
            controls.pan(delta.x, delta.y, height);
        }
        if response.double_clicked() {
            controls.reset_camera();
        }
        if response.hovered() {
            let (scroll, zoom) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            if zoom != 1.0 {
                controls.zoom(1.0 / zoom);
            }
            self.scroll -= scroll / SCROLL_PER_FRAME;
            let steps = self.scroll.trunc();
            if steps != 0.0 {
                self.scroll -= steps;
                let frame = (controls.frame() as i64 + steps as i64).clamp(0, controls.num_frames().max(1) as i64 - 1);
                controls.set_frame(frame as usize);
            }
        }
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
        });
        ui.horizontal(|ui| {
            if ui.button(if self.playing { "Pause" } else { "Play" }).clicked() {
                self.playing = !self.playing;
            }
            if controls.num_discrete_frames() > 0 {
                let mut on = controls.scrub_discrete();
                if ui.checkbox(&mut on, "scrub discrete").changed() {
                    controls.set_scrub_discrete(on);
                }
            }
        });
        ui.separator();

        // appearance
        egui::Grid::new("appearance").num_columns(2).spacing([12.0, 6.0]).show(ui, |ui| {
            choice(ui, "theme", &["light", "dark"], controls.theme(), |name| controls.set_theme(name));
            ui.end_row();
            if controls.has_roles() {
                choice(ui, "style", &["atoms", "protein"], controls.style(), |s| controls.set_style(s));
                ui.end_row();
                choice(ui, "side chains", &["sticks", "spheres"], controls.sidechains(), |s| controls.set_sidechains(s));
                ui.end_row();
            }
            let colorings = controls.coloring_names();
            if !colorings.is_empty() {
                let names: Vec<&str> = std::iter::once("element").chain(colorings.iter().map(String::as_str)).collect();
                let current = controls.coloring().unwrap_or_else(|| "element".into());
                let mut picked = None;
                choice(ui, "color", &names, &current, |name| picked = Some(name.to_string()));
                if let Some(name) = picked {
                    controls.set_coloring(Some(name).filter(|n| n != "element"));
                }
                ui.end_row();
            }
        });
        ui.separator();

        // layers
        for (layer, visible) in controls.layers() {
            let mut on = visible;
            if ui.checkbox(&mut on, layer.as_str()).changed() {
                controls.set_visible(&layer, on);
            }
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
            choice(ui, "surface color", &["uniform", "atoms"], if by_atom { "atoms" } else { "uniform" }, |c| {
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
            for (index, (label, _)) in self.options.buttons.iter().enumerate() {
                if ui.button(label).clicked() {
                    copy = Some(Some(index));
                }
            }
        });
        if let Some(which) = copy {
            match self.copy_text(which) {
                Ok(text) => {
                    println!("{text}");
                    ui.ctx().copy_text(text);
                    self.status = "copied".into();
                }
                Err(error) => self.status = format!("{error}"),
            }
        }
        if !self.status.is_empty() {
            ui.small(&self.status);
        }

        // plots
        let x = self.current_t();
        for (name, xs, ys) in &self.options.plots {
            let value = interpolate(xs, ys, x);
            ui.add_space(6.0);
            ui.label(format!("{name}: {value:.4}"));
            let points: PlotPoints = xs.iter().zip(ys).filter(|(_, y)| y.is_finite()).map(|(x, y)| [*x, *y]).collect();
            Plot::new(name.as_str())
                .height(90.0)
                .width(ui.available_width())
                .allow_zoom(false)
                .allow_scroll(false)
                .allow_drag(false)
                .allow_boxed_zoom(false)
                .show_x(false)
                .show_y(false)
                .show(ui, |plot| {
                    plot.line(Line::new(name.as_str(), points));
                    plot.vline(VLine::new("t", x).color(Color32::GRAY));
                    if value.is_finite() {
                        plot.points(Points::new("now", vec![[x, value]]).radius(4.0).color(Color32::from_rgb(220, 50, 50)));
                    }
                });
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

    /// The path parameter of the current frame (discrete states spread evenly over the path).
    fn current_t(&mut self) -> f64 {
        let t = &self.options.t;
        let controls = self.viewer.controls();
        if t.is_empty() {
            return controls.frame() as f64;
        }
        if controls.scrub_discrete() {
            let fraction = controls.frame() as f64 / (controls.num_frames().max(2) - 1) as f64;
            return t[0] + fraction * (t[t.len() - 1] - t[0]);
        }
        t[controls.frame().min(t.len() - 1)]
    }
}

fn choice(ui: &mut egui::Ui, label: &str, options: &[&str], current: &str, mut apply: impl FnMut(&str)) {
    ui.label(label);
    egui::ComboBox::from_id_salt(label).selected_text(current).show_ui(ui, |ui| {
        for option in options {
            if ui.selectable_label(*option == current, *option).clicked() && *option != current {
                apply(option);
            }
        }
    });
}

fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    match xs.iter().position(|&v| v >= x) {
        None => *ys.last().unwrap_or(&f64::NAN),
        Some(0) => ys[0],
        Some(i) => {
            let w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]).max(1e-12);
            ys[i - 1] + w * (ys[i] - ys[i - 1])
        }
    }
}

fn save_png(path: &str, size: (u32, u32), format: wgpu::TextureFormat, pixels: &[u8]) -> Result<(), String> {
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
