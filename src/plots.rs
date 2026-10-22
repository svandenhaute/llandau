//! Scalars along a trajectory, plotted with a cursor at the current frame: small
//! in the explorer's panel, large (axes readable from the back of a room) under
//! the explorer's 3D view or over a deck's live view. Also the key help both show.

use egui::{Color32, RichText, TextStyle};
use egui_plot::{Line, Plot, PlotPoints, Points, VLine};

/// A scalar curve: (name, xs, values).
pub type Curve = (String, Vec<f64>, Vec<f64>);

const CURSOR: Color32 = Color32::from_rgb(220, 50, 50);
/// The large plots' share of a view's height, and their text size (points).
pub const PLOTS_HEIGHT: f32 = 0.4;
const PLOTS_TEXT: f32 = 18.0;

/// A viewer's plots: curves on the scale of `t`, the parameter of each path frame.
#[derive(Clone, Debug, Default)]
pub struct Plots {
    pub curves: Vec<Curve>,
    pub t: Vec<f64>,
}

impl Plots {
    pub fn is_empty(&self) -> bool {
        self.curves.is_empty()
    }

    /// The path parameter at a (fractional) path frame (see Scene::path_frame), interpolated
    /// between the frames' parameters; the frame itself without `t`.
    pub fn x(&self, path_frame: f32) -> f64 {
        let t = &self.t;
        if t.is_empty() {
            return path_frame as f64;
        }
        let last = (t.len() - 1) as f64;
        let frame = (path_frame as f64).clamp(0.0, last);
        let lo = frame.floor() as usize;
        let hi = (lo + 1).min(t.len() - 1);
        t[lo] + (frame - lo as f64) * (t[hi] - t[lo])
    }

    /// The panel's plots: each a value and a small plot without axes.
    pub fn small(&self, ui: &mut egui::Ui, x: f64) {
        for (name, xs, ys) in &self.curves {
            let value = interpolate(xs, ys, x);
            ui.add_space(6.0);
            ui.label(format!("{name}: {value:.4}"));
            plot(name, xs)
                .height(90.0)
                .width(ui.available_width())
                .show_x(false)
                .show_y(false)
                .show(ui, |plot| curve(plot, name, xs, ys, x, value));
        }
    }

    /// The large plots under a 3D view, as a bottom panel taking PLOTS_HEIGHT of `ui`.
    pub fn bottom_panel(&self, ui: &mut egui::Ui, x: f64) {
        if self.is_empty() {
            return;
        }
        egui::Panel::bottom("plots")
            .exact_size((PLOTS_HEIGHT * ui.available_height()).round())
            .resizable(false)
            .show(ui, |ui| self.large(ui, x, PLOTS_TEXT));
    }

    /// Plots side by side filling `ui`, with axes and their numbers at `text`
    /// points, each titled with its name and value.
    fn large(&self, ui: &mut egui::Ui, x: f64, text: f32) {
        let style = ui.style_mut();
        for (kind, size) in [(TextStyle::Body, text), (TextStyle::Small, 0.8 * text), (TextStyle::Heading, 1.2 * text)] {
            if let Some(font) = style.text_styles.get_mut(&kind) {
                font.size = size;
            }
        }
        let n = self.curves.len().max(1);
        ui.columns(n, |columns| {
            for ((name, xs, ys), ui) in self.curves.iter().zip(columns) {
                let value = interpolate(xs, ys, x);
                ui.label(RichText::new(format!("{name}  {value:.4}")).heading());
                plot(name, xs)
                    .height(ui.available_height())
                    .width(ui.available_width())
                    .show(ui, |plot| curve(plot, name, xs, ys, x, value));
            }
        });
    }
}

/// A plot that only shows: no zooming or dragging.
fn plot<'a>(name: &str, xs: &[f64]) -> Plot<'a> {
    let (lo, hi) = (xs.first().copied().unwrap_or(0.0), xs.last().copied().unwrap_or(1.0));
    Plot::new(name.to_string())
        .allow_zoom(false)
        .allow_scroll(false)
        .allow_drag(false)
        .allow_boxed_zoom(false)
        .allow_double_click_reset(false)
        .show_crosshair(false)
        .include_x(lo.min(hi))
        .include_x(lo.max(hi))
}

fn curve(plot: &mut egui_plot::PlotUi<'_>, name: &str, xs: &[f64], ys: &[f64], x: f64, value: f64) {
    let points: PlotPoints = xs.iter().zip(ys).filter(|(_, y)| y.is_finite()).map(|(x, y)| [*x, *y]).collect();
    plot.line(Line::new(name.to_string(), points).width(2.0));
    plot.vline(VLine::new("t", x).color(Color32::GRAY));
    if value.is_finite() {
        plot.points(Points::new("now", vec![[x, value]]).radius(5.0).color(CURSOR));
    }
}

/// The value at `x`, linear between the samples (xs ascending).
pub fn interpolate(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    match xs.iter().position(|&v| v >= x) {
        None => *ys.last().unwrap_or(&f64::NAN),
        Some(0) => ys[0],
        Some(i) => {
            let w = (x - xs[i - 1]) / (xs[i] - xs[i - 1]).max(1e-12);
            ys[i - 1] + w * (ys[i] - ys[i - 1])
        }
    }
}

/// The keys, as a centered black card with white text over `ui` (in either theme):
/// (key, what it does) rows, "" keys head groups.
pub fn help(ui: &egui::Ui, keys: &[(&str, &str)], text: f32) {
    // smaller text when the rows would not fit the view: a row is a line (about 1.4 times
    // the text size in egui's font) plus 0.35 of spacing, within the card's margins
    let text = text.min((ui.max_rect().height() - 2.0 * text) / (1.95 * keys.len() as f32)).max(6.0);
    let area = egui::Area::new(egui::Id::unique("llandau keys")).anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0]).fade_in(false);
    area.show(ui.ctx(), |ui| {
        ui.style_mut().visuals = egui::Visuals::dark();
        ui.style_mut().visuals.override_text_color = Some(Color32::WHITE);
        egui::Frame::popup(ui.style()).fill(Color32::BLACK).inner_margin(text).show(ui, |ui| {
            egui::Grid::new("keys").num_columns(2).spacing([1.5 * text, 0.35 * text]).show(ui, |ui| {
                for (key, action) in keys {
                    if key.is_empty() {
                        ui.label(RichText::new(*action).size(text).strong());
                        ui.end_row();
                        continue;
                    }
                    ui.label(RichText::new(*key).size(text).monospace().strong());
                    ui.label(RichText::new(*action).size(text));
                    ui.end_row();
                }
            });
        });
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cursor_follows_the_frame() {
        let plots = Plots { curves: vec![], t: vec![0.0, 0.5, 2.0] };
        assert_eq!(plots.x(1.0), 0.5);
        assert_eq!(plots.x(1.5), 1.25); // a discrete state between two path frames
        assert_eq!(plots.x(9.0), 2.0); // past the last frame
        assert_eq!(Plots::default().x(7.0), 7.0);
        assert_eq!(interpolate(&[0.0, 1.0], &[2.0, 4.0], 0.25), 2.5);
    }

    /// egui's own fonts lack many symbols (⌘, ⏺; arrows only in monospace): every
    /// key the help shows must be drawable.
    #[test]
    fn every_key_has_its_glyphs() {
        let ctx = egui::Context::default();
        ctx.run_ui(Default::default(), |_| {}).textures_delta.clear();
        let (text, mono) = (egui::FontId::proportional(14.0), egui::FontId::monospace(14.0));
        for (key, action) in crate::window::KEYS.iter().chain(crate::slides::live::KEYS) {
            assert!(ctx.fonts_mut(|f| f.has_glyphs(&mono, key)), "key {key:?}");
            assert!(ctx.fonts_mut(|f| f.has_glyphs(&text, action)), "{action:?}");
        }
        for hint in crate::window::HINTS {
            assert!(ctx.fonts_mut(|f| f.has_glyphs(&mono, hint)), "hint {hint:?}");
        }
    }

    /// The help card is measured in its first frame, unseen: egui must ask for the
    /// next one (the deck's player only redraws when asked), which shows it.
    #[test]
    fn the_help_asks_for_the_frame_that_shows_it() {
        let ctx = egui::Context::default();
        let frame = || {
            let mut output = ctx.run_ui(Default::default(), |ui| help(ui, crate::window::KEYS, 15.0));
            output.textures_delta.clear();
            let again = output.viewport_output.values().any(|v| v.repaint_delay.as_millis() < 100);
            let jobs = ctx.tessellate(output.shapes, output.pixels_per_point);
            let vertices = jobs.iter().map(|p| match &p.primitive {
                egui::epaint::Primitive::Mesh(mesh) => mesh.vertices.len(),
                _ => 0,
            });
            (again, vertices.sum::<usize>())
        };
        let (again, _) = frame();
        assert!(again);
        let (_, vertices) = frame();
        assert!(vertices > 1000, "the card's text: {vertices} vertices");
    }
}
