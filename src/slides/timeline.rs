//! A deck as one timeline: clips of shapes and images, each visible over a
//! span of time and animated over a window within it, and the marks where the
//! presentation pauses. Evaluating it at any time gives what is on screen, so
//! playing, stepping back and scrubbing are the same operation.
//!
//! The animations follow manim's: a clip's progress is its rate function of the
//! time within its window, and lagged groups are clips with shifted windows.

use std::borrow::Cow;

/// Fill and stroke colors (RGBA, as written: sRGB), the stroke width in
/// frame units, how strokes join and end, and whether each subpath is painted
/// (fill, then stroke) over the ones before it rather than all fills first.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Style {
    pub fill: [f32; 4],
    pub stroke: [f32; 4],
    pub width: f32,
    pub join: Join,
    pub cap: Cap,
    pub stacked: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Join {
    #[default]
    Miter,
    Round,
    Bevel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Cap {
    #[default]
    Butt,
    Round,
    Square,
}

impl Style {
    /// fill RGBA, stroke RGBA, width; optionally join (0 miter, 1 round,
    /// 2 bevel), cap (0 butt, 1 round, 2 square) and stacked (0 or 1)
    pub fn from_array(a: &[f32]) -> Style {
        let join = match a.get(9).map(|v| v.round() as i32) {
            Some(1) => Join::Round,
            Some(2) => Join::Bevel,
            _ => Join::Miter,
        };
        let cap = match a.get(10).map(|v| v.round() as i32) {
            Some(1) => Cap::Round,
            Some(2) => Cap::Square,
            _ => Cap::Butt,
        };
        let stacked = a.get(11).is_some_and(|v| *v > 0.5);
        Style { fill: [a[0], a[1], a[2], a[3]], stroke: [a[4], a[5], a[6], a[7]], width: a[8], join, cap, stacked }
    }

    fn lerp(&self, other: &Style, t: f32) -> Style {
        let mix = |a: [f32; 4], b: [f32; 4]| std::array::from_fn(|i| a[i] + t * (b[i] - a[i]));
        let (join, cap, stacked) =
            if t < 0.5 { (self.join, self.cap, self.stacked) } else { (other.join, other.cap, other.stacked) };
        Style {
            fill: mix(self.fill, other.fill),
            stroke: mix(self.stroke, other.stroke),
            width: self.width + t * (other.width - self.width),
            join,
            cap,
            stacked,
        }
    }
}

/// Cubic Bézier curves (4 control points each, in frame units, y up) and a style.
#[derive(Clone, Debug, PartialEq)]
pub struct Shape {
    pub points: Vec<[f32; 2]>,
    pub style: Style,
}

/// An image placed on the frame: center, width and height (frame units), opacity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placement {
    pub image: usize,
    pub rect: [f32; 4],
    pub opacity: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// shape `a` as it is
    Show,
    /// shape `a` turning into shape `b` (same number of points)
    Morph,
    /// the first part of shape `a`'s curves (manim's Create)
    Create,
    /// shape `a` drawn as an outline, then filled (manim's Write and DrawBorderThenFill)
    Write,
    /// placement `a` moving to placement `b`
    Image,
}

#[derive(Clone, Copy, Debug)]
pub struct Clip {
    pub kind: Kind,
    pub a: usize,
    pub b: usize,
    /// drawing order: higher on top
    pub z: f32,
    /// visible over [start, end)
    pub span: [f64; 2],
    /// the progress goes from 0 to 1 over this window
    pub window: [f64; 2],
    /// index of the rate function
    pub rate: usize,
    /// progress runs from 1 to 0 (Uncreate, Unwrite)
    pub reverse: bool,
    /// a fade: the clips of one layer are drawn together into a transparent
    /// layer that is blended over the slide with the fade's opacity, so they
    /// cover each other as they will and blend with what is already there
    pub layer: Option<usize>,
    /// the layer's opacity runs from 1 to 0 (FadeOut)
    pub fade_out: bool,
}

/// A rate function, sampled on [0, 1].
pub struct Rate(pub Vec<f32>);

impl Rate {
    pub fn linear() -> Rate {
        Rate(vec![0.0, 1.0])
    }

    pub fn at(&self, t: f32) -> f32 {
        let n = self.0.len() - 1;
        let x = t.clamp(0.0, 1.0) * n as f32;
        let i = (x.floor() as usize).min(n.saturating_sub(1));
        let f = x - i as f32;
        self.0[i] + f * (self.0[(i + 1).min(n)] - self.0[i])
    }
}

/// A live 3D view: viewer `viewer`'s picture over `rect` (center, width,
/// height in frame units) during `span`, on top of everything.
#[derive(Clone, Copy, Debug)]
pub struct LiveSpan {
    pub viewer: usize,
    pub rect: [f32; 4],
    pub span: [f64; 2],
}

/// One step of the presentation: it ends at `time`, where the player pauses.
#[derive(Clone, Debug)]
pub struct Mark {
    pub time: f64,
    pub slide: usize,
    pub notes: String,
}

#[derive(Default)]
pub struct Timeline {
    pub shapes: Vec<Shape>,
    pub placements: Vec<Placement>,
    pub clips: Vec<Clip>,
    pub lives: Vec<LiveSpan>,
    pub rates: Vec<Rate>,
    pub marks: Vec<Mark>,
    pub background: [f32; 4],
    /// frame width and height (frame units)
    pub frame: [f32; 2],
}

/// What to draw: shapes and images in drawing order. A shape in a layer
/// (layer id, opacity) is drawn into that layer rather than onto the slide.
pub enum Item<'a> {
    Shape { shape: Cow<'a, Shape>, id: Option<usize>, layer: Option<(usize, f32)> },
    Image(Placement),
    Live { viewer: usize, rect: [f32; 4] },
}

/// Stroke width of the outline in Write, in frame units (manim: 2 × 0.01).
const WRITE_OUTLINE: f32 = 0.02;

impl Timeline {
    pub fn duration(&self) -> f64 {
        let marks = self.marks.last().map(|m| m.time).unwrap_or(0.0);
        let clips = self.clips.iter().map(|c| c.window[1]).fold(0.0, f64::max);
        marks.max(clips)
    }

    /// Everything on screen at `time`, bottom to top. Unchanged shapes carry
    /// their id (their tessellation can be kept).
    pub fn at(&self, time: f64) -> Vec<Item<'_>> {
        let mut active: Vec<&Clip> =
            self.clips.iter().filter(|c| c.span[0] <= time && time < c.span[1]).collect();
        active.sort_by(|a, b| a.z.total_cmp(&b.z));
        let mut items: Vec<Item<'_>> = active
            .into_iter()
            .filter_map(|clip| {
                let mut item = self.evaluate(clip, time)?;
                if let (Some(layer), Item::Shape { layer: slot, .. }) = (clip.layer, &mut item) {
                    let alpha = self.progress(clip, time);
                    *slot = Some((layer, if clip.fade_out { 1.0 - alpha } else { alpha }));
                }
                Some(item)
            })
            .collect();
        items.extend(self.live_at(time).map(|i| Item::Live { viewer: self.lives[i].viewer, rect: self.lives[i].rect }));
        items
    }

    /// Indices of the live views shown at `time`.
    pub fn live_at(&self, time: f64) -> impl DoubleEndedIterator<Item = usize> + '_ {
        (0..self.lives.len()).filter(move |&i| self.lives[i].span[0] <= time && time < self.lives[i].span[1])
    }

    fn progress(&self, clip: &Clip, time: f64) -> f32 {
        let [w0, w1] = clip.window;
        let t = if w1 > w0 { ((time - w0) / (w1 - w0)).clamp(0.0, 1.0) as f32 } else if time >= w1 { 1.0 } else { 0.0 };
        let t = if clip.reverse { 1.0 - t } else { t };
        self.rates.get(clip.rate).map(|r| r.at(t)).unwrap_or(t)
    }

    fn evaluate(&self, clip: &Clip, time: f64) -> Option<Item<'_>> {
        let alpha = self.progress(clip, time);
        let shape = |i: usize| self.shapes.get(i);
        let item = |shape: Shape| Item::Shape { shape: Cow::Owned(shape), id: None, layer: None };
        let as_is = |id: usize| Some(Item::Shape { shape: Cow::Borrowed(shape(id)?), id: Some(id), layer: None });
        match clip.kind {
            Kind::Show => as_is(clip.a),
            Kind::Morph => {
                if alpha <= 0.0 || clip.a == clip.b {
                    return as_is(clip.a);
                }
                if alpha >= 1.0 {
                    return as_is(clip.b);
                }
                Some(item(morph(shape(clip.a)?, shape(clip.b)?, alpha)))
            }
            Kind::Create => {
                let a = shape(clip.a)?;
                if alpha >= 1.0 {
                    return as_is(clip.a);
                }
                Some(item(Shape { points: partial(&a.points, 0.0, alpha), style: a.style }))
            }
            Kind::Write => {
                let a = shape(clip.a)?;
                if alpha >= 1.0 {
                    return as_is(clip.a);
                }
                let outline = outline_style(&a.style);
                Some(item(if alpha < 0.5 {
                    Shape { points: partial(&a.points, 0.0, 2.0 * alpha), style: outline }
                } else {
                    Shape { points: a.points.clone(), style: outline.lerp(&a.style, 2.0 * alpha - 1.0) }
                }))
            }
            Kind::Image => {
                let (a, b) = (self.placements.get(clip.a)?, self.placements.get(clip.b)?);
                let mix = |x: f32, y: f32| x + alpha * (y - x);
                Some(Item::Image(Placement {
                    image: a.image,
                    rect: std::array::from_fn(|i| mix(a.rect[i], b.rect[i])),
                    opacity: mix(a.opacity, b.opacity),
                }))
            }
        }
    }
}

/// The outline Write draws first: no fill and a thin stroke in the stroke
/// color. (manim takes the color after hiding the fill, so it is the stroke
/// color even for shapes drawn without a stroke: white for SVG paths.)
fn outline_style(style: &Style) -> Style {
    Style { fill: [style.fill[0], style.fill[1], style.fill[2], 0.0], stroke: style.stroke, width: WRITE_OUTLINE, ..*style }
}

pub fn morph(a: &Shape, b: &Shape, t: f32) -> Shape {
    let points = if a.points.len() == b.points.len() {
        a.points.iter().zip(&b.points).map(|(p, q)| [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]).collect()
    } else if t < 0.5 {
        a.points.clone()
    } else {
        b.points.clone()
    };
    Shape { points, style: a.style.lerp(&b.style, t) }
}

/// The part of the curves between proportions `a` and `b`, counted in curves
/// as manim's pointwise_become_partial does.
pub fn partial(points: &[[f32; 2]], a: f32, b: f32) -> Vec<[f32; 2]> {
    let n = points.len() / 4;
    if n == 0 || b <= a {
        return vec![];
    }
    if a <= 0.0 && b >= 1.0 {
        return points.to_vec();
    }
    let locate = |t: f32| {
        let x = t.clamp(0.0, 1.0) * n as f32;
        let i = (x.floor() as usize).min(n - 1);
        (i, x - i as f32)
    };
    let (i0, r0) = locate(a);
    let (mut i1, mut r1) = locate(b);
    if r1 == 0.0 && i1 > i0 {
        // ends exactly at a curve's start: that curve adds nothing
        (i1, r1) = (i1 - 1, 1.0);
    }
    let curve = |i: usize| -> [[f32; 2]; 4] { [points[4 * i], points[4 * i + 1], points[4 * i + 2], points[4 * i + 3]] };
    let mut out = Vec::with_capacity(4 * (i1 - i0 + 1));
    if i0 == i1 {
        out.extend(sub_curve(curve(i0), r0, r1));
    } else {
        out.extend(sub_curve(curve(i0), r0, 1.0));
        for i in i0 + 1..i1 {
            out.extend(curve(i));
        }
        out.extend(sub_curve(curve(i1), 0.0, r1));
    }
    out
}

/// The cubic between parameters t0 and t1 of `c`.
fn sub_curve(c: [[f32; 2]; 4], t0: f32, t1: f32) -> [[f32; 2]; 4] {
    // split at t1, keep the first part; split that at t0 / t1, keep the second
    let first = split(c, t1).0;
    if t1 <= 0.0 {
        return [c[0]; 4];
    }
    split(first, t0 / t1).1
}

fn split(c: [[f32; 2]; 4], t: f32) -> ([[f32; 2]; 4], [[f32; 2]; 4]) {
    let lerp = |p: [f32; 2], q: [f32; 2]| [p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])];
    let (ab, bc, cd) = (lerp(c[0], c[1]), lerp(c[1], c[2]), lerp(c[2], c[3]));
    let (abc, bcd) = (lerp(ab, bc), lerp(bc, cd));
    let mid = lerp(abc, bcd);
    ([c[0], ab, abc, mid], [mid, bcd, cd, c[3]])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line(a: [f32; 2], b: [f32; 2]) -> Vec<[f32; 2]> {
        let lerp = |t: f32| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
        vec![a, lerp(1.0 / 3.0), lerp(2.0 / 3.0), b]
    }

    #[test]
    fn partial_counts_curves() {
        let mut points = line([0.0, 0.0], [1.0, 0.0]);
        points.extend(line([1.0, 0.0], [1.0, 1.0]));
        let half = partial(&points, 0.0, 0.5);
        assert_eq!(half.len(), 4);
        assert_eq!(*half.last().unwrap(), [1.0, 0.0]);
        let quarter = partial(&points, 0.0, 0.25);
        assert!((quarter[3][0] - 0.5).abs() < 1e-6);
        let three_quarters = partial(&points, 0.0, 0.75);
        assert_eq!(three_quarters.len(), 8);
        assert!((three_quarters[7][1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn clips_follow_their_windows() {
        let style = Style { fill: [1.0, 0.0, 0.0, 1.0], stroke: [0.0; 4], width: 0.0, join: Join::Miter, cap: Cap::Butt, stacked: false };
        let mut timeline = Timeline { rates: vec![Rate::linear()], ..Default::default() };
        timeline.shapes.push(Shape { points: line([0.0, 0.0], [1.0, 0.0]), style });
        timeline.shapes.push(Shape { points: line([0.0, 1.0], [1.0, 1.0]), style });
        timeline.clips.push(Clip {
            kind: Kind::Morph,
            a: 0,
            b: 1,
            z: 0.0,
            span: [1.0, f64::INFINITY],
            window: [1.0, 2.0],
            rate: 0,
            reverse: false,
            layer: None,
            fade_out: false,
        });
        assert!(timeline.at(0.5).is_empty());
        let y = |t| match &timeline.at(t)[0] {
            Item::Shape { shape, .. } => shape.points[0][1],
            _ => unreachable!(),
        };
        assert_eq!(y(1.0), 0.0);
        assert!((y(1.5) - 0.5).abs() < 1e-6);
        assert_eq!(y(5.0), 1.0);
    }

    #[test]
    fn write_outlines_then_fills() {
        let style = Style { fill: [1.0, 0.0, 0.0, 1.0], stroke: [0.0; 4], width: 0.0, join: Join::Miter, cap: Cap::Butt, stacked: false };
        let mut timeline = Timeline::default();
        timeline.shapes.push(Shape { points: line([0.0, 0.0], [1.0, 0.0]), style });
        timeline.clips.push(Clip {
            kind: Kind::Write,
            a: 0,
            b: 0,
            z: 0.0,
            span: [0.0, f64::INFINITY],
            window: [0.0, 1.0],
            rate: usize::MAX, // linear
            reverse: false,
            layer: None,
            fade_out: false,
        });
        let style_at = |t| match &timeline.at(t)[0] {
            Item::Shape { shape, .. } => shape.style,
            _ => unreachable!(),
        };
        assert_eq!(style_at(0.25).fill[3], 0.0);
        assert_eq!(style_at(0.25).stroke, style.stroke);
        assert_eq!(style_at(0.25).width, WRITE_OUTLINE);
        assert!((style_at(0.75).fill[3] - 0.5).abs() < 1e-6);
        assert_eq!(style_at(1.0), style);
    }

    #[test]
    fn fades_carry_their_layers_opacity() {
        let style = Style { fill: [1.0, 0.0, 0.0, 1.0], stroke: [0.0; 4], width: 0.0, join: Join::Miter, cap: Cap::Butt, stacked: false };
        let mut timeline = Timeline::default();
        timeline.shapes.push(Shape { points: line([0.0, 0.0], [1.0, 0.0]), style });
        for (fade_out, start) in [(false, 0.0), (true, 2.0)] {
            timeline.clips.push(Clip {
                kind: Kind::Morph,
                a: 0,
                b: 0,
                z: 0.0,
                span: [start, start + 2.0],
                window: [start, start + 1.0],
                rate: usize::MAX,
                reverse: false,
                layer: Some(7),
                fade_out,
            });
        }
        let layer_at = |t| match &timeline.at(t)[0] {
            Item::Shape { layer, id, .. } => (*layer, *id),
            _ => unreachable!(),
        };
        assert_eq!(layer_at(0.5), (Some((7, 0.5)), Some(0))); // unchanged shape: still cached by id
        assert_eq!(layer_at(1.5), (Some((7, 1.0)), Some(0)));
        assert_eq!(layer_at(2.25), (Some((7, 0.75)), Some(0)));
    }
}
