//! SVG files as cubic Bézier paths with flat colors, as manim's SVGMobject
//! reads them: one path per drawn element, in drawing order.
//!
//! Rectangular clip paths (an axis-aligned rectangle, as plots clip their curves to the axes)
//! are applied: curves outside the rectangle are dropped, curves crossing it are cut at its
//! edge (strokes) or the shape is clipped as a polygon (fills). Other clip paths are ignored.

use std::sync::{Arc, OnceLock};

use usvg::tiny_skia_path::{PathSegment, Point};

/// A path in SVG user units (y down), already transformed to the page.
pub struct SvgPath {
    /// 4 control points per curve; a subpath starts where a curve does not
    /// begin at the end of the previous one
    pub points: Vec<[f32; 2]>,
    pub fill: [f32; 4],
    pub stroke: [f32; 4],
    pub stroke_width: f32,
    /// ids of the enclosing groups, outermost first
    pub groups: Vec<String>,
}

pub struct Svg {
    /// page width and height in user units
    pub size: [f32; 2],
    pub paths: Vec<SvgPath>,
}

fn fonts() -> Arc<usvg::fontdb::Database> {
    static FONTS: OnceLock<Arc<usvg::fontdb::Database>> = OnceLock::new();
    FONTS
        .get_or_init(|| {
            let mut fonts = usvg::fontdb::Database::new();
            fonts.load_system_fonts();
            Arc::new(fonts)
        })
        .clone()
}

pub fn parse(data: &[u8]) -> Result<Svg, String> {
    // loading the system fonts takes seconds: only for SVGs with text elements
    let has_text = data.windows(5).any(|w| w == b"<text");
    let options = if has_text { usvg::Options { fontdb: fonts(), ..Default::default() } } else { usvg::Options::default() };
    let tree = usvg::Tree::from_data(data, &options).map_err(|e| e.to_string())?;
    let mut paths = Vec::new();
    let mut groups = Vec::new();
    walk(tree.root(), 1.0, None, &mut groups, &mut paths);
    Ok(Svg { size: [tree.size().width(), tree.size().height()], paths })
}

/// Axis-aligned rectangle on the page: x0, y0, x1, y1.
type Rect = [f32; 4];

fn walk(group: &usvg::Group, opacity: f32, clip: Option<Rect>, groups: &mut Vec<String>, out: &mut Vec<SvgPath>) {
    let opacity = opacity * group.opacity().get();
    let clip = match group.clip_path().and_then(|c| clip_rect(group, c)) {
        Some(rect) => Some(intersect(clip, rect)),
        None => clip,
    };
    let named = !group.id().is_empty();
    if named {
        groups.push(group.id().to_string());
    }
    for node in group.children() {
        match node {
            usvg::Node::Group(child) => walk(child, opacity, clip, groups, out),
            usvg::Node::Text(text) => walk(text.flattened(), opacity, clip, groups, out),
            usvg::Node::Path(path) if path.is_visible() => {
                if let Some(path) = convert(path, opacity, clip, groups) {
                    out.push(path);
                }
            }
            _ => {}
        }
    }
    if named {
        groups.pop();
    }
}

fn intersect(a: Option<Rect>, b: Rect) -> Rect {
    match a {
        None => b,
        Some(a) => [a[0].max(b[0]), a[1].max(b[1]), a[2].min(b[2]), a[3].min(b[3])],
    }
}

fn bbox(points: &[[f32; 2]]) -> Rect {
    points.iter().fold([f32::MAX, f32::MAX, f32::MIN, f32::MIN], |r, p| {
        [r[0].min(p[0]), r[1].min(p[1]), r[2].max(p[0]), r[3].max(p[1])]
    })
}

/// The rectangle a clip path cuts out of the page for the group referencing it, or None when
/// the clip is not a single axis-aligned rectangle (such clips are ignored).
fn clip_rect(group: &usvg::Group, clip: &usvg::ClipPath) -> Option<Rect> {
    let mut paths = clip.root().children().iter().filter_map(|n| match n {
        usvg::Node::Path(p) => Some(p),
        _ => None,
    });
    let path = paths.next()?;
    if paths.next().is_some() {
        return None;
    }
    // the clip's subroot has its own transform, applied in the user space of the group
    let transform = group.abs_transform().pre_concat(clip.transform()).pre_concat(path.abs_transform());
    let mut points = Vec::new();
    for segment in path.data().segments() {
        match segment {
            PathSegment::MoveTo(p) | PathSegment::LineTo(p) => {
                let mut p = p;
                transform.map_point(&mut p);
                points.push([p.x, p.y]);
            }
            PathSegment::Close => {}
            _ => return None,
        }
    }
    if points.len() < 4 {
        return None;
    }
    let r = bbox(&points);
    let eps = 1e-3 * (r[2] - r[0]).max(r[3] - r[1]);
    if r[2] - r[0] <= eps || r[3] - r[1] <= eps {
        return None;
    }
    let on = |v: f32, a: f32, b: f32| (v - a).abs() <= eps || (v - b).abs() <= eps;
    if !points.iter().all(|p| on(p[0], r[0], r[2]) && on(p[1], r[1], r[3])) {
        return None;
    }
    match clip.clip_path() {
        Some(inner) => clip_rect(group, inner).map(|i| intersect(Some(r), i)),
        None => Some(r),
    }
}

fn inside(r: &Rect, clip: &Rect) -> bool {
    r[0] >= clip[0] && r[1] >= clip[1] && r[2] <= clip[2] && r[3] <= clip[3]
}

fn outside(r: &Rect, clip: &Rect) -> bool {
    r[2] < clip[0] || r[0] > clip[2] || r[3] < clip[1] || r[1] > clip[3]
}

fn lerp(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
}

fn push_line(points: &mut Vec<[f32; 2]>, a: [f32; 2], b: [f32; 2]) {
    points.extend([a, lerp(a, b, 1.0 / 3.0), lerp(a, b, 2.0 / 3.0), b]);
}

/// Points along a cubic curve, t = 0 ..= 1 in n steps (a straight curve: its two ends).
fn flatten(curve: &[[f32; 2]], n: usize) -> Vec<[f32; 2]> {
    let (a, c1, c2, b) = (curve[0], curve[1], curve[2], curve[3]);
    let straight = |p: [f32; 2], t: f32| (p[0] - lerp(a, b, t)[0]).abs() < 1e-4 && (p[1] - lerp(a, b, t)[1]).abs() < 1e-4;
    if straight(c1, 1.0 / 3.0) && straight(c2, 2.0 / 3.0) {
        return vec![a, b];
    }
    (0..=n)
        .map(|i| {
            let t = i as f32 / n as f32;
            let (s, u) = (1.0 - t, t);
            let w = [s * s * s, 3.0 * s * s * u, 3.0 * s * u * u, u * u * u];
            [
                w[0] * a[0] + w[1] * c1[0] + w[2] * c2[0] + w[3] * b[0],
                w[0] * a[1] + w[1] * c1[1] + w[2] * c2[1] + w[3] * b[1],
            ]
        })
        .collect()
}

/// Liang-Barsky: the part of segment a-b inside the rectangle.
fn clip_segment(a: [f32; 2], b: [f32; 2], clip: &Rect) -> Option<([f32; 2], [f32; 2])> {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let (mut t0, mut t1) = (0.0f32, 1.0f32);
    for (p, q) in [(-dx, a[0] - clip[0]), (dx, clip[2] - a[0]), (-dy, a[1] - clip[1]), (dy, clip[3] - a[1])] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }
    (t0 <= t1).then(|| (lerp(a, b, t0), lerp(a, b, t1)))
}

/// Sutherland-Hodgman: the polygon clipped to the rectangle.
fn clip_polygon(polygon: Vec<[f32; 2]>, clip: &Rect) -> Vec<[f32; 2]> {
    let mut out = polygon;
    for edge in 0..4 {
        let inside = |p: &[f32; 2]| match edge {
            0 => p[0] >= clip[0],
            1 => p[0] <= clip[2],
            2 => p[1] >= clip[1],
            _ => p[1] <= clip[3],
        };
        let cross = |a: [f32; 2], b: [f32; 2]| {
            let (i, v) = if edge < 2 { (0, clip[edge * 2]) } else { (1, clip[(edge - 2) * 2 + 1]) };
            lerp(a, b, (v - a[i]) / (b[i] - a[i]))
        };
        let input = std::mem::take(&mut out);
        if input.is_empty() {
            return out;
        }
        let mut prev = input[input.len() - 1];
        for p in input {
            if inside(&p) {
                if !inside(&prev) {
                    out.push(cross(prev, p));
                }
                out.push(p);
            } else if inside(&prev) {
                out.push(cross(prev, p));
            }
            prev = p;
        }
    }
    out
}

/// The curves (4 points each) clipped to the rectangle: whole curves outside are dropped and
/// crossing ones cut at the edge; a filled subpath is clipped as a polygon instead, so the fill
/// stays a closed shape (its stroke then follows the clip edge too).
fn clip_curves(points: Vec<[f32; 2]>, clip: &Rect, filled: bool) -> Vec<[f32; 2]> {
    let all = bbox(&points);
    if inside(&all, clip) {
        return points;
    }
    if outside(&all, clip) {
        return Vec::new();
    }
    const STEPS: usize = 16;
    let mut out = Vec::new();
    let curves: Vec<&[[f32; 2]]> = points.chunks_exact(4).collect();
    let mut i = 0;
    while i < curves.len() {
        // one subpath: curves that each start where the previous one ends
        let mut j = i + 1;
        while j < curves.len() && curves[j][0] == curves[j - 1][3] {
            j += 1;
        }
        let subpath = &curves[i..j];
        if filled {
            let mut polygon: Vec<[f32; 2]> = Vec::new();
            for curve in subpath {
                let flat = flatten(curve, STEPS);
                polygon.extend(&flat[if polygon.is_empty() { 0 } else { 1 }..]);
            }
            if polygon.len() > 1 && polygon[0] == polygon[polygon.len() - 1] {
                polygon.pop(); // closed explicitly: the polygon closes itself
            }
            let polygon = clip_polygon(polygon, clip);
            if polygon.len() >= 3 {
                for k in 0..polygon.len() {
                    push_line(&mut out, polygon[k], polygon[(k + 1) % polygon.len()]);
                }
            }
        } else {
            for curve in subpath {
                let r = bbox(curve);
                if inside(&r, clip) {
                    out.extend_from_slice(curve);
                } else if !outside(&r, clip) {
                    let flat = flatten(curve, STEPS);
                    for pair in flat.windows(2) {
                        if let Some((a, b)) = clip_segment(pair[0], pair[1], clip) {
                            push_line(&mut out, a, b);
                        }
                    }
                }
            }
        }
        i = j;
    }
    out
}

fn color(paint: &usvg::Paint, opacity: f32) -> [f32; 4] {
    let c = match paint {
        usvg::Paint::Color(c) => *c,
        // gradients and patterns: their first color
        usvg::Paint::LinearGradient(g) => g.stops().first().map(|s| s.color()).unwrap_or(usvg::Color::black()),
        usvg::Paint::RadialGradient(g) => g.stops().first().map(|s| s.color()).unwrap_or(usvg::Color::black()),
        usvg::Paint::Pattern(_) => usvg::Color::black(),
    };
    [c.red as f32 / 255.0, c.green as f32 / 255.0, c.blue as f32 / 255.0, opacity]
}

fn convert(path: &usvg::Path, opacity: f32, clip: Option<Rect>, groups: &[String]) -> Option<SvgPath> {
    let transform = path.abs_transform();
    let map = |p: Point| {
        let mut p = p;
        transform.map_point(&mut p);
        [p.x, p.y]
    };
    let (fill, stroke, stroke_width) = (
        path.fill().map(|f| color(f.paint(), opacity * f.opacity().get())).unwrap_or([0.0; 4]),
        path.stroke().map(|s| color(s.paint(), opacity * s.opacity().get())).unwrap_or([0.0; 4]),
        // widths scale with the transform (its geometric mean scale)
        path.stroke().map(|s| s.width().get() * (transform.sx * transform.sy - transform.kx * transform.ky).abs().sqrt()).unwrap_or(0.0),
    );
    let mut points = Vec::new();
    let mut start = [0.0f32; 2];
    let mut last = [0.0f32; 2];
    let line = |points: &mut Vec<[f32; 2]>, a: [f32; 2], b: [f32; 2]| {
        let lerp = |t: f32| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
        points.extend([a, lerp(1.0 / 3.0), lerp(2.0 / 3.0), b]);
    };
    for segment in path.data().segments() {
        match segment {
            PathSegment::MoveTo(p) => {
                start = map(p);
                last = start;
            }
            PathSegment::LineTo(p) => {
                let p = map(p);
                line(&mut points, last, p);
                last = p;
            }
            PathSegment::QuadTo(c, p) => {
                let (c, p) = (map(c), map(p));
                let c1 = [last[0] + 2.0 / 3.0 * (c[0] - last[0]), last[1] + 2.0 / 3.0 * (c[1] - last[1])];
                let c2 = [p[0] + 2.0 / 3.0 * (c[0] - p[0]), p[1] + 2.0 / 3.0 * (c[1] - p[1])];
                points.extend([last, c1, c2, p]);
                last = p;
            }
            PathSegment::CubicTo(c1, c2, p) => {
                let p = map(p);
                points.extend([last, map(c1), map(c2), p]);
                last = p;
            }
            PathSegment::Close => {
                if last != start {
                    line(&mut points, last, start);
                }
                last = start;
            }
        }
    }
    let points = match clip {
        Some(clip) => clip_curves(points, &clip, fill[3] > 0.0),
        None => points,
    };
    (!points.is_empty()).then(|| SvgPath { points, fill, stroke, stroke_width, groups: groups.to_vec() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn squares_and_groups() {
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="20" height="10">
            <g id="layer"><rect x="1" y="2" width="3" height="4" fill="#ff0000"/></g>
            <path d="M 0 0 Q 5 5 10 0" stroke="#000" stroke-width="2" fill="none"/>
        </svg>"##;
        let svg = parse(svg).unwrap();
        assert_eq!(svg.size, [20.0, 10.0]);
        assert_eq!(svg.paths.len(), 2);
        let rect = &svg.paths[0];
        assert_eq!(rect.groups, vec!["layer".to_string()]);
        assert_eq!(rect.points.len(), 16); // four lines, closed
        assert_eq!(rect.points[0], [1.0, 2.0]);
        assert_eq!(rect.fill, [1.0, 0.0, 0.0, 1.0]);
        let curve = &svg.paths[1];
        assert_eq!(curve.points.len(), 4);
        assert_eq!(curve.stroke_width, 2.0);
        assert_eq!(curve.fill[3], 0.0);
    }

    #[test]
    fn rectangular_clip_cuts_strokes_and_fills() {
        let svg = br##"<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
            <defs><clipPath id="c"><path d="M10 10 L50 10 L50 50 L10 50 Z"/></clipPath></defs>
            <g clip-path="url(#c)">
              <path d="M0 30 L100 30" stroke="#000" stroke-width="1" fill="none"/>
              <path d="M0 80 L100 80" stroke="#000" stroke-width="1" fill="none"/>
              <rect x="40" y="40" width="40" height="40" fill="#f00"/>
              <path d="M20 20 C 30 0, 40 0, 45 20" stroke="#000" fill="none"/>
            </g>
            <path d="M0 90 L100 90" stroke="#000" fill="none"/>
        </svg>"##;
        let svg = parse(svg).unwrap();
        let within = |p: &SvgPath| p.points.iter().all(|q| q[0] >= 9.99 && q[0] <= 50.01 && q[1] >= 9.99 && q[1] <= 50.01);
        assert_eq!(svg.paths.len(), 4); // the second line lies outside the clip
        let line = &svg.paths[0];
        assert_eq!(line.points.len(), 4);
        assert_eq!((line.points[0], line.points[3]), ([10.0, 30.0], [50.0, 30.0]));
        let rect = &svg.paths[1];
        assert!(within(rect) && rect.points.len() == 16, "clipped to the 40..50 square: {:?}", rect.points);
        let arc = &svg.paths[2];
        assert!(within(arc) && arc.points.len() > 4, "the arc leaving the top is cut: {:?}", arc.points.len());
        assert_eq!(svg.paths[3].points[0], [0.0, 90.0]); // unclipped
    }
}
