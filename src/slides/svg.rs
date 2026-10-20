//! SVG files as cubic Bézier paths with flat colors, as manim's SVGMobject
//! reads them: one path per drawn element, in drawing order.

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
    let options = usvg::Options { fontdb: fonts(), ..Default::default() };
    let tree = usvg::Tree::from_data(data, &options).map_err(|e| e.to_string())?;
    let mut paths = Vec::new();
    let mut groups = Vec::new();
    walk(tree.root(), 1.0, &mut groups, &mut paths);
    Ok(Svg { size: [tree.size().width(), tree.size().height()], paths })
}

fn walk(group: &usvg::Group, opacity: f32, groups: &mut Vec<String>, out: &mut Vec<SvgPath>) {
    let opacity = opacity * group.opacity().get();
    let named = !group.id().is_empty();
    if named {
        groups.push(group.id().to_string());
    }
    for node in group.children() {
        match node {
            usvg::Node::Group(child) => walk(child, opacity, groups, out),
            usvg::Node::Text(text) => walk(text.flattened(), opacity, groups, out),
            usvg::Node::Path(path) if path.is_visible() => {
                if let Some(path) = convert(path, opacity, groups) {
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

fn convert(path: &usvg::Path, opacity: f32, groups: &[String]) -> Option<SvgPath> {
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
}
