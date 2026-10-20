//! Slides without videos: a deck is a timeline of shapes and images that is
//! drawn live, at any moment, in a window or offscreen (see `timeline`).
//! Python builds the timeline (llandau.scene); this draws and presents it.

pub mod live;
pub mod paint;
pub mod player;
pub mod svg;
pub mod timeline;

use numpy::{PyArray1, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};

use crate::Viewer;
use crate::render::{Gpu, Offscreen};
use live::{Entry, Lives};
use crate::{runtime_error, value_error};
use paint::{Image, Painter};
use timeline::{Clip, Kind, LiveSpan, Mark, Placement, Rate, Shape, Style, Timeline};

/// A deck's timeline, drawn with its own GPU device.
#[pyclass(unsendable, module = "llandau._core")]
pub struct Slides {
    timeline: Timeline,
    images: Vec<Image>,
    lives: Lives,
    gpu: Option<Gpu>,
    painter: Option<Painter>,
    offscreen: Option<Offscreen>,
}

impl Slides {
    fn gpu(&mut self) -> PyResult<(&Gpu, &mut Painter)> {
        if self.gpu.is_none() {
            let gpu = Gpu::shared().map_err(runtime_error)?;
            self.painter = Some(Painter::new(&gpu.device));
            self.gpu = Some(gpu);
        }
        Ok((self.gpu.as_ref().unwrap(), self.painter.as_mut().unwrap()))
    }
}

#[pymethods]
impl Slides {
    /// frame: width and height of the frame (manim's 14.22 × 8 units);
    /// background: RGBA.
    #[new]
    #[pyo3(signature = (frame = (128.0 / 9.0, 8.0), background = (0.0, 0.0, 0.0, 1.0)))]
    fn new(frame: (f32, f32), background: (f32, f32, f32, f32)) -> Self {
        let (r, g, b, a) = background;
        Slides {
            timeline: Timeline { background: [r, g, b, a], frame: [frame.0, frame.1], ..Default::default() },
            images: Vec::new(),
            lives: Lives::default(),
            gpu: None,
            painter: None,
            offscreen: None,
        }
    }

    #[getter]
    fn background(&self) -> (f32, f32, f32, f32) {
        let [r, g, b, a] = self.timeline.background;
        (r, g, b, a)
    }

    #[setter]
    fn set_background(&mut self, color: (f32, f32, f32, f32)) {
        self.timeline.background = [color.0, color.1, color.2, color.3];
    }

    /// A shape: cubic Bézier control points (4 per curve, (n, 2), frame
    /// units, y up) and its style: fill RGBA, stroke RGBA, stroke width, and
    /// optionally join (0 miter, 1 round, 2 bevel) and cap (0 butt, 1 round,
    /// 2 square).
    /// Returns its id.
    fn add_shape(&mut self, points: PyReadonlyArray2<f32>, style: PyReadonlyArray1<f32>) -> PyResult<usize> {
        let style = style.as_slice()?;
        if style.len() != 9 && style.len() != 11 {
            return Err(value_error("style: fill RGBA, stroke RGBA, stroke width (and join, cap)"));
        }
        let points = control_points(&points)?;
        self.timeline.shapes.push(Shape { points, style: Style::from_array(style) });
        Ok(self.timeline.shapes.len() - 1)
    }

    /// An RGBA image ((height, width, 4) uint8). Returns its id.
    fn add_image(&mut self, pixels: PyReadonlyArray3<u8>) -> PyResult<usize> {
        let shape = pixels.shape();
        if shape[2] != 4 {
            return Err(value_error("pixels: (height, width, 4) RGBA"));
        }
        let data = pixels.as_slice().map(<[u8]>::to_vec).unwrap_or_else(|_| pixels.as_array().iter().copied().collect());
        self.images.push(Image { width: shape[1] as u32, height: shape[0] as u32, pixels: data });
        Ok(self.images.len() - 1)
    }

    /// An image file (PNG or JPEG). Returns (id, width, height) in pixels.
    fn add_image_file(&mut self, path: &str) -> PyResult<(usize, u32, u32)> {
        let image = image::open(path).map_err(|e| value_error(format!("{path}: {e}")))?.to_rgba8();
        let (width, height) = image.dimensions();
        self.images.push(Image { width, height, pixels: image.into_raw() });
        Ok((self.images.len() - 1, width, height))
    }

    /// Image `image` with its center, width and height in frame units, and
    /// opacity. Returns the placement's id.
    fn add_placement(&mut self, image: usize, rect: (f32, f32, f32, f32), opacity: f32) -> PyResult<usize> {
        if image >= self.images.len() {
            return Err(value_error(format!("no image {image}")));
        }
        self.timeline.placements.push(Placement { image, rect: [rect.0, rect.1, rect.2, rect.3], opacity });
        Ok(self.timeline.placements.len() - 1)
    }

    /// A rate function sampled at evenly spaced times in [0, 1]. Returns its id.
    fn add_rate(&mut self, samples: Vec<f32>) -> PyResult<usize> {
        if samples.len() < 2 {
            return Err(value_error("a rate function needs at least 2 samples"));
        }
        self.timeline.rates.push(Rate(samples));
        Ok(self.timeline.rates.len() - 1)
    }

    /// A clip: `kind` of shapes (or placements, for "image") `a` and `b`,
    /// drawn at depth `z` from `start` until `end` (inf: for good), its
    /// progress going from 0 to 1 between `window` times through rate `rate`
    /// (None: linear); `reverse` runs it backwards.
    ///   show: a as it is; morph: a turns into b (same number of points);
    ///   create: a's curves drawn from the start; write: a's outline drawn,
    ///   then filled; image: placement a moves and fades into placement b.
    #[pyo3(signature = (kind, a, b, z, start, end, window, rate = None, reverse = false))]
    #[allow(clippy::too_many_arguments)]
    fn add_clip(
        &mut self,
        kind: &str,
        a: usize,
        b: usize,
        z: f32,
        start: f64,
        end: f64,
        window: (f64, f64),
        rate: Option<usize>,
        reverse: bool,
    ) -> PyResult<()> {
        let kind = match kind {
            "show" => Kind::Show,
            "morph" => Kind::Morph,
            "create" => Kind::Create,
            "write" => Kind::Write,
            "image" => Kind::Image,
            _ => return Err(value_error(format!("unknown clip kind {kind:?}"))),
        };
        let count = if kind == Kind::Image { self.timeline.placements.len() } else { self.timeline.shapes.len() };
        if a >= count || b >= count {
            return Err(value_error(format!("no {} {}", if kind == Kind::Image { "placement" } else { "shape" }, a.max(b))));
        }
        let rate = match rate {
            Some(r) if r >= self.timeline.rates.len() => return Err(value_error(format!("no rate {r}"))),
            Some(r) => r,
            None => usize::MAX,
        };
        self.timeline.clips.push(Clip { kind, a, b, z, span: [start, end], window: [window.0, window.1], rate, reverse });
        Ok(())
    }

    /// A live 3D view: `viewer` drawn over `rect` (center, width, height in
    /// frame units) from `start` until `end`, on top of the slide. When its
    /// step starts, `enter(viewer)` is called and the camera moves to `camera`
    /// (a camera dict). Not drawn by render().
    #[pyo3(signature = (viewer, rect, start, end, enter = None, camera = None))]
    #[allow(clippy::too_many_arguments)]
    fn add_live(
        &mut self,
        viewer: Py<Viewer>,
        rect: (f32, f32, f32, f32),
        start: f64,
        end: f64,
        enter: Option<Py<PyAny>>,
        camera: Option<Py<PyDict>>,
    ) -> usize {
        let viewer = self.lives.viewer_index(viewer);
        self.timeline.lives.push(LiveSpan { viewer, rect: [rect.0, rect.1, rect.2, rect.3], span: [start, end] });
        self.lives.entries.push(Entry { enter, camera });
        self.timeline.lives.len() - 1
    }

    /// The end of a step: the presentation pauses at `time`. `slide` numbers
    /// the slide it belongs to; `notes` are the step's notes.
    #[pyo3(signature = (time, slide = 0, notes = String::new()))]
    fn add_mark(&mut self, time: f64, slide: usize, notes: String) {
        self.timeline.marks.push(Mark { time, slide, notes });
    }

    /// (time, slide, notes) of each step.
    #[getter]
    fn marks(&self) -> Vec<(f64, usize, String)> {
        self.timeline.marks.iter().map(|m| (m.time, m.slide, m.notes.clone())).collect()
    }

    #[getter]
    fn duration(&self) -> f64 {
        self.timeline.duration()
    }

    #[getter]
    fn num_shapes(&self) -> usize {
        self.timeline.shapes.len()
    }

    #[getter]
    fn num_clips(&self) -> usize {
        self.timeline.clips.len()
    }

    /// The deck at `time` as an RGBA image ((height, width, 4) uint8).
    fn render<'py>(&mut self, py: Python<'py>, time: f64, width: u32, height: u32) -> PyResult<Bound<'py, PyAny>> {
        const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
        let size = (width.max(1), height.max(1));
        self.gpu()?;
        let gpu = self.gpu.as_ref().unwrap();
        if !matches!(&self.offscreen, Some(o) if o.size() == size) {
            self.offscreen = Some(Offscreen::with_format(&gpu.device, size, FORMAT));
        }
        let target = self.offscreen.as_ref().unwrap();
        let painter = self.painter.as_mut().unwrap();
        painter.draw(&gpu.device, &gpu.queue, &self.timeline, &self.images, time, target.view(), FORMAT, size);
        let pixels = target.read(gpu);
        Ok(PyArray1::from_vec(py, pixels).reshape([size.1 as usize, size.0 as usize, 4])?.into_any())
    }

    /// Present the deck in a window, from step `start`; returns when it closes.
    /// screenshot: draw the end of that step into this PNG file and close.
    #[pyo3(signature = (start = 0, fullscreen = false, title = None, size = (1280.0, 720.0), screenshot = None))]
    fn play(
        &mut self,
        py: Python<'_>,
        start: usize,
        fullscreen: bool,
        title: Option<String>,
        size: (f64, f64),
        screenshot: Option<String>,
    ) -> PyResult<()> {
        self.gpu()?;
        let title = title.unwrap_or_else(|| "slides".into());
        let options = player::Options { title, size, fullscreen, start, screenshot };
        let (gpu, painter) = (self.gpu.as_ref().unwrap(), self.painter.as_mut().unwrap());
        player::play(py, gpu, painter, &self.timeline, &self.images, &mut self.lives, options)
    }
}

fn control_points(points: &PyReadonlyArray2<f32>) -> PyResult<Vec<[f32; 2]>> {
    let shape = points.shape();
    if shape[1] != 2 || shape[0] % 4 != 0 {
        return Err(value_error("points: (4 × curves, 2)"));
    }
    let array = points.as_array();
    Ok(array.rows().into_iter().map(|r| [r[0], r[1]]).collect())
}

/// Parse an SVG (its bytes) into {"size": (width, height), "paths": [...]},
/// each path a dict of points ((4 × curves, 2), user units, y down), fill
/// and stroke RGBA, stroke width and the ids of its enclosing groups.
#[pyfunction]
pub fn parse_svg<'py>(py: Python<'py>, data: &Bound<'py, PyBytes>) -> PyResult<Bound<'py, PyDict>> {
    let svg = svg::parse(data.as_bytes()).map_err(value_error)?;
    let paths = PyList::empty(py);
    for path in svg.paths {
        let entry = PyDict::new(py);
        let n = path.points.len();
        let flat: Vec<f32> = path.points.into_iter().flatten().collect();
        entry.set_item("points", PyArray1::from_vec(py, flat).reshape([n, 2])?)?;
        entry.set_item("fill", path.fill.to_vec())?;
        entry.set_item("stroke", path.stroke.to_vec())?;
        entry.set_item("stroke_width", path.stroke_width)?;
        entry.set_item("groups", path.groups)?;
        paths.append(entry)?;
    }
    let result = PyDict::new(py);
    result.set_item("size", (svg.size[0], svg.size[1]))?;
    result.set_item("paths", paths)?;
    Ok(result)
}

/// Width and height of an image file in pixels.
#[pyfunction]
pub fn image_size(path: &str) -> PyResult<(u32, u32)> {
    image::image_dimensions(path).map_err(|e| value_error(format!("{path}: {e}")))
}
