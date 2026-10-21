//! The view's state: everything the explorer's panel can change, as one value
//! (`Viewer.state` in Python, a dict). Recordings sample it at every change
//! while the explorer records, and replay it: the camera and the sliders move
//! between samples, the rest jumps. A deck plays a recording on a step's clock
//! (`live(play=...)`), the explorer renders one to a video.

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::Viewer;
use crate::camera::Camera;
use crate::scene::SurfaceColors;

/// What a recording keeps of the view at one moment.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    pub camera: Camera,
    frame: usize,
    scrub_discrete: bool,
    theme: &'static str,
    shading: &'static str,
    style: &'static str,
    sidechains: &'static str,
    coloring: Option<String>,
    layers: Vec<(String, bool)>,
    color_gamma: f32,
    fog: f32,
    occlusion: f32,
    radius_scale: f32,
    surface_opacity: f32,
    surface_colors_by_atom: bool,
    vector_scales: Vec<(String, f32)>,
    plots: bool,
}

/// The keys of the state dict, as `Viewer.state` gives and takes them.
pub const KEYS: &[&str] = &[
    "camera", "frame", "scrub_discrete", "theme", "shading", "style", "sidechains", "coloring", "layers",
    "color_gamma", "fog", "occlusion", "radius_scale", "surface_opacity", "surface_colors", "vector_scales", "plots",
];

fn name(value: &str) -> &'static str {
    // the viewer's names are static; a dict's are matched to them
    for s in ["light", "dark", "lit", "flat", "atoms", "protein", "sticks", "spheres"] {
        if s == value {
            return s;
        }
    }
    ""
}

impl State {
    pub fn capture(v: &Viewer) -> State {
        State {
            camera: v.camera,
            frame: v.frame,
            scrub_discrete: v.scene.scrub_discrete,
            theme: v.theme(),
            shading: v.shading(),
            style: v.style(),
            sidechains: v.sidechains(),
            coloring: v.scene.coloring.clone(),
            layers: v.layers(),
            color_gamma: v.color_gamma(),
            fog: v.fog(),
            occlusion: v.scene.occlusion,
            radius_scale: v.scene.radius_scale,
            surface_opacity: v.scene.surface.opacity,
            surface_colors_by_atom: v.scene.surface.colors == SurfaceColors::Atoms,
            vector_scales: v.vector_scales(),
            plots: v.show_plots,
        }
    }

    /// The state as a dict (camera: a camera dict; layers and vector_scales: dicts by name;
    /// coloring: a name or None; surface_colors: "atoms" or "uniform").
    pub fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("camera", self.camera.to_dict(py)?)?;
        d.set_item("frame", self.frame)?;
        d.set_item("scrub_discrete", self.scrub_discrete)?;
        d.set_item("theme", self.theme)?;
        d.set_item("shading", self.shading)?;
        d.set_item("style", self.style)?;
        d.set_item("sidechains", self.sidechains)?;
        d.set_item("coloring", self.coloring.clone())?;
        let layers = PyDict::new(py);
        for (name, visible) in &self.layers {
            layers.set_item(name, *visible)?;
        }
        d.set_item("layers", layers)?;
        d.set_item("color_gamma", self.color_gamma)?;
        d.set_item("fog", self.fog)?;
        d.set_item("occlusion", self.occlusion)?;
        d.set_item("radius_scale", self.radius_scale)?;
        d.set_item("surface_opacity", self.surface_opacity)?;
        d.set_item("surface_colors", if self.surface_colors_by_atom { "atoms" } else { "uniform" })?;
        let scales = PyDict::new(py);
        for (name, scale) in &self.vector_scales {
            scales.set_item(name, *scale)?;
        }
        d.set_item("vector_scales", scales)?;
        d.set_item("plots", self.plots)?;
        Ok(d)
    }

    /// `base` with the entries of a (partial) state dict applied; unknown keys and
    /// unknown names are errors.
    pub fn from_dict(dict: &Bound<'_, PyDict>, base: &State) -> PyResult<State> {
        let mut state = base.clone();
        for (key, value) in dict.iter() {
            let key: String = key.extract()?;
            let named = |what: &str| -> PyResult<&'static str> {
                let s: String = value.extract()?;
                let n = name(&s);
                if n.is_empty() {
                    return Err(crate::value_error(format!("{what}: unknown {s:?}")));
                }
                Ok(n)
            };
            match key.as_str() {
                "camera" => {
                    let mut camera = state.camera;
                    camera.update(&value.extract::<Bound<'_, PyDict>>()?)?;
                    state.camera = camera;
                }
                "frame" => state.frame = value.extract()?,
                "scrub_discrete" => state.scrub_discrete = value.extract()?,
                "theme" => state.theme = named("theme")?,
                "shading" => state.shading = named("shading")?,
                "style" => state.style = named("style")?,
                "sidechains" => state.sidechains = named("sidechains")?,
                "coloring" => {
                    let c: Option<String> = value.extract()?;
                    state.coloring = c.filter(|c| c != "element");
                }
                "layers" => {
                    let layers: Bound<'_, PyDict> = value.extract()?;
                    for (name, visible) in layers.iter() {
                        let (name, visible): (String, bool) = (name.extract()?, visible.extract()?);
                        match state.layers.iter_mut().find(|(n, _)| *n == name) {
                            Some(entry) => entry.1 = visible,
                            None => return Err(pyo3::exceptions::PyKeyError::new_err(format!("no layer {name:?}"))),
                        }
                    }
                }
                "color_gamma" => state.color_gamma = value.extract()?,
                "fog" => state.fog = value.extract()?,
                "occlusion" => state.occlusion = value.extract()?,
                "radius_scale" => state.radius_scale = value.extract()?,
                "surface_opacity" => state.surface_opacity = value.extract()?,
                "surface_colors" => {
                    let s: String = value.extract()?;
                    state.surface_colors_by_atom = match s.as_str() {
                        "atoms" => true,
                        "uniform" => false,
                        other => return Err(crate::value_error(format!("surface_colors: unknown {other:?}"))),
                    };
                }
                "vector_scales" => {
                    let scales: Bound<'_, PyDict> = value.extract()?;
                    for (name, scale) in scales.iter() {
                        let (name, scale): (String, f32) = (name.extract()?, scale.extract()?);
                        match state.vector_scales.iter_mut().find(|(n, _)| *n == name) {
                            Some(entry) => entry.1 = scale,
                            None => return Err(pyo3::exceptions::PyKeyError::new_err(format!("no vectors {name:?}"))),
                        }
                    }
                }
                "plots" => state.plots = value.extract()?,
                other => return Err(crate::value_error(format!("state: unknown key {other:?} (keys: {})", KEYS.join(", ")))),
            }
        }
        Ok(state)
    }

    /// Between this state and `other`: the camera and the sliders move, the rest is this state's.
    pub fn lerp(&self, other: &State, t: f32) -> State {
        let mix = |a: f32, b: f32| a + t * (b - a);
        State {
            camera: self.camera.lerp(&other.camera, t),
            color_gamma: mix(self.color_gamma, other.color_gamma),
            fog: mix(self.fog, other.fog),
            occlusion: mix(self.occlusion, other.occlusion),
            radius_scale: mix(self.radius_scale, other.radius_scale),
            surface_opacity: mix(self.surface_opacity, other.surface_opacity),
            vector_scales: self
                .vector_scales
                .iter()
                .map(|(name, a)| {
                    let b = other.vector_scales.iter().find(|(n, _)| n == name).map_or(*a, |(_, b)| *b);
                    (name.clone(), mix(*a, b))
                })
                .collect(),
            ..self.clone()
        }
    }

    /// Make the view this state, setting only what differs from `current` (the
    /// view's state now): changing the style or theme rebuilds the scene.
    pub fn apply(&self, v: &mut Viewer, current: &State) {
        if self.theme != current.theme {
            let _ = v.set_theme(self.theme);
        }
        if self.shading != current.shading {
            let _ = v.set_shading(self.shading);
        }
        if self.style != current.style {
            let _ = v.set_style(self.style);
        }
        if self.sidechains != current.sidechains {
            let _ = v.set_sidechains(self.sidechains);
        }
        if self.coloring != current.coloring {
            let _ = v.set_coloring(self.coloring.clone());
        }
        for (name, visible) in &self.layers {
            if !current.layers.contains(&(name.clone(), *visible)) {
                let _ = v.set_visible(name, *visible);
            }
        }
        if self.color_gamma != current.color_gamma {
            v.set_color_gamma(self.color_gamma);
        }
        if self.fog != current.fog {
            v.set_fog(self.fog);
        }
        if self.occlusion != current.occlusion {
            v.set_occlusion(self.occlusion);
        }
        if self.radius_scale != current.radius_scale {
            v.set_radius_scale(self.radius_scale);
        }
        if self.surface_opacity != current.surface_opacity {
            v.set_surface_opacity(self.surface_opacity);
        }
        if self.surface_colors_by_atom != current.surface_colors_by_atom {
            v.set_surface_colors_by_atom(self.surface_colors_by_atom);
        }
        for (name, scale) in &self.vector_scales {
            if !current.vector_scales.contains(&(name.clone(), *scale)) {
                let _ = v.set_vector_scale(name, *scale);
            }
        }
        // the frame counts the discrete states when scrubbing them
        if self.scrub_discrete != current.scrub_discrete {
            let _ = v.set_scrub_discrete(self.scrub_discrete);
        }
        if self.frame != current.frame || self.scrub_discrete != current.scrub_discrete {
            v.set_frame(self.frame);
        }
        v.show_plots = self.plots;
        v.camera = self.camera;
    }
}

/// States sampled at times in seconds from the start of the recording.
#[derive(Default, Clone)]
pub struct Recording {
    samples: Vec<(f64, State)>,
}

impl Recording {
    pub fn push(&mut self, time: f64, state: State) {
        self.samples.push((time, state));
    }

    /// Keep the sample only when the state changed (the first always): a
    /// recording of what was done, not of every redraw. A changed state is also
    /// preceded by the previous one at its own time, so a jump stays a jump.
    pub fn push_change(&mut self, time: f64, state: State) {
        match self.samples.last() {
            Some((_, last)) if *last == state => {}
            Some((t, last)) if time - t > 1e-3 => {
                let hold = (time - 1e-3, last.clone());
                self.samples.push(hold);
                self.samples.push((time, state));
            }
            _ => self.samples.push((time, state)),
        }
    }

    pub fn duration(&self) -> f64 {
        self.samples.last().map_or(0.0, |(t, _)| *t)
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn samples(&self) -> &[(f64, State)] {
        &self.samples
    }

    /// The state at `time`: the last sample's, with the camera and sliders moving
    /// on toward the next sample's.
    pub fn at(&self, time: f64) -> Option<State> {
        let i = self.samples.partition_point(|(t, _)| *t <= time).max(1) - 1;
        let (t0, a) = self.samples.get(i)?;
        if let Some((t1, b)) = self.samples.get(i + 1) {
            let w = ((time - t0) / (t1 - t0).max(1e-9)).clamp(0.0, 1.0) as f32;
            return Some(a.lerp(b, w));
        }
        Some(a.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_camera_turns_the_short_way_and_zooms_evenly() {
        let a = Camera { azimuth: 3.0, distance: 1.0, ..Camera::default() };
        let b = Camera { azimuth: -3.0, distance: 100.0, ..Camera::default() };
        let m = a.lerp(&b, 0.5);
        // from 3 to -3 rad is 0.28 rad through pi, not 6 rad through 0
        assert!((m.azimuth - (3.0 + (std::f32::consts::TAU - 6.0) / 2.0)).abs() < 1e-5);
        assert!((m.distance - 10.0).abs() < 1e-3);
        assert_eq!(a.lerp(&b, 0.0), a);
    }

    fn state(fog: f32, frame: usize) -> State {
        State {
            camera: Camera::default(),
            frame,
            scrub_discrete: false,
            theme: "light",
            shading: "lit",
            style: "atoms",
            sidechains: "sticks",
            coloring: None,
            layers: vec![("atoms".into(), true)],
            color_gamma: 1.0,
            fog,
            occlusion: 0.0,
            radius_scale: 1.0,
            surface_opacity: 0.4,
            surface_colors_by_atom: false,
            vector_scales: vec![("t".into(), 1.0)],
            plots: false,
        }
    }

    #[test]
    fn a_recording_keeps_changes_and_moves_the_sliders_between_them() {
        let mut recording = Recording::default();
        recording.push_change(0.0, state(0.0, 0));
        recording.push_change(0.5, state(0.0, 0)); // nothing changed: not kept
        recording.push_change(1.0, state(1.0, 3));
        assert_eq!(recording.samples().len(), 3); // the hold just before the change, then the change
        assert_eq!(recording.duration(), 1.0);
        let mid = recording.at(0.5).unwrap();
        assert_eq!((mid.fog, mid.frame), (0.0, 0)); // held until just before 1.0
        let late = recording.at(0.9995).unwrap();
        assert!(late.fog > 0.4 && late.fog < 0.6 && late.frame == 0);
        assert_eq!(recording.at(5.0).unwrap().fog, 1.0);
    }
}
