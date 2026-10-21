//! The explorer's recordings: the view's state, sampled at every redraw while
//! recording, and replayed at a fixed frame rate (the camera interpolated
//! between samples) to render a video offscreen.

use crate::Viewer;
use crate::camera::Camera;
use crate::scene::SurfaceColors;

/// What a recording keeps of the view at one moment.
#[derive(Clone, Debug, PartialEq)]
pub struct State {
    camera: Camera,
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
        v.camera = self.camera;
    }
}

/// States sampled at times in seconds from the start of the recording.
#[derive(Default)]
pub struct Recording {
    samples: Vec<(f64, State)>,
}

impl Recording {
    pub fn push(&mut self, time: f64, state: State) {
        self.samples.push((time, state));
    }

    pub fn duration(&self) -> f64 {
        self.samples.last().map_or(0.0, |(t, _)| *t)
    }

    /// The state at `time`: the last sample's then, with the camera moving on
    /// toward the next sample's.
    pub fn at(&self, time: f64) -> Option<State> {
        let i = self.samples.partition_point(|(t, _)| *t <= time).max(1) - 1;
        let (t0, a) = self.samples.get(i)?;
        let mut state = a.clone();
        if let Some((t1, b)) = self.samples.get(i + 1) {
            let w = ((time - t0) / (t1 - t0).max(1e-9)).clamp(0.0, 1.0) as f32;
            state.camera = a.camera.lerp(&b.camera, w);
        }
        Some(state)
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
}
