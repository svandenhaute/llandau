//! Turntable camera around a target, z up (as the trajectories are stored).

use glam::camera::rh::{proj::directx, view};
use glam::{Mat4, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    /// angle around z, radians
    pub azimuth: f32,
    /// angle above the xy plane, radians
    pub elevation: f32,
    /// vertical field of view, radians
    pub fov_y: f32,
    /// radius of the scene, for the near and far planes
    pub extent: f32,
}

const MAX_ELEVATION: f32 = 1.55;

impl Default for Camera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: 10.0,
            azimuth: -std::f32::consts::FRAC_PI_2,
            elevation: 0.3,
            fov_y: 35f32.to_radians(),
            extent: 5.0,
        }
    }
}

impl Camera {
    pub fn eye(&self) -> Vec3 {
        let (sa, ca) = self.azimuth.sin_cos();
        let (se, ce) = self.elevation.sin_cos();
        self.target + self.distance * Vec3::new(ce * ca, ce * sa, se)
    }

    pub fn view(&self) -> Mat4 {
        view::look_at_mat4(self.eye(), self.target, Vec3::Z)
    }

    pub fn projection(&self, aspect: f32) -> Mat4 {
        let near = (self.distance - 2.0 * self.extent).max(0.01 * self.distance);
        let far = self.distance + 2.0 * self.extent;
        // depth in [0, 1] and y up, as wgpu expects
        directx::perspective(self.fov_y, aspect, near, far)
    }

    /// Frame a sphere (center, radius) so it fits the view.
    pub fn fit(&mut self, center: Vec3, radius: f32) {
        self.target = center;
        self.extent = radius.max(1e-3);
        self.distance = 1.05 * self.extent / (0.5 * self.fov_y).sin();
    }

    /// Rotate by a mouse drag of (dx, dy) pixels.
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        self.azimuth -= 0.008 * dx;
        self.elevation = (self.elevation + 0.008 * dy).clamp(-MAX_ELEVATION, MAX_ELEVATION);
    }

    /// Move the target by a mouse drag of (dx, dy) pixels in a view `height` pixels high.
    pub fn pan(&mut self, dx: f32, dy: f32, height: f32) {
        let view = self.view();
        let right = view.row(0).truncate();
        let up = view.row(1).truncate();
        let per_pixel = 2.0 * self.distance * (0.5 * self.fov_y).tan() / height.max(1.0);
        self.target += per_pixel * (-dx * right + dy * up);
    }

    /// Multiply the distance to the target (< 1 moves closer).
    pub fn zoom(&mut self, factor: f32) {
        self.distance = (self.distance * factor).max(1e-3);
    }

    /// Interpolate towards `other` (t in [0, 1]); angles take the short way around.
    pub fn lerp(&self, other: &Camera, t: f32) -> Camera {
        let tau = std::f32::consts::TAU;
        let mut azimuth = (other.azimuth - self.azimuth) % tau;
        if azimuth > std::f32::consts::PI {
            azimuth -= tau;
        } else if azimuth < -std::f32::consts::PI {
            azimuth += tau;
        }
        let mix = |a: f32, b: f32| a + t * (b - a);
        Camera {
            target: self.target.lerp(other.target, t),
            distance: self.distance * (other.distance / self.distance).powf(t),
            azimuth: self.azimuth + t * azimuth,
            elevation: mix(self.elevation, other.elevation),
            fov_y: mix(self.fov_y, other.fov_y),
            extent: mix(self.extent, other.extent),
        }
    }
}
