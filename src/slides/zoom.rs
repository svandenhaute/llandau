//! Zooming into the slide in the player: a magnification around a point of the
//! frame, so a detail of a figure can be shown large. Everything on the slide
//! zooms, live views included (drawn at the size they then have).

use super::paint::letterbox;

pub const MIN_SCALE: f32 = 1.0;
pub const MAX_SCALE: f32 = 8.0;

/// The slide magnified `scale` times around `center` (frame units), which is
/// drawn at the middle of the viewport.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Zoom {
    pub center: [f32; 2],
    pub scale: f32,
}

impl Default for Zoom {
    fn default() -> Self {
        Zoom { center: [0.0, 0.0], scale: 1.0 }
    }
}

impl Zoom {
    pub fn is_identity(&self) -> bool {
        self.scale == 1.0 && self.center == [0.0, 0.0]
    }

    /// The frame point shown at a pixel of a target of `size` (the frame letterboxed in it).
    pub fn point_at(&self, frame: [f32; 2], size: (u32, u32), pixel: (f64, f64)) -> [f32; 2] {
        let [vx, vy, vw, vh] = letterbox(frame, size);
        let u = ((pixel.0 as f32 - vx) / vw - 0.5) * frame[0];
        let v = (0.5 - (pixel.1 as f32 - vy) / vh) * frame[1];
        [self.center[0] + u / self.scale, self.center[1] + v / self.scale]
    }

    /// The pixel a frame point is shown at.
    pub fn pixel_of(&self, frame: [f32; 2], size: (u32, u32), point: [f32; 2]) -> [f32; 2] {
        let [vx, vy, vw, vh] = letterbox(frame, size);
        let u = (point[0] - self.center[0]) * self.scale;
        let v = (point[1] - self.center[1]) * self.scale;
        [vx + (u / frame[0] + 0.5) * vw, vy + (0.5 - v / frame[1]) * vh]
    }

    /// Multiply the magnification by `factor`, keeping what is under `pixel` there.
    pub fn zoom_at(&mut self, frame: [f32; 2], size: (u32, u32), pixel: (f64, f64), factor: f32) {
        let fixed = self.point_at(frame, size, pixel);
        self.scale = (self.scale * factor).clamp(MIN_SCALE, MAX_SCALE);
        let now = self.point_at(frame, size, pixel);
        self.center = [self.center[0] + fixed[0] - now[0], self.center[1] + fixed[1] - now[1]];
        self.clamp(frame);
    }

    /// Move the view by a drag of (dx, dy) pixels: the slide follows the cursor.
    pub fn pan(&mut self, frame: [f32; 2], size: (u32, u32), dx: f64, dy: f64) {
        let [_, _, vw, vh] = letterbox(frame, size);
        self.center[0] -= dx as f32 / vw * frame[0] / self.scale;
        self.center[1] += dy as f32 / vh * frame[1] / self.scale;
        self.clamp(frame);
    }

    /// Keep the frame filling the viewport: no empty margin comes into view.
    fn clamp(&mut self, frame: [f32; 2]) {
        if self.scale <= MIN_SCALE {
            *self = Zoom::default();
            return;
        }
        for i in 0..2 {
            let reach = frame[i] / 2.0 * (1.0 - 1.0 / self.scale);
            self.center[i] = self.center[i].clamp(-reach, reach);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME: [f32; 2] = [16.0, 9.0];
    const SIZE: (u32, u32) = (1600, 900);

    #[test]
    fn zooming_keeps_the_point_under_the_cursor() {
        let mut zoom = Zoom::default();
        let pixel = (400.0, 225.0); // a quarter in: frame point (-4, 2.25)
        assert_eq!(zoom.point_at(FRAME, SIZE, pixel), [-4.0, 2.25]);
        zoom.zoom_at(FRAME, SIZE, pixel, 2.0);
        assert_eq!(zoom.scale, 2.0);
        let p = zoom.point_at(FRAME, SIZE, pixel);
        assert!((p[0] + 4.0).abs() < 1e-5 && (p[1] - 2.25).abs() < 1e-5);
        let back = zoom.pixel_of(FRAME, SIZE, [-4.0, 2.25]);
        assert!((back[0] - 400.0).abs() < 1e-3 && (back[1] - 225.0).abs() < 1e-3);
    }

    #[test]
    fn the_frame_keeps_filling_the_viewport() {
        let mut zoom = Zoom::default();
        zoom.zoom_at(FRAME, SIZE, (0.0, 0.0), 2.0); // the corner stays at the corner
        assert_eq!(zoom.center, [-4.0, 2.25]);
        zoom.pan(FRAME, SIZE, 10_000.0, 10_000.0); // dragging far: stops at the edge
        assert_eq!(zoom.center, [-4.0, 2.25]);
        zoom.pan(FRAME, SIZE, -10_000.0, -10_000.0);
        assert_eq!(zoom.center, [4.0, -2.25]);
        zoom.zoom_at(FRAME, SIZE, (800.0, 450.0), 0.1); // out below 1: back to the whole slide
        assert_eq!(zoom, Zoom::default());
        zoom.zoom_at(FRAME, SIZE, (800.0, 450.0), 100.0);
        assert_eq!(zoom.scale, MAX_SCALE);
    }
}
