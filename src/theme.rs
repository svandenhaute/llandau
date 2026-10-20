//! Colors of everything that is not an element color, per background.

pub type Rgba = [f32; 4];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Theme {
    pub background: Rgba,
    pub bond: Rgba,
    /// trail segments alternate between these two
    pub trail: [Rgba; 2],
    pub vector: Rgba,
    pub tube: Rgba,
    pub clash: Rgba,
}

const fn gray(value: f32) -> Rgba {
    [value, value, value, 1.0]
}

pub const LIGHT: Theme = Theme {
    background: gray(1.0),
    bond: gray(0.12),
    trail: [gray(0.72), gray(0.42)],
    vector: gray(0.05),
    tube: [0.55, 0.62, 0.75, 1.0],
    clash: [0.85, 0.1, 0.75, 1.0], // magenta: not an element color
};

pub const DARK: Theme = Theme {
    background: gray(0.0),
    bond: gray(0.75),
    trail: [gray(0.35), gray(0.6)],
    vector: gray(0.95),
    tube: [0.45, 0.55, 0.7, 1.0],
    clash: [1.0, 0.3, 0.9, 1.0],
};

impl Theme {
    pub fn named(name: &str) -> Option<Theme> {
        match name {
            "light" => Some(LIGHT),
            "dark" => Some(DARK),
            _ => None,
        }
    }
}

/// Mix `color` towards `target` by `amount`, and set its alpha.
pub fn fade(color: Rgba, target: Rgba, amount: f32, alpha: f32) -> Rgba {
    let mix = |a: f32, b: f32| a + amount * (b - a);
    [mix(color[0], target[0]), mix(color[1], target[1]), mix(color[2], target[2]), alpha]
}
