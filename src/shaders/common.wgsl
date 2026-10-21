// Shared by the 3D shaders (prepended to each): the globals, and shading,
// outlines and depth cueing in view space.

struct Globals {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec4<f32>,
    light: vec4<f32>,
    // x: 1 for flat shading; y: outline width in angstrom, z: at least this many pixels;
    // w: halo width in angstrom
    shading: vec4<f32>,
    outline: vec4<f32>,
    background: vec4<f32>,
    // x, y: view depths of the nearest and farthest atom; z: fog; w: halo depth in angstrom
    cue: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

// 0 for pipelines whose flat shading leaves out the outline (trails, overlays)
override OUTLINES: f32 = 1.0;

fn shade(color: vec3<f32>, normal: vec3<f32>, to_eye: vec3<f32>) -> vec3<f32> {
    let diffuse = max(dot(normal, g.light.xyz), 0.0);
    let half_vector = normalize(g.light.xyz + to_eye);
    let specular = pow(max(dot(normal, half_vector), 0.0), 40.0);
    return color * (0.38 + 0.62 * diffuse) + 0.12 * specular;
}

// Colors come in as written in the color tables (sRGB-like); the output is
// sRGB-encoded, so decoding with the gamma in light.w (2.2: exact, 1: none)
// sets how dark and saturated they come out.
fn linear(c: vec3<f32>) -> vec3<f32> {
    return pow(max(c, vec3(0.0)), vec3(g.light.w));
}

// How much of the outline color a fragment `pixels` from the silhouette takes,
// at view depth `z`.
fn outline(pixels: f32, z: f32) -> f32 {
    return OUTLINES * clamp(outline_width(z) - pixels + 0.5, 0.0, 1.0);
}

// Outline width in pixels at view depth z: wider in front, thinner at the back.
fn outline_width(z: f32) -> f32 {
    return max(g.shading.y / pixel_size(z), g.shading.z) * mix(1.25, 0.75, depth(z));
}

// The size of a pixel at view depth z, in angstrom.
fn pixel_size(z: f32) -> f32 {
    return 2.0 * -z / (g.proj[1][1] * g.viewport.y);
}

// 0 at the nearest atom, 1 at the farthest.
fn depth(z: f32) -> f32 {
    return clamp((-z - g.cue.x) / max(g.cue.y - g.cue.x, 1e-3), 0.0, 1.0);
}

// Depth cueing: farther away, closer to the background.
fn fog(color: vec3<f32>, z: f32) -> vec3<f32> {
    return mix(color, g.background.rgb, g.cue.z * depth(z));
}

