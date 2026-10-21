// Slides: flat colored triangles (tessellated paths) and textured quads
// (images), drawn in frame units and blended as written (sRGB values, as
// cairo does); then downsampled from the supersampled target.

struct Frame {
    // frame units to clip space
    scale: vec2<f32>,
    // 1 when the final target decodes sRGB (blit only)
    srgb: f32,
    _pad: f32,
};
@group(0) @binding(0) var<uniform> frame: Frame;

struct ShapeOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) color: vec4<f32>,
};

@vertex
fn shape_vs(@location(0) position: vec2<f32>, @location(1) color: vec4<f32>) -> ShapeOut {
    var out: ShapeOut;
    out.clip = vec4(position * frame.scale, 0.0, 1.0);
    out.color = color;
    return out;
}

@fragment
fn shape_fs(in: ShapeOut) -> @location(0) vec4<f32> {
    return in.color;
}

@group(1) @binding(0) var image: texture_2d<f32>;
@group(1) @binding(1) var image_sampler: sampler;

struct ImageOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) opacity: f32,
};

@vertex
fn image_vs(@location(0) position: vec2<f32>, @location(1) uv: vec2<f32>, @location(2) opacity: f32) -> ImageOut {
    var out: ImageOut;
    out.clip = vec4(position * frame.scale, 0.0, 1.0);
    out.uv = uv;
    out.opacity = opacity;
    return out;
}

@fragment
fn image_fs(in: ImageOut) -> @location(0) vec4<f32> {
    let c = textureSample(image, image_sampler, in.uv);
    return vec4(c.rgb, c.a * in.opacity);
}

@fragment
fn layer_fs(in: ImageOut) -> @location(0) vec4<f32> {
    // a fade's layer, drawn over transparent black: its colors are premultiplied
    let c = textureSample(image, image_sampler, in.uv);
    let rgb = select(c.rgb / c.a, vec3(0.0), c.a <= 0.0);
    return vec4(rgb, c.a * in.opacity);
}

struct BlitOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn blit_vs(@builtin(vertex_index) index: u32) -> BlitOut {
    // one triangle covering the viewport
    let uv = vec2(f32((index << 1u) & 2u), f32(index & 2u));
    var out: BlitOut;
    out.clip = vec4(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

fn decode(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, c <= vec3(0.04045));
}

@fragment
fn blit_fs(in: BlitOut) -> @location(0) vec4<f32> {
    // halfway between the texels of each 2x2 block: the linear filter averages them
    let c = textureSample(image, image_sampler, in.uv);
    if frame.srgb > 0.5 {
        return vec4(decode(c.rgb), c.a);
    }
    return c;
}
