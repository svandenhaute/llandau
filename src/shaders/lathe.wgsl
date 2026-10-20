// Instanced surfaces of revolution (bond and tube cylinders, arrow heads):
// a unit mesh around z, stretched between two points. Each half takes its
// own color.

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

struct Vertex {
    // on the unit mesh: radius 1 in xy, length 1 along z
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
};

struct Instance {
    @location(2) start: vec3<f32>,
    @location(3) radius: f32,
    @location(4) end: vec3<f32>,
    @location(5) color_start: vec4<f32>,
    @location(6) color_end: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) position: vec3<f32>,
    @location(2) along: f32,
    @location(3) color_start: vec4<f32>,
    @location(4) color_end: vec4<f32>,
    @location(5) radius: f32,
    @location(6) axis: vec3<f32>,
};

@vertex
fn vs(vertex: Vertex, instance: Instance) -> VertexOut {
    let axis = instance.end - instance.start;
    let length = max(length(axis), 1e-6);
    let w = axis / length;
    var helper = vec3(0.0, 0.0, 1.0);
    if abs(w.z) > 0.99 {
        helper = vec3(1.0, 0.0, 0.0);
    }
    let u = normalize(cross(helper, w));
    let v = cross(w, u);
    let r = instance.radius;
    let world = instance.start + vertex.position.z * axis + r * (vertex.position.x * u + vertex.position.y * v);
    // normals transform with the inverse transpose of the (radius, radius, length) scaling
    let normal = vertex.normal.x / r * u + vertex.normal.y / r * v + vertex.normal.z / length * w;
    let position = (g.view * vec4(world, 1.0)).xyz;

    var out: VertexOut;
    out.clip = g.proj * vec4(position, 1.0);
    out.normal = (g.view * vec4(normal, 0.0)).xyz;
    out.position = position;
    out.along = vertex.position.z;
    out.color_start = instance.color_start;
    out.color_end = instance.color_end;
    out.radius = r;
    out.axis = (g.view * vec4(w, 0.0)).xyz;
    return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    let color = select(in.color_start, in.color_end, in.along > 0.5);
    let normal = normalize(in.normal);
    let to_eye = normalize(-in.position);
    if g.shading.x > 0.5 {
        // distance from the axis on screen as a fraction of the radius, whatever the tilt:
        // normal and eye direction in the plane normal to the axis (exact pixels: fwidth
        // makes a wobbly rim)
        let axis = normalize(in.axis);
        let around = normal - dot(normal, axis) * axis;
        let eye = to_eye - dot(to_eye, axis) * axis;
        let facing = dot(around, eye) / max(length(around) * length(eye), 1e-6);
        let across = sqrt(max(1.0 - facing * facing, 0.0));
        let pixel_size = 2.0 * -in.position.z / (g.proj[1][1] * g.viewport.y);
        let pixels = (1.0 - across) * in.radius / pixel_size;
        let flat = mix(linear(color.rgb), linear(g.outline.rgb), outline(pixels, in.position.z));
        return vec4(fog(flat, in.position.z), color.a);
    }
    return vec4(fog(shade(linear(color.rgb), normal, to_eye), in.position.z), color.a);
}

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
