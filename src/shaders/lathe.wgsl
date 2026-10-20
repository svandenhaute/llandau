// Instanced surfaces of revolution (bond and tube cylinders, arrow heads):
// a unit mesh around z, stretched between two points. Each half takes its
// own color.

struct Globals {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec4<f32>,
    light: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

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
    return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    let color = select(in.color_start, in.color_end, in.along > 0.5);
    let normal = normalize(in.normal);
    return vec4(shade(linear(color.rgb), normal, normalize(-in.position)), color.a);
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
