// Triangle meshes in world space (the molecular surface).

struct Globals {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec4<f32>,
    light: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

struct Vertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) position: vec3<f32>,
    @location(2) color: vec4<f32>,
};

@vertex
fn vs(vertex: Vertex) -> VertexOut {
    let position = (g.view * vec4(vertex.position, 1.0)).xyz;
    var out: VertexOut;
    out.clip = g.proj * vec4(position, 1.0);
    out.normal = (g.view * vec4(vertex.normal, 0.0)).xyz;
    out.position = position;
    out.color = vertex.color;
    return out;
}

@fragment
fn fs(in: VertexOut) -> @location(0) vec4<f32> {
    var normal = normalize(in.normal);
    let to_eye = normalize(-in.position);
    if dot(normal, to_eye) < 0.0 {
        normal = -normal;  // seen from inside
    }
    let diffuse = max(dot(normal, g.light.xyz), 0.0);
    let half_vector = normalize(g.light.xyz + to_eye);
    let specular = pow(max(dot(normal, half_vector), 0.0), 40.0);
    // a darker rim where the surface turns away: it reads as an outline
    let rim = 1.0 - 0.5 * pow(1.0 - abs(dot(normal, to_eye)), 2.0);
    return vec4((linear(in.color.rgb) * (0.38 + 0.62 * diffuse) + 0.12 * specular) * rim, in.color.a);
}

// Colors come in as sRGB (as written in color tables); light and blend in linear.
fn linear(c: vec3<f32>) -> vec3<f32> {
    return select(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, c <= vec3(0.04045));
}
