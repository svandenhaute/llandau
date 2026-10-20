// Ray-cast sphere impostors: one camera-facing quad per atom, exact
// per-pixel depth and a smooth (alpha-to-coverage) silhouette.

struct Globals {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    viewport: vec4<f32>,
    light: vec4<f32>,
};
@group(0) @binding(0) var<uniform> g: Globals;

struct Sphere {
    @location(0) center: vec3<f32>,
    @location(1) radius: f32,
    @location(2) color: vec4<f32>,
};

struct VertexOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) position: vec3<f32>,
    @location(1) center: vec3<f32>,
    @location(2) radius: f32,
    @location(3) color: vec4<f32>,
    @location(4) uv: vec2<f32>,
};

const MARGIN: f32 = 1.1;  // room for the antialiased edge

@vertex
fn vs(@builtin(vertex_index) index: u32, sphere: Sphere) -> VertexOut {
    var corners = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    let uv = corners[index] * MARGIN;
    let center = (g.view * vec4(sphere.center, 1.0)).xyz;
    let distance = length(center);
    let r = sphere.radius;
    // The silhouette cone cuts the plane through the center, normal to the
    // line of sight, in a circle of this radius.
    let half_size = r * distance / sqrt(max(distance * distance - r * r, 1e-6));
    let w = center / distance;
    var helper = vec3(0.0, 1.0, 0.0);
    if abs(w.y) > 0.99 {
        helper = vec3(1.0, 0.0, 0.0);
    }
    let u = normalize(cross(helper, w));
    let v = cross(w, u);
    let position = center + (uv.x * u + uv.y * v) * half_size;

    var out: VertexOut;
    out.clip = g.proj * vec4(position, 1.0);
    out.position = position;
    out.center = center;
    out.radius = r;
    out.color = sphere.color;
    out.uv = uv;
    return out;
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(in: VertexOut) -> FragmentOut {
    let edge = length(in.uv);
    let coverage = clamp((1.0 - edge) / max(fwidth(edge), 1e-6) + 0.5, 0.0, 1.0);
    if coverage <= 0.0 {
        discard;
    }
    let ray = normalize(in.position);
    let b = dot(ray, in.center);
    let c = dot(in.center, in.center) - in.radius * in.radius;
    let t = b - sqrt(max(b * b - c, 0.0));
    let hit = t * ray;
    let normal = normalize(hit - in.center);
    let clip = g.proj * vec4(hit, 1.0);

    var out: FragmentOut;
    out.color = vec4(shade(in.color.rgb, normal, -ray), coverage * in.color.a);
    out.depth = clip.z / clip.w;
    return out;
}

fn shade(color: vec3<f32>, normal: vec3<f32>, to_eye: vec3<f32>) -> vec3<f32> {
    let diffuse = max(dot(normal, g.light.xyz), 0.0);
    let half_vector = normalize(g.light.xyz + to_eye);
    let specular = pow(max(dot(normal, half_vector), 0.0), 40.0);
    return color * (0.38 + 0.62 * diffuse) + 0.12 * specular;
}
