// Ray-cast sphere impostors: one camera-facing quad per atom, exact
// per-pixel depth and a smooth (alpha-to-coverage) silhouette. In flat
// shading, a halo ring around the silhouette, set back in depth, marks
// where an atom passes in front of others far behind it.

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
// 1 for the pipeline that draws halos (opaque atoms)
override HALO: f32 = 0.0;

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
    @location(5) joint: f32,
    // the silhouette's radius in the plane through the center
    @location(6) half_size: f32,
};

@vertex
fn vs(@builtin(vertex_index) index: u32, sphere: Sphere) -> VertexOut {
    var corners = array(vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(-1.0, 1.0), vec2(1.0, 1.0));
    let center = (g.view * vec4(sphere.center, 1.0)).xyz;
    let distance = length(center);
    let r = abs(sphere.radius);  // negative for joints (no outline)
    // The silhouette cone cuts the plane through the center, normal to the
    // line of sight, in a circle of this radius.
    let half_size = r * distance / sqrt(max(distance * distance - r * r, 1e-6));
    // room for the halo and the antialiased edge
    let joint = select(0.0, 1.0, sphere.radius < 0.0);
    let room = (halo_width(center.z, joint) + 1.5) * pixel_size(center.z) / half_size;
    let uv = corners[index] * (1.0 + room);
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
    out.joint = joint;
    out.half_size = half_size;
    return out;
}

struct FragmentOut {
    @location(0) color: vec4<f32>,
    @builtin(frag_depth) depth: f32,
};

@fragment
fn fs(in: VertexOut) -> FragmentOut {
    // pixels inside the silhouette, exactly (fwidth, per 2x2 block, makes a wobbly rim)
    let edge = length(in.uv);
    let pixels = (1.0 - edge) * in.half_size / pixel_size(in.center.z);
    let halo = halo_width(in.center.z, in.joint);
    let coverage = clamp(pixels + halo + 0.5, 0.0, 1.0);
    if coverage <= 0.0 {
        discard;
    }
    var out: FragmentOut;
    if pixels < 0.0 {
        // the halo, g.cue.w behind the atom's back
        let z = in.center.z - in.radius - g.cue.w;
        let clip = g.proj * vec4(in.position * (z / in.position.z), 1.0);
        out.color = vec4(fog(linear(g.outline.rgb), in.center.z), coverage * in.color.a);
        out.depth = clip.z / clip.w;
        return out;
    }
    let ray = normalize(in.position);
    let b = dot(ray, in.center);
    let c = dot(in.center, in.center) - in.radius * in.radius;
    let t = b - sqrt(max(b * b - c, 0.0));
    let hit = t * ray;
    let normal = normalize(hit - in.center);
    let clip = g.proj * vec4(hit, 1.0);

    var color = shade(linear(in.color.rgb), normal, -ray);
    if g.shading.x > 0.5 {
        let amount = (1.0 - in.joint) * outline(pixels, in.center.z);
        color = mix(linear(in.color.rgb), linear(g.outline.rgb), amount);
    }

    // the halo's pixels are opaque, so the silhouette's edge is too
    let alpha = select(clamp(pixels + 0.5, 0.0, 1.0), 1.0, halo > 0.0);
    out.color = vec4(fog(color, hit.z), alpha * in.color.a);
    out.depth = clip.z / clip.w;
    return out;
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

// Halo width in pixels outside the silhouette (0: none, e.g. for joints or lit shading).
fn halo_width(z: f32, joint: f32) -> f32 {
    let on = HALO * OUTLINES * step(0.5, g.shading.x) * (1.0 - joint);
    return on * max(g.shading.w / pixel_size(z), g.shading.z) * mix(1.25, 0.75, depth(z));
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
