// Ray-cast sphere impostors: one camera-facing quad per atom, exact
// per-pixel depth and a smooth (alpha-to-coverage) silhouette. In flat
// shading, a halo ring around the silhouette, set back in depth, marks
// where an atom passes in front of others far behind it.

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

// Halo width in pixels outside the silhouette (0: none, e.g. for joints or lit shading).
fn halo_width(z: f32, joint: f32) -> f32 {
    let on = HALO * OUTLINES * step(0.5, g.shading.x) * (1.0 - joint);
    return on * max(g.shading.w / pixel_size(z), g.shading.z) * mix(1.25, 0.75, depth(z));
}
