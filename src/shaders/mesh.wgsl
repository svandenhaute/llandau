// Triangle meshes in world space (the molecular surface).


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
    if g.shading.x > 0.5 {
        let facing = dot(normal, to_eye);
        let pixels = facing / max(fwidth(facing), 1e-6);
        let flat = mix(linear(in.color.rgb), linear(g.outline.rgb), outline(pixels, in.position.z));
        return vec4(fog(flat, in.position.z), in.color.a);
    }
    // a darker rim where the surface turns away: it reads as an outline
    let rim = 1.0 - 0.5 * pow(1.0 - abs(dot(normal, to_eye)), 2.0);
    let lit = shade(linear(in.color.rgb), normal, to_eye) * rim;
    return vec4(fog(lit, in.position.z), in.color.a);
}
