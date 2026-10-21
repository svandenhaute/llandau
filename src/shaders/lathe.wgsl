// Instanced surfaces of revolution (bond and tube cylinders, arrow heads):
// a unit mesh around z, stretched between two points. Each half takes its
// own color.


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
        let pixels = (1.0 - across) * in.radius / pixel_size(in.position.z);
        let flat = mix(linear(color.rgb), linear(g.outline.rgb), outline(pixels, in.position.z));
        return vec4(fog(flat, in.position.z), color.a);
    }
    return vec4(fog(shade(linear(color.rgb), normal, to_eye), in.position.z), color.a);
}
