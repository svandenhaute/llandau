//! Drawing a timeline at one moment: paths tessellated into triangles (lyon),
//! images as textured quads, all into a supersampled, multisampled target
//! that is then averaged down into the window or an offscreen texture.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use lyon::math::point;
use lyon::path::Path;
use lyon::tessellation::{
    BuffersBuilder, FillOptions, FillRule, FillTessellator, FillVertex, LineCap, LineJoin, StrokeOptions,
    StrokeTessellator, StrokeVertex, VertexBuffers,
};
use wgpu::util::DeviceExt;

use super::timeline::{Cap, Item, Join, Shape, Timeline};

const SAMPLES: u32 = 4;
/// Supersampling on top of the multisampling: rendered at this many times the size.
const SUPERSAMPLE: u32 = 2;
/// Tessellation tolerance, in pixels of the supersampled target.
const TOLERANCE: f32 = 0.25;
const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// An image to draw: RGBA, rows top to bottom.
pub struct Image {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ShapeVertex {
    position: [f32; 2],
    color: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct ImageVertex {
    position: [f32; 2],
    uv: [f32; 2],
    opacity: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct FrameUniform {
    scale: [f32; 2],
    srgb: f32,
    _pad: f32,
}

#[derive(Default)]
struct Mesh {
    vertices: Vec<ShapeVertex>,
    indices: Vec<u32>,
}

enum Command {
    Shapes(std::ops::Range<u32>),
    Image { image: usize, first: u32 },
    Live { viewer: usize, first: u32 },
}

struct Targets {
    size: (u32, u32),
    msaa: wgpu::TextureView,
    resolved: wgpu::TextureView,
    blit_group: wgpu::BindGroup,
}

pub struct Painter {
    shape_pipeline: wgpu::RenderPipeline,
    image_pipeline: wgpu::RenderPipeline,
    blit_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    blit_layout: wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
    frame_buffer: wgpu::Buffer,
    frame_group: wgpu::BindGroup,
    blit_frame_buffer: wgpu::Buffer,
    blit_frame_group: wgpu::BindGroup,
    texture_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    images: Vec<wgpu::BindGroup>,
    /// live views' pictures, by viewer
    live: HashMap<usize, wgpu::BindGroup>,
    targets: Option<Targets>,
    /// tessellations of unchanged shapes, by shape id, at `cache_tolerance`
    cache: HashMap<usize, Mesh>,
    cache_tolerance: f32,
}

impl Painter {
    pub fn new(device: &wgpu::Device) -> Painter {
        let shader = device.create_shader_module(wgpu::include_wgsl!("../shaders/slides.wgsl"));
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("frame"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("image"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let uniform = |label| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<FrameUniform>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &uniform_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
            });
            (buffer, group)
        };
        let (frame_buffer, frame_group) = uniform("frame");
        let (blit_frame_buffer, blit_frame_group) = uniform("blit frame");
        let shape_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shapes"),
            bind_group_layouts: &[Some(&uniform_layout)],
            immediate_size: 0,
        });
        let textured_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("images"),
            bind_group_layouts: &[Some(&uniform_layout), Some(&texture_layout)],
            immediate_size: 0,
        });
        let attribute = |shader_location, offset, format| wgpu::VertexAttribute { format, offset, shader_location };
        let shape_attributes =
            [attribute(0, 0, wgpu::VertexFormat::Float32x2), attribute(1, 8, wgpu::VertexFormat::Float32x4)];
        let image_attributes = [
            attribute(0, 0, wgpu::VertexFormat::Float32x2),
            attribute(1, 8, wgpu::VertexFormat::Float32x2),
            attribute(2, 16, wgpu::VertexFormat::Float32),
        ];
        let pipeline = |label, layout: &wgpu::PipelineLayout, entry: &str, stride: usize, attributes: &[wgpu::VertexAttribute]| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(&format!("{entry}_vs")),
                    compilation_options: Default::default(),
                    buffers: &[Some(wgpu::VertexBufferLayout {
                        array_stride: stride as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes,
                    })],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState { count: SAMPLES, mask: !0, alpha_to_coverage_enabled: false },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(&format!("{entry}_fs")),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: FORMAT,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let shape_pipeline =
            pipeline("shapes", &shape_layout, "shape", std::mem::size_of::<ShapeVertex>(), &shape_attributes);
        let image_pipeline =
            pipeline("images", &textured_layout, "image", std::mem::size_of::<ImageVertex>(), &image_attributes);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        Painter {
            shape_pipeline,
            image_pipeline,
            blit_pipelines: HashMap::new(),
            blit_layout: textured_layout,
            shader,
            frame_buffer,
            frame_group,
            blit_frame_buffer,
            blit_frame_group,
            texture_layout,
            sampler,
            images: Vec::new(),
            live: HashMap::new(),
            targets: None,
            cache: HashMap::new(),
            cache_tolerance: 0.0,
        }
    }

    fn upload_images(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, images: &[Image]) {
        for image in &images[self.images.len()..] {
            let levels = mip_levels(image);
            let texture = device.create_texture_with_data(
                queue,
                &wgpu::TextureDescriptor {
                    label: Some("image"),
                    size: wgpu::Extent3d { width: image.width, height: image.height, depth_or_array_layers: 1 },
                    mip_level_count: levels.len() as u32,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: FORMAT,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                wgpu::util::TextureDataOrder::LayerMajor,
                &levels.concat(),
            );
            let view = texture.create_view(&Default::default());
            self.images.push(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("image"),
                layout: &self.texture_layout,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            }));
        }
    }

    /// The picture of a live view (drawn by its viewer), sampled as written.
    pub fn set_live(&mut self, device: &wgpu::Device, viewer: usize, view: &wgpu::TextureView) {
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("live"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.live.insert(viewer, group);
    }

    fn ensure_targets(&mut self, device: &wgpu::Device, size: (u32, u32)) {
        if matches!(&self.targets, Some(t) if t.size == size) {
            return;
        }
        let texture = |label, samples, usage| {
            device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: samples,
                    dimension: wgpu::TextureDimension::D2,
                    format: FORMAT,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };
        let msaa = texture("slides msaa", SAMPLES, wgpu::TextureUsages::RENDER_ATTACHMENT);
        let resolved = texture(
            "slides resolved",
            1,
            wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let blit_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("blit"),
            layout: &self.texture_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&resolved) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        self.targets = Some(Targets { size, msaa, resolved, blit_group });
    }

    fn blit_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) -> &wgpu::RenderPipeline {
        self.blit_pipelines.entry(format).or_insert_with(|| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blit"),
                layout: Some(&self.blit_layout),
                vertex: wgpu::VertexState {
                    module: &self.shader,
                    entry_point: Some("blit_vs"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.shader,
                    entry_point: Some("blit_fs"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })],
                }),
                multiview_mask: None,
                cache: None,
            })
        })
    }

    /// Draw `timeline` at `time` into `target` (`size` pixels, in `format`),
    /// letterboxed to the frame's aspect ratio. `images` are the timeline's.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        timeline: &Timeline,
        images: &[Image],
        time: f64,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        size: (u32, u32),
    ) {
        self.upload_images(device, queue, images);
        let viewport = letterbox(timeline.frame, size);
        let max = device.limits().max_texture_dimension_2d;
        let inner = (
            ((viewport[2] as u32) * SUPERSAMPLE).clamp(1, max),
            ((viewport[3] as u32) * SUPERSAMPLE).clamp(1, max),
        );
        self.ensure_targets(device, inner);

        // tessellate, in drawing order
        let tolerance = TOLERANCE * timeline.frame[0] / inner.0 as f32;
        if (tolerance - self.cache_tolerance).abs() > 0.25 * self.cache_tolerance {
            self.cache.clear();
            self.cache_tolerance = tolerance;
        }
        let tolerance = self.cache_tolerance;
        let mut mesh = Mesh::default();
        let mut image_vertices: Vec<ImageVertex> = Vec::new();
        let mut commands: Vec<Command> = Vec::new();
        let mut used = std::collections::HashSet::new();
        for item in timeline.at(time) {
            match item {
                Item::Shape { shape, id } => {
                    let start = mesh.indices.len() as u32;
                    match id {
                        Some(id) => {
                            used.insert(id);
                            let cached = self.cache.entry(id).or_insert_with(|| tessellate(&shape, tolerance));
                            append(&mut mesh, cached);
                        }
                        None => append(&mut mesh, &tessellate(&shape, tolerance)),
                    }
                    let end = mesh.indices.len() as u32;
                    match commands.last_mut() {
                        Some(Command::Shapes(range)) if range.end == start => range.end = end,
                        _ => commands.push(Command::Shapes(start..end)),
                    }
                }
                Item::Image(placement) => {
                    if placement.image >= self.images.len() {
                        continue;
                    }
                    let first = quad(&mut image_vertices, placement.rect, placement.opacity);
                    commands.push(Command::Image { image: placement.image, first });
                }
                Item::Live { viewer, rect } => {
                    if self.live.contains_key(&viewer) {
                        let first = quad(&mut image_vertices, rect, 1.0);
                        commands.push(Command::Live { viewer, first });
                    }
                }
            }
        }
        self.cache.retain(|id, _| used.contains(id));

        let frame = FrameUniform { scale: [2.0 / timeline.frame[0], 2.0 / timeline.frame[1]], srgb: 0.0, _pad: 0.0 };
        queue.write_buffer(&self.frame_buffer, 0, bytemuck::bytes_of(&frame));
        let srgb = if format.is_srgb() { 1.0 } else { 0.0 };
        queue.write_buffer(&self.blit_frame_buffer, 0, bytemuck::bytes_of(&FrameUniform { srgb, ..frame }));
        let buffer = |label, contents: &[u8], usage| {
            // wgpu rejects empty buffers
            let contents = if contents.is_empty() { &[0u8; 32][..] } else { contents };
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage })
        };
        let vertices = buffer("shape vertices", bytemuck::cast_slice(&mesh.vertices), wgpu::BufferUsages::VERTEX);
        let indices = buffer("shape indices", bytemuck::cast_slice(&mesh.indices), wgpu::BufferUsages::INDEX);
        let quads = buffer("image vertices", bytemuck::cast_slice(&image_vertices), wgpu::BufferUsages::VERTEX);

        let blit = self.blit_pipeline(device, format).clone();
        let targets = self.targets.as_ref().unwrap();
        let [r, g, b, a] = timeline.background.map(f64::from);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("slides") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("slide"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.msaa,
                    depth_slice: None,
                    resolve_target: Some(&targets.resolved),
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }), store: wgpu::StoreOp::Discard },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.frame_group, &[]);
            for command in &commands {
                match command {
                    Command::Shapes(range) => {
                        pass.set_pipeline(&self.shape_pipeline);
                        pass.set_vertex_buffer(0, vertices.slice(..));
                        pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
                        pass.draw_indexed(range.clone(), 0, 0..1);
                    }
                    Command::Image { image, first } => {
                        pass.set_pipeline(&self.image_pipeline);
                        pass.set_bind_group(1, &self.images[*image], &[]);
                        pass.set_vertex_buffer(0, quads.slice(..));
                        pass.draw(*first..*first + 6, 0..1);
                    }
                    Command::Live { viewer, first } => {
                        pass.set_pipeline(&self.image_pipeline);
                        pass.set_bind_group(1, &self.live[viewer], &[]);
                        pass.set_vertex_buffer(0, quads.slice(..));
                        pass.draw(*first..*first + 6, 0..1);
                    }
                }
            }
        }
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("blit"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: target,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::BLACK), store: wgpu::StoreOp::Store },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_viewport(viewport[0], viewport[1], viewport[2], viewport[3], 0.0, 1.0);
            pass.set_pipeline(&blit);
            pass.set_bind_group(0, &self.blit_frame_group, &[]);
            pass.set_bind_group(1, &targets.blit_group, &[]);
            pass.draw(0..3, 0..1);
        }
        queue.submit([encoder.finish()]);
    }
}

/// The largest rectangle of the frame's aspect ratio centered in `size`, whole pixels.
pub fn letterbox(frame: [f32; 2], size: (u32, u32)) -> [f32; 4] {
    let (w, h) = (size.0.max(1) as f32, size.1.max(1) as f32);
    let scale = (w / frame[0]).min(h / frame[1]);
    let (vw, vh) = ((frame[0] * scale).round().max(1.0), (frame[1] * scale).round().max(1.0));
    [((w - vw) / 2.0).floor(), ((h - vh) / 2.0).floor(), vw, vh]
}

/// A textured rectangle (center, width, height); returns its first vertex.
fn quad(vertices: &mut Vec<ImageVertex>, rect: [f32; 4], opacity: f32) -> u32 {
    let [cx, cy, w, h] = rect;
    let (x0, x1, y0, y1) = (cx - w / 2.0, cx + w / 2.0, cy - h / 2.0, cy + h / 2.0);
    let first = vertices.len() as u32;
    let v = |x, y, u, v| ImageVertex { position: [x, y], uv: [u, v], opacity };
    vertices.extend([
        v(x0, y1, 0.0, 0.0),
        v(x0, y0, 0.0, 1.0),
        v(x1, y1, 1.0, 0.0),
        v(x1, y1, 1.0, 0.0),
        v(x0, y0, 0.0, 1.0),
        v(x1, y0, 1.0, 1.0),
    ]);
    first
}

fn append(mesh: &mut Mesh, part: &Mesh) {
    let base = mesh.vertices.len() as u32;
    mesh.vertices.extend_from_slice(&part.vertices);
    mesh.indices.extend(part.indices.iter().map(|i| i + base));
}

fn build_path(points: &[[f32; 2]]) -> (Path, bool) {
    let mut builder = Path::builder();
    let mut open = false;
    let mut start = [0.0f32; 2];
    let mut last = [f32::NAN; 2];
    let mut any = false;
    for c in points.chunks_exact(4) {
        if c[0] != last {
            if open {
                builder.end(last == start);
            }
            builder.begin(point(c[0][0], c[0][1]));
            open = true;
            start = c[0];
        }
        builder.cubic_bezier_to(point(c[1][0], c[1][1]), point(c[2][0], c[2][1]), point(c[3][0], c[3][1]));
        last = c[3];
        any = true;
    }
    if open {
        builder.end(last == start);
    }
    (builder.build(), any)
}

fn tessellate(shape: &Shape, tolerance: f32) -> Mesh {
    let mut buffers: VertexBuffers<ShapeVertex, u32> = VertexBuffers::new();
    let (path, any) = build_path(&shape.points);
    if !any {
        return Mesh::default();
    }
    let style = &shape.style;
    if style.fill[3] > 0.0 {
        let color = style.fill;
        let _ = FillTessellator::new().tessellate_path(
            &path,
            &FillOptions::tolerance(tolerance).with_fill_rule(FillRule::NonZero),
            &mut BuffersBuilder::new(&mut buffers, |v: FillVertex| ShapeVertex { position: v.position().to_array(), color }),
        );
    }
    if style.stroke[3] > 0.0 && style.width > 0.0 {
        let color = style.stroke;
        let join = match style.join {
            Join::Miter => LineJoin::Miter,
            Join::Round => LineJoin::Round,
            Join::Bevel => LineJoin::Bevel,
        };
        let cap = match style.cap {
            Cap::Butt => LineCap::Butt,
            Cap::Round => LineCap::Round,
            Cap::Square => LineCap::Square,
        };
        let options = StrokeOptions::tolerance(tolerance)
            .with_line_width(style.width)
            .with_line_join(join)
            .with_miter_limit(10.0)
            .with_line_cap(cap);
        let _ = StrokeTessellator::new().tessellate_path(
            &path,
            &options,
            &mut BuffersBuilder::new(&mut buffers, |v: StrokeVertex| ShapeVertex { position: v.position().to_array(), color }),
        );
    }
    Mesh { vertices: buffers.vertices, indices: buffers.indices }
}

/// The image and its halvings down to 1 pixel (2x2 box filter), for minification.
fn mip_levels(image: &Image) -> Vec<Vec<u8>> {
    let mut levels = vec![image.pixels.clone()];
    let (mut w, mut h) = (image.width as usize, image.height as usize);
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let previous = levels.last().unwrap();
        let mut next = vec![0u8; nw * nh * 4];
        for y in 0..nh {
            for x in 0..nw {
                for c in 0..4 {
                    let mut sum = 0u32;
                    for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                        let (sx, sy) = ((2 * x + dx).min(w - 1), (2 * y + dy).min(h - 1));
                        sum += previous[(sy * w + sx) * 4 + c] as u32;
                    }
                    next[(y * nw + x) * 4 + c] = (sum / 4) as u8;
                }
            }
        }
        levels.push(next);
        (w, h) = (nw, nh);
    }
    levels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_keeps_the_aspect_ratio() {
        assert_eq!(letterbox([16.0, 9.0], (1600, 900)), [0.0, 0.0, 1600.0, 900.0]);
        assert_eq!(letterbox([16.0, 9.0], (1600, 1000)), [0.0, 50.0, 1600.0, 900.0]);
        assert_eq!(letterbox([16.0, 9.0], (2000, 900)), [200.0, 0.0, 1600.0, 900.0]);
    }

    #[test]
    fn mip_levels_halve() {
        let image = Image { width: 4, height: 2, pixels: vec![255; 32] };
        let levels = mip_levels(&image);
        assert_eq!(levels.iter().map(Vec::len).collect::<Vec<_>>(), vec![32, 8, 4]);
    }
}
