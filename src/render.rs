//! wgpu device, pipelines and the draw of one frame into a texture view
//! (a window surface or an offscreen texture).
//!
//! Draw order: opaque geometry, trails (blended), the ghost (a depth-only pass,
//! then its color where it is the front surface, so it reads as one translucent
//! shape), and the overlay on top of everything.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use wgpu::util::DeviceExt;

use crate::camera::Camera;
use crate::scene::{Geometry, LatheInstance, SphereInstance};
use crate::surface::MeshVertex;

pub const SAMPLES: u32 = 4;
/// Between the raw table colors (1.0, washed out) and an exact sRGB decode (2.2).
pub const DEFAULT_COLOR_GAMMA: f32 = 1.6;
/// Outline in flat shading: this many angstrom wide, but at least this many pixels.
const OUTLINE_WIDTH: f32 = 0.04;
const OUTLINE_MIN_PIXELS: f32 = 1.2;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

#[derive(Clone)]
pub struct Gpu {
    pub instance: wgpu::Instance,
    pub adapter: wgpu::Adapter,
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
}

impl Gpu {
    /// The process's GPU, shared by every viewer and deck so that they can draw
    /// into each other's textures (a deck shows live viewers).
    pub fn shared() -> Result<Self, String> {
        static SHARED: std::sync::OnceLock<Result<Gpu, String>> = std::sync::OnceLock::new();
        SHARED.get_or_init(Gpu::new).clone()
    }

    pub fn new() -> Result<Self, String> {
        // WGPU_BACKEND (e.g. vulkan, gl, metal) overrides the backends
        let mut descriptor = wgpu::InstanceDescriptor::new_without_display_handle_from_env();
        if cfg!(target_os = "macos") && std::env::var_os("WGPU_BACKEND").is_none() {
            descriptor.backends = wgpu::Backends::METAL; // not Vulkan through MoltenVK
        }
        let instance = wgpu::Instance::new(descriptor);
        let request = |force_fallback_adapter| {
            pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter,
                ..Default::default()
            }))
        };
        // without a GPU: a software adapter (e.g. Mesa's lavapipe), if there is one
        let adapter = request(false).or_else(|_| request(true)).map_err(|e| format!("no GPU adapter: {e}"))?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("landau"),
            ..Default::default()
        }))
        .map_err(|e| format!("no GPU device: {e}"))?;
        Ok(Self { instance, adapter, device, queue })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view: [[f32; 4]; 4],
    proj: [[f32; 4]; 4],
    viewport: [f32; 4],
    light: [f32; 4],
    shading: [f32; 4],
    outline: [f32; 4],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Shader {
    Spheres,
    Lathe,
    /// plain triangle lists (not instanced)
    Mesh,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Variant {
    Opaque,
    /// alpha blended, still writing depth (thin trails)
    Blended,
    GhostDepth,
    GhostColor,
    /// blended over everything, no depth test
    Overlay,
}

/// Instance buffers, one per kind of instance and how it is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Slot {
    Spheres,
    Tubes,
    Cones,
    Trails,
    GhostSpheres,
    GhostTubes,
    Overlay,
    Surface,
    GhostSurface,
}

struct Targets {
    format: wgpu::TextureFormat,
    size: (u32, u32),
    color: wgpu::TextureView,
    depth: wgpu::TextureView,
}

/// A vertex buffer that grows as needed.
#[derive(Default)]
struct Instances {
    buffer: Option<wgpu::Buffer>,
    count: u32,
}

impl Instances {
    fn upload<T: Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, items: &[T]) {
        self.count = items.len() as u32;
        let bytes: &[u8] = bytemuck::cast_slice(items);
        if bytes.is_empty() {
            return;
        }
        match &self.buffer {
            Some(buffer) if buffer.size() >= bytes.len() as u64 => queue.write_buffer(buffer, 0, bytes),
            _ => {
                self.buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("instances"),
                    contents: bytes,
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }));
            }
        }
    }
}

/// A unit mesh for the lathe shader: (position, normal) per vertex.
struct Mesh {
    buffer: wgpu::Buffer,
    vertices: u32,
}

impl Mesh {
    fn new(device: &wgpu::Device, label: &str, vertices: &[[f32; 6]]) -> Self {
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        Self { buffer, vertices: vertices.len() as u32 }
    }
}

pub struct Renderer {
    globals: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    layout: wgpu::PipelineLayout,
    sphere_shader: wgpu::ShaderModule,
    lathe_shader: wgpu::ShaderModule,
    mesh_shader: wgpu::ShaderModule,
    pipelines: HashMap<(wgpu::TextureFormat, Shader, Variant), wgpu::RenderPipeline>,
    targets: Option<Targets>,
    tube: Mesh,
    thin_tube: Mesh,
    cone: Mesh,
    slots: HashMap<Slot, Instances>,
    pub background: [f32; 4],
    /// decoding gamma of the colors (see linear() in the shaders)
    pub color_gamma: f32,
    /// flat colors with an outline instead of lighting
    pub flat: bool,
    pub outline: [f32; 4],
}

impl Renderer {
    pub fn new(device: &wgpu::Device) -> Self {
        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
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
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("landau"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        Self {
            globals,
            bind_group,
            layout,
            sphere_shader: device.create_shader_module(wgpu::include_wgsl!("shaders/spheres.wgsl")),
            lathe_shader: device.create_shader_module(wgpu::include_wgsl!("shaders/lathe.wgsl")),
            mesh_shader: device.create_shader_module(wgpu::include_wgsl!("shaders/mesh.wgsl")),
            pipelines: HashMap::new(),
            targets: None,
            tube: Mesh::new(device, "tube", &tube_mesh(24)),
            thin_tube: Mesh::new(device, "thin tube", &tube_mesh(8)),
            cone: Mesh::new(device, "cone", &cone_mesh(24)),
            slots: HashMap::new(),
            background: [1.0, 1.0, 1.0, 1.0],
            color_gamma: DEFAULT_COLOR_GAMMA,
            flat: false,
            outline: [0.0, 0.0, 0.0, 1.0],
        }
    }

    fn upload_slot<T: Pod>(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, slot: Slot, items: &[T]) {
        self.slots.entry(slot).or_default().upload(device, queue, items);
    }

    /// Upload the instances that change with the frame.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, geometry: &Geometry) {
        self.upload_slot(device, queue, Slot::Spheres, &geometry.spheres);
        self.upload_slot(device, queue, Slot::Tubes, &geometry.tubes);
        self.upload_slot(device, queue, Slot::Cones, &geometry.cones);
        self.upload_slot(device, queue, Slot::GhostSpheres, &geometry.ghost_spheres);
        self.upload_slot(device, queue, Slot::GhostTubes, &geometry.ghost_tubes);
        self.upload_slot(device, queue, Slot::Overlay, &geometry.overlay);
        self.upload_slot(device, queue, Slot::Surface, &geometry.surface);
        self.upload_slot(device, queue, Slot::GhostSurface, &geometry.ghost_surface);
    }

    pub fn upload_trails(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, trails: &[LatheInstance]) {
        self.upload_slot(device, queue, Slot::Trails, trails);
    }

    /// Draw the uploaded instances into `target`, of `size` pixels; within
    /// `viewport` (x, y, width, height in pixels) if given, else all of it.
    #[allow(clippy::too_many_arguments)]
    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        target: &wgpu::TextureView,
        format: wgpu::TextureFormat,
        size: (u32, u32),
        viewport: Option<[f32; 4]>,
        camera: &Camera,
    ) {
        if !matches!(&self.targets, Some(t) if t.format == format && t.size == size) {
            self.targets = Some(create_targets(device, format, size));
        }
        // (slot, shader, mesh, variant) in draw order
        let passes: [(Slot, Shader, Option<&Mesh>, Variant); 12] = [
            (Slot::Spheres, Shader::Spheres, None, Variant::Opaque),
            (Slot::Tubes, Shader::Lathe, Some(&self.tube), Variant::Opaque),
            (Slot::Cones, Shader::Lathe, Some(&self.cone), Variant::Opaque),
            (Slot::Surface, Shader::Mesh, None, Variant::Opaque),
            (Slot::Trails, Shader::Lathe, Some(&self.thin_tube), Variant::Blended),
            (Slot::GhostSpheres, Shader::Spheres, None, Variant::GhostDepth),
            (Slot::GhostTubes, Shader::Lathe, Some(&self.tube), Variant::GhostDepth),
            (Slot::GhostSpheres, Shader::Spheres, None, Variant::GhostColor),
            (Slot::GhostTubes, Shader::Lathe, Some(&self.tube), Variant::GhostColor),
            // the translucent surface over everything else, its front layer only
            (Slot::GhostSurface, Shader::Mesh, None, Variant::GhostDepth),
            (Slot::GhostSurface, Shader::Mesh, None, Variant::GhostColor),
            (Slot::Overlay, Shader::Lathe, Some(&self.thin_tube), Variant::Overlay),
        ];
        for &(_, shader, _, variant) in &passes {
            if !self.pipelines.contains_key(&(format, shader, variant)) {
                let pipeline = self.create_pipeline(device, format, shader, variant);
                self.pipelines.insert((format, shader, variant), pipeline);
            }
        }
        let targets = self.targets.as_ref().unwrap();

        let viewport = viewport.unwrap_or([0.0, 0.0, size.0 as f32, size.1 as f32]);
        let aspect = viewport[2] / viewport[3].max(1.0);
        let light = Vec3::new(-0.35, 0.45, 1.0).normalize();
        let globals = Globals {
            view: camera.view().to_cols_array_2d(),
            proj: camera.projection(aspect).to_cols_array_2d(),
            viewport: [viewport[2], viewport[3], 0.0, 0.0],
            light: light.extend(self.color_gamma).to_array(),
            shading: [if self.flat { 1.0 } else { 0.0 }, OUTLINE_WIDTH, OUTLINE_MIN_PIXELS, 0.0],
            outline: self.outline,
        };
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));

        let [r, g, b, a] = self.background.map(f64::from);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("scene"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &targets.color,
                    depth_slice: None,
                    resolve_target: Some(target),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r, g, b, a }),
                        store: wgpu::StoreOp::Discard,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &targets.depth,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_bind_group(0, &self.bind_group, &[]);
            let [x, y, width, height] = viewport;
            pass.set_viewport(x, y, width.max(1.0), height.max(1.0), 0.0, 1.0);
            for &(slot, shader, mesh, variant) in &passes {
                let Some(Instances { buffer: Some(buffer), count: count @ 1.. }) = self.slots.get(&slot) else {
                    continue;
                };
                pass.set_pipeline(&self.pipelines[&(format, shader, variant)]);
                match (shader, mesh) {
                    (_, Some(mesh)) => {
                        pass.set_vertex_buffer(0, mesh.buffer.slice(..));
                        pass.set_vertex_buffer(1, buffer.slice(..));
                        pass.draw(0..mesh.vertices, 0..*count);
                    }
                    (Shader::Mesh, None) => {
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..*count, 0..1);
                    }
                    (_, None) => {
                        pass.set_vertex_buffer(0, buffer.slice(..));
                        pass.draw(0..4, 0..*count);
                    }
                }
            }
        }
        queue.submit([encoder.finish()]);
    }

    fn create_pipeline(
        &self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        shader: Shader,
        variant: Variant,
    ) -> wgpu::RenderPipeline {
        let (depth_write, compare) = match variant {
            Variant::Opaque | Variant::Blended | Variant::GhostDepth => (true, wgpu::CompareFunction::Less),
            Variant::GhostColor => (false, wgpu::CompareFunction::LessEqual),
            Variant::Overlay => (false, wgpu::CompareFunction::Always),
        };
        let blend = match variant {
            Variant::Blended | Variant::GhostColor | Variant::Overlay => Some(wgpu::BlendState::ALPHA_BLENDING),
            Variant::Opaque | Variant::GhostDepth => None,
        };
        let write_mask = match variant {
            Variant::GhostDepth => wgpu::ColorWrites::empty(),
            _ => wgpu::ColorWrites::ALL,
        };
        // spheres smooth their silhouettes through coverage where they are not blended
        let alpha_to_coverage = shader == Shader::Spheres && blend.is_none();

        let attribute = |shader_location, offset, format| wgpu::VertexAttribute { format, offset, shader_location };
        let sphere_attributes = [
            attribute(0, 0, wgpu::VertexFormat::Float32x3),
            attribute(1, 12, wgpu::VertexFormat::Float32),
            attribute(2, 16, wgpu::VertexFormat::Float32x4),
        ];
        let mesh_attributes =
            [attribute(0, 0, wgpu::VertexFormat::Float32x3), attribute(1, 12, wgpu::VertexFormat::Float32x3)];
        let lathe_attributes = [
            attribute(2, 0, wgpu::VertexFormat::Float32x3),
            attribute(3, 12, wgpu::VertexFormat::Float32),
            attribute(4, 16, wgpu::VertexFormat::Float32x3),
            attribute(5, 32, wgpu::VertexFormat::Float32x4),
            attribute(6, 48, wgpu::VertexFormat::Float32x4),
        ];
        let sphere_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SphereInstance>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &sphere_attributes,
        })];
        let lathe_buffers = [
            Some(wgpu::VertexBufferLayout {
                array_stride: 24,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &mesh_attributes,
            }),
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<LatheInstance>() as u64,
                step_mode: wgpu::VertexStepMode::Instance,
                attributes: &lathe_attributes,
            }),
        ];
        let surface_attributes = [
            attribute(0, 0, wgpu::VertexFormat::Float32x3),
            attribute(1, 12, wgpu::VertexFormat::Float32x3),
            attribute(2, 24, wgpu::VertexFormat::Float32x4),
        ];
        let surface_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<MeshVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &surface_attributes,
        })];
        let (module, buffers, topology, cull_mode): (_, &[_], _, _) = match shader {
            Shader::Mesh => (&self.mesh_shader, &surface_buffers, wgpu::PrimitiveTopology::TriangleList, None),
            Shader::Spheres => (&self.sphere_shader, &sphere_buffers, wgpu::PrimitiveTopology::TriangleStrip, None),
            Shader::Lathe => {
                (&self.lathe_shader, &lathe_buffers, wgpu::PrimitiveTopology::TriangleList, Some(wgpu::Face::Back))
            }
        };
        let targets = [Some(wgpu::ColorTargetState { format, blend, write_mask })];
        let outlines = match variant {
            Variant::Blended | Variant::Overlay => 0.0,
            _ => 1.0,
        };
        let constants = [("OUTLINES", outlines)];
        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(&format!("{shader:?} {variant:?}")),
            layout: Some(&self.layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers,
            },
            primitive: wgpu::PrimitiveState { topology, cull_mode, ..Default::default() },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(depth_write),
                depth_compare: Some(compare),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState { count: SAMPLES, mask: !0, alpha_to_coverage_enabled: alpha_to_coverage },
            fragment: Some(wgpu::FragmentState {
                module,
                entry_point: Some("fs"),
                compilation_options: wgpu::PipelineCompilationOptions { constants: &constants, ..Default::default() },
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}

fn create_targets(device: &wgpu::Device, format: wgpu::TextureFormat, size: (u32, u32)) -> Targets {
    let texture = |label, format| {
        device
            .create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d { width: size.0.max(1), height: size.1.max(1), depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: SAMPLES,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            })
            .create_view(&Default::default())
    };
    Targets { format, size, color: texture("color", format), depth: texture("depth", DEPTH_FORMAT) }
}

fn ring(sides: u32, i: u32) -> (f32, f32) {
    let angle = std::f32::consts::TAU * i as f32 / sides as f32;
    (angle.cos(), angle.sin())
}

/// Open unit tube along z, counter-clockwise from outside.
fn tube_mesh(sides: u32) -> Vec<[f32; 6]> {
    (0..sides)
        .flat_map(|i| {
            let ((c0, s0), (c1, s1)) = (ring(sides, i), ring(sides, i + 1));
            let a0 = [c0, s0, 0.0, c0, s0, 0.0];
            let a1 = [c1, s1, 0.0, c1, s1, 0.0];
            let b0 = [c0, s0, 1.0, c0, s0, 0.0];
            let b1 = [c1, s1, 1.0, c1, s1, 0.0];
            [a0, a1, b1, a0, b1, b0]
        })
        .collect()
}

/// Unit cone (base radius 1 at z = 0, apex at z = 1) with its base disk.
fn cone_mesh(sides: u32) -> Vec<[f32; 6]> {
    (0..sides)
        .flat_map(|i| {
            let ((c0, s0), (c1, s1)) = (ring(sides, i), ring(sides, i + 1));
            let (cm, sm) = ring(2 * sides, 2 * i + 1);
            // x² + y² = (1 - z)² has the gradient (cos, sin, 1) on the side
            let side = [[c0, s0, 0.0, c0, s0, 1.0], [c1, s1, 0.0, c1, s1, 1.0], [0.0, 0.0, 1.0, cm, sm, 1.0]];
            let base = [[0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [c1, s1, 0.0, 0.0, 0.0, -1.0], [c0, s0, 0.0, 0.0, 0.0, -1.0]];
            side.into_iter().chain(base)
        })
        .collect()
}

/// An offscreen target and its readback buffer, kept while the size stays the same.
pub struct Offscreen {
    size: (u32, u32),
    format: wgpu::TextureFormat,
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    buffer: wgpu::Buffer,
    padded_row: u32,
}

impl Offscreen {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

    pub fn with_format(device: &wgpu::Device, size: (u32, u32), format: wgpu::TextureFormat) -> Self {
        let size = (size.0.max(1), size.1.max(1));
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen"),
            size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&Default::default());
        let padded_row = (4 * size.0).div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT) * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_row * size.1) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Self { size, format, texture, view, buffer, padded_row }
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn view(&self) -> &wgpu::TextureView {
        &self.view
    }

    /// Copy the texture back: 4 bytes per pixel, rows top to bottom, in the texture's format.
    pub fn read(&self, gpu: &Gpu) -> Vec<u8> {
        let mut encoder = gpu.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self.texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_row),
                    rows_per_image: Some(self.size.1),
                },
            },
            wgpu::Extent3d { width: self.size.0, height: self.size.1, depth_or_array_layers: 1 },
        );
        gpu.queue.submit([encoder.finish()]);
        let slice = self.buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        gpu.device.poll(wgpu::PollType::wait_indefinitely()).expect("GPU readback failed");
        let row = (4 * self.size.0) as usize;
        let mut pixels = Vec::with_capacity(row * self.size.1 as usize);
        {
            let mapped = slice.get_mapped_range().expect("GPU readback failed");
            for line in mapped.chunks(self.padded_row as usize) {
                pixels.extend_from_slice(&line[..row]);
            }
        }
        self.buffer.unmap();
        pixels
    }
}

/// Render offscreen and read the pixels back as RGBA8 rows (sRGB); `cache`
/// keeps the target between calls of the same size.
pub fn render_image(
    gpu: &Gpu,
    renderer: &mut Renderer,
    cache: &mut Option<Offscreen>,
    size: (u32, u32),
    camera: &Camera,
) -> Vec<u8> {
    let size = (size.0.max(1), size.1.max(1));
    if !matches!(cache, Some(o) if o.size == size && o.format == Offscreen::FORMAT) {
        *cache = Some(Offscreen::with_format(&gpu.device, size, Offscreen::FORMAT));
    }
    let target = cache.as_ref().unwrap();
    renderer.draw(&gpu.device, &gpu.queue, &target.view, Offscreen::FORMAT, size, None, camera);
    target.read(gpu)
}
