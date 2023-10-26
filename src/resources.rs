use std::{
    iter, mem,
    num::{NonZeroU32, NonZeroU64},
    ops::Range,
    sync::OnceLock,
};
use wgpu::util::DeviceExt;

use bytemuck::{Pod, Zeroable};
use glam::{Mat4, UVec2, Vec2, Vec4};

use crate::{
    asset::{BoundingSphere, DirectionalLight, Scene, Transform},
    camera::Camera,
    context::Context,
    temporal_resolve,
};

#[repr(C)]
#[derive(Debug, Copy, Clone, Zeroable, Pod)]
pub struct Consts {
    pub camera_pos: Vec4,
    pub camera_front: Vec4,
    pub proj_view: Mat4,
    pub proj: Mat4,
    pub prev_proj_view: Mat4,
    pub inverse_proj_view: Mat4,
    pub sun: DirectionalLight,
    pub frustrum_z_planes: Vec2,
    pub surface_size: UVec2,
    pub jitter: Vec2,
    pub frame_index: u32,
    pub camera_fov: f32,
}

impl Consts {
    pub fn new(camera: &Camera, context: &Context, prev: Option<Consts>) -> Self {
        let proj_view = camera.proj_view();
        let prev_proj_view = prev.map(|prev| prev.proj_view).unwrap_or(proj_view);

        let frame_index = prev.map(|consts| consts.frame_index + 1).unwrap_or(0);

        let surface_size = UVec2 {
            x: context.surface_size.width,
            y: context.surface_size.height,
        };

        let jitter = temporal_resolve::jitter(frame_index as usize, surface_size);

        let sun = DirectionalLight {
            direction: Vec4::new(0.0, 1.0, 0.1, 1.0).normalize(),
            irradiance: Vec4::splat(1.0),
        };

        Self {
            camera_pos: camera.pos.extend(1.0),
            camera_front: camera.front.extend(1.0),
            camera_fov: camera.fov,
            inverse_proj_view: proj_view.inverse(),
            proj_view,
            proj: camera.proj,
            prev_proj_view,
            sun,
            frustrum_z_planes: Vec2 {
                x: camera.z_near,
                y: camera.z_far,
            },
            surface_size,
            frame_index,
            jitter,
        }
    }
}

pub struct ConstState {
    pub const_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
}

impl ConstState {
    pub fn new(context: &Context) -> Self {
        let const_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::UNIFORM,
            size: mem::size_of::<Consts>() as wgpu::BufferAddress,
            mapped_at_creation: false,
            label: Some("constant buffer"),
        });

        let texture_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("texture sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mipmap_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            anisotropy_clamp: 16,
            ..Default::default()
        });

        let linear_sampler = context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear sampler"),
            mipmap_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            ..Default::default()
        });

        let const_buffer_size = NonZeroU64::new(mem::size_of::<Consts>() as u64);

        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("const state"),
                layout: Self::bind_group_layout(&context),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &const_buffer,
                            size: const_buffer_size,
                            offset: 0,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&texture_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Sampler(&linear_sampler),
                    },
                ],
            });

        Self {
            const_buffer,
            bind_group,
        }
    }

    pub fn bind_group_layout(context: &Context) -> &wgpu::BindGroupLayout {
        static LAYOUT: OnceLock<wgpu::BindGroupLayout> = OnceLock::new();

        LAYOUT.get_or_init(|| {
            let const_buffer_size = NonZeroU64::new(mem::size_of::<Consts>() as u64);

            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("const state"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::all(),
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: const_buffer_size,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::all(),
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::all(),
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                            count: None,
                        },
                    ],
                })
        })
    }
}

pub struct RenderTarget {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub format: wgpu::TextureFormat,
}

impl RenderTarget {
    pub fn new(
        context: &Context,
        label: &str,
        format: wgpu::TextureFormat,
        usage: wgpu::TextureUsages,
    ) -> Self {
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            dimension: wgpu::TextureDimension::D2,
            size: context.surface_size,
            mip_level_count: 1,
            sample_count: 1,
            view_formats: &[],
            format,
            usage,
        });

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some(label),
            ..Default::default()
        });

        Self {
            texture,
            view,
            format,
        }
    }
}

pub struct Skybox {
    pub texture: wgpu::Texture,
    pub array_view: wgpu::TextureView,
    pub cube_view: wgpu::TextureView,
}

impl Skybox {
    pub fn new(context: &Context) -> Self {
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("skybox"),
            size: SKYBOX_SIZE,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SKYBOX_FORMAT,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let array_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("skybox array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let cube_view = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("skybox cube"),
            dimension: Some(wgpu::TextureViewDimension::Cube),
            ..Default::default()
        });

        Self {
            texture,
            array_view,
            cube_view,
        }
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, Zeroable, Pod)]
pub struct ShadowCascade {
    pub proj_view: Mat4,
    pub split: f32,
    pub split_depth: f32,
    pub padding: [u32; 2],
}

pub struct ShadowCascades {
    pub cascades: Vec<wgpu::TextureView>,
    pub cascade_array: wgpu::TextureView,
    pub cascade_info: wgpu::Buffer,
}

impl ShadowCascades {
    pub fn new(context: &Context) -> Self {
        let texture = context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("shadow cascade"),
            size: wgpu::Extent3d {
                width: SHADOW_CASCADE_SIZE,
                height: SHADOW_CASCADE_SIZE,
                depth_or_array_layers: SHADOW_CASCADE_COUNT as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: SHADOW_CASCADE_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let cascades = (0..SHADOW_CASCADE_COUNT)
            .map(|layer| {
                texture.create_view(&wgpu::TextureViewDescriptor {
                    label: Some("shadow cascade"),
                    base_array_layer: layer as u32,
                    array_layer_count: Some(1),
                    dimension: Some(wgpu::TextureViewDimension::D2),
                    ..Default::default()
                })
            })
            .collect();

        let cascade_array = texture.create_view(&wgpu::TextureViewDescriptor {
            label: Some("shadow cascade array"),
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });

        let cascade_info = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cascade buffer"),
            size: mem::size_of::<ShadowCascade>() as u64 * SHADOW_CASCADE_COUNT as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            cascades,
            cascade_array,
            cascade_info,
        }
    }
}

pub struct RenderState {
    pub visibility: RenderTarget,
    pub depth: RenderTarget,
    pub color: RenderTarget,
    pub color_accum: RenderTarget,
    pub post: RenderTarget,
}

impl RenderState {
    pub fn new(context: &Context) -> Self {
        Self {
            visibility: RenderTarget::new(
                context,
                "visibility buffer",
                VISIBILITY_BUFFER_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            depth: RenderTarget::new(
                context,
                "depth buffer",
                DEPTH_BUFFER_FORMAT,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            color: RenderTarget::new(
                context,
                "color buffer",
                COLOR_BUFFER_FORMAT,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::STORAGE_BINDING,
            ),
            color_accum: RenderTarget::new(
                context,
                "color accum buffer",
                COLOR_BUFFER_FORMAT,
                wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            ),
            post: RenderTarget::new(
                context,
                "post buffer",
                COLOR_BUFFER_FORMAT,
                wgpu::TextureUsages::STORAGE_BINDING
                    | wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            ),
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
struct Primitive {
    transform: Mat4,
    inverse_transpose_transform: Mat4,
    bounding_sphere: BoundingSphere,
}

pub struct PrimitiveDrawInfo {
    pub bounding_sphere: BoundingSphere,
    pub indices: Range<u32>,
    pub material: u32,
}

pub struct SceneState {
    pub primitive_draw_infos: Vec<PrimitiveDrawInfo>,
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl SceneState {
    pub fn new(context: &Context, scene: &Scene) -> Self {
        let mut primitives = Vec::new();
        let mut primitive_draw_infos = Vec::new();

        scene.visit_instances(|instance, parent_transform| {
            let mut transform: Mat4 = instance.transform.into();
            if let Some(parent_transform) = parent_transform.copied() {
                transform = parent_transform * transform
            }

            if let Some(mesh) = instance.mesh {
                for primitive in &scene.meshes[mesh as usize].primitives {
                    primitives.push(Primitive {
                        inverse_transpose_transform: transform.inverse().transpose(),
                        bounding_sphere: primitive.bounding_sphere,
                        transform,
                    });
                    primitive_draw_infos.push(PrimitiveDrawInfo {
                        indices: primitive.indices.clone(),
                        material: primitive.material,
                        bounding_sphere: primitive
                            .bounding_sphere
                            .transformed(Transform::from(transform)),
                    });
                }
            }

            transform
        });

        let primitive_buffer =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("primitive buffer"),
                    usage: wgpu::BufferUsages::STORAGE,
                    contents: bytemuck::cast_slice(&primitives),
                });

        let material_buffer =
            context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("material buffer"),
                    usage: wgpu::BufferUsages::STORAGE,
                    contents: bytemuck::cast_slice(&scene.materials),
                });

        let index_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("index buffer"),
                usage: wgpu::BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(&scene.indices),
            });

        let vertex_buffer = context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("vertex buffer"),
                usage: wgpu::BufferUsages::STORAGE,
                contents: bytemuck::cast_slice(&scene.vertices),
            });

        let textures: Vec<_> = scene
            .textures
            .iter()
            .map(|texture| {
                context
                    .device
                    .create_texture_with_data(
                        &context.queue,
                        &wgpu::TextureDescriptor {
                            dimension: wgpu::TextureDimension::D2,
                            usage: wgpu::TextureUsages::TEXTURE_BINDING,
                            mip_level_count: texture.mip_level_count,
                            size: texture.extent,
                            format: texture.format,
                            view_formats: &[],
                            sample_count: 1,
                            label: None,
                        },
                        &texture.mips,
                    )
                    .create_view(&wgpu::TextureViewDescriptor::default())
            })
            .collect();

        let layout_entries: Vec<_> = (0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::all(),
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .chain(iter::once(wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::all(),
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: NonZeroU32::new(textures.len() as u32),
            }))
            .collect();

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("scene state"),
                    entries: &layout_entries,
                });

        let buffers = [
            &primitive_buffer,
            &material_buffer,
            &index_buffer,
            &vertex_buffer,
        ];

        let texture_refs = textures.iter().collect::<Vec<_>>();

        let bind_group_entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    offset: 0,
                    size: None,
                    buffer,
                }),
            })
            .chain(iter::once(wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureViewArray(&texture_refs),
            }))
            .collect();

        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("scene state"),
                layout: &bind_group_layout,
                entries: &bind_group_entries,
            });

        SceneState {
            bind_group,
            bind_group_layout,
            primitive_draw_infos,
        }
    }
}

pub const DEPTH_BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
pub const VISIBILITY_BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
pub const COLOR_BUFFER_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

pub const SKYBOX_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const SKYBOX_SIZE: wgpu::Extent3d = wgpu::Extent3d {
    width: 64,
    height: 64,
    depth_or_array_layers: 6,
};

pub const SHADOW_CASCADE_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth16Unorm;
pub const SHADOW_CASCADE_SIZE: u32 = 2048;
pub const SHADOW_CASCADE_COUNT: usize = 1;
