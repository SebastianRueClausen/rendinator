use std::{borrow::Cow, mem};

use glam::{Mat4, UVec2, Vec2, Vec3};

use crate::{
    context::Context,
    resources::{self, ConstState, Consts, RenderState},
    util,
};

pub struct TemporalResolvePhase {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl TemporalResolvePhase {
    pub fn new(context: &mut Context, render_state: &RenderState) -> Self {
        let module = context.create_shader_module(
            include_str!("shaders/temporal_resolve.wgsl"),
            "shaders/temporal_resolve.wgsl",
        );

        let shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("temporal resolve"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("temporal resolve"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Depth,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                access: wgpu::StorageTextureAccess::WriteOnly,
                                format: resources::COLOR_BUFFER_FORMAT,
                                view_dimension: wgpu::TextureViewDimension::D2,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("temporal resolve"),
                    bind_group_layouts: &[
                        ConstState::bind_group_layout(context),
                        &bind_group_layout,
                    ],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..mem::size_of::<Mat4>() as u32,
                    }],
                });

        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("temporal resolve"),
                entry_point: "main",
                module: &shader,
                layout: Some(&pipeline_layout),
            });

        let bind_group = create_bind_group(context, render_state, &bind_group_layout);

        Self {
            pipeline,
            bind_group_layout,
            bind_group,
        }
    }

    pub fn record(
        &self,
        context: &Context,
        consts: &Consts,
        const_state: &ConstState,
        render_state: &RenderState,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let reproject = Mat4::from_translation(Vec3::new(0.5, 0.5, 0.0))
            * Mat4::from_scale(Vec3::new(0.5, -0.5, 1.0))
            * consts.prev_proj_view
            * consts.inverse_proj_view;

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("temporal resolve"),
        });

        compute_pass.set_pipeline(&self.pipeline);

        compute_pass.set_push_constants(0, bytemuck::bytes_of(&reproject));
        compute_pass.set_bind_group(0, &const_state.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.bind_group, &[]);

        let x = util::div_ceil(consts.surface_size.x, 8);
        let y = util::div_ceil(consts.surface_size.y, 8);

        compute_pass.dispatch_workgroups(x, y, 1);
        drop(compute_pass);

        encoder.copy_texture_to_texture(
            wgpu::ImageCopyTexture {
                texture: &render_state.post.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyTexture {
                texture: &render_state.color_accum.texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            context.surface_size,
        );
    }

    pub fn resize_surface(&mut self, context: &Context, render_state: &RenderState) {
        self.bind_group = create_bind_group(context, render_state, &self.bind_group_layout);
    }
}

pub fn jitter(frame_index: usize, surface_size: UVec2) -> Vec2 {
    let jitter = HALTON_SEQUENCE[frame_index as usize % 12];
    jitter / surface_size.as_vec2()
}

fn create_bind_group(
    context: &Context,
    render_state: &RenderState,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    let views = [
        &render_state.color.view,
        &render_state.depth.view,
        &render_state.color_accum.view,
        &render_state.post.view,
    ];

    let entries: Vec<_> = views
        .iter()
        .enumerate()
        .map(|(binding, view)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: wgpu::BindingResource::TextureView(view),
        })
        .collect();

    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("temporal resolve"),
            entries: &entries,
            layout,
        })
}

const HALTON_SEQUENCE: [Vec2; 12] = [
    Vec2::new(0.0, -0.16666666),
    Vec2::new(-0.25, 0.16666669),
    Vec2::new(0.25, -0.3888889),
    Vec2::new(-0.375, -0.055555552),
    Vec2::new(0.125, 0.2777778),
    Vec2::new(-0.125, -0.2777778),
    Vec2::new(0.375, 0.055555582),
    Vec2::new(-0.4375, 0.3888889),
    Vec2::new(0.0625, -0.46296296),
    Vec2::new(-0.1875, -0.12962961),
    Vec2::new(0.3125, 0.2037037),
    Vec2::new(-0.3125, -0.35185185),
];
