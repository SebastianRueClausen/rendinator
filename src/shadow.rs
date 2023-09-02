use std::{borrow::Cow, mem};

use bytemuck::NoUninit;

use crate::{
    context::Context,
    resources::{
        ConstState, DepthPyramid, SceneState, ShadowCascades, SHADOW_CASCADE_FORMAT,
        SHADOW_CASCADE_SIZE,
    },
};

#[repr(C)]
#[derive(Clone, Copy, NoUninit)]
struct ShadowParams {
    cascade_size: u32,
    lambda: f32,
    near_offset: f32,
}

pub struct ShadowPhase {
    render_cascade: wgpu::RenderPipeline,
    setup_cascades: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    bind_group: wgpu::BindGroup,
}

impl ShadowPhase {
    pub fn new(
        context: &mut Context,
        scene_state: &SceneState,
        shadow_cascades: &ShadowCascades,
        depth_pyramid: &DepthPyramid,
    ) -> Self {
        let cascade_setup_module = context.create_shader_module(
            include_str!("shaders/cascade_setup.wgsl"),
            "shaders/cascade_setup.wgsl",
            &[],
        );

        let cascade_setup_shader =
            context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("cascade setup"),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(cascade_setup_module)),
                });

        let cascade_render_module = context.create_shader_module(
            include_str!("shaders/shadow_render.wgsl"),
            "shaders/shadow_render.wgsl",
            &[],
        );

        let cascade_render_shader =
            context
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("cascade render"),
                    source: wgpu::ShaderSource::Naga(Cow::Owned(cascade_render_module)),
                });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("shadow"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                                view_dimension: wgpu::TextureViewDimension::D2,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group =
            create_bind_group(context, &shadow_cascades, depth_pyramid, &bind_group_layout);

        let setup_cascades_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("cascade setup"),
                    bind_group_layouts: &[
                        ConstState::bind_group_layout(context),
                        &bind_group_layout,
                    ],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..mem::size_of::<ShadowParams>() as u32,
                    }],
                });

        let setup_cascades =
            context
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("cascade setup"),
                    layout: Some(&setup_cascades_layout),
                    module: &cascade_setup_shader,
                    entry_point: "main",
                });

        let render_cascade_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render cascade"),
                    bind_group_layouts: &[&bind_group_layout, &scene_state.bind_group_layout],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::VERTEX,
                        range: 0..mem::size_of::<u32>() as u32,
                    }],
                });

        let render_cascade =
            context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("render cascade"),
                    layout: Some(&render_cascade_layout),
                    vertex: wgpu::VertexState {
                        module: &cascade_render_shader,
                        entry_point: "vertex",
                        buffers: &[],
                    },
                    primitive: wgpu::PrimitiveState::default(),
                    depth_stencil: Some(wgpu::DepthStencilState {
                        format: SHADOW_CASCADE_FORMAT,
                        depth_write_enabled: true,
                        depth_compare: wgpu::CompareFunction::Never,
                        stencil: wgpu::StencilState::default(),
                        bias: wgpu::DepthBiasState::default(),
                    }),
                    multisample: wgpu::MultisampleState::default(),
                    fragment: None,
                    multiview: None,
                });

        Self {
            bind_group,
            bind_group_layout,
            setup_cascades,
            render_cascade,
        }
    }

    pub fn resize_surface(
        &mut self,
        context: &Context,
        shadow_cascades: &ShadowCascades,
        depth_pyramid: &DepthPyramid,
    ) {
        self.bind_group = create_bind_group(
            context,
            &shadow_cascades,
            depth_pyramid,
            &self.bind_group_layout,
        );
    }

    pub fn record(
        &self,
        const_state: &ConstState,
        shadow_cascades: &ShadowCascades,
        scene_state: &SceneState,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("setup cascades"),
        });

        let params = ShadowParams {
            cascade_size: SHADOW_CASCADE_SIZE,
            lambda: 0.4,
            near_offset: 250.0,
        };

        compute_pass.set_pipeline(&self.setup_cascades);
        compute_pass.set_bind_group(0, &const_state.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.bind_group, &[]);
        compute_pass.set_push_constants(0, bytemuck::bytes_of(&params));
        compute_pass.dispatch_workgroups(1, 1, 1);

        drop(compute_pass);

        for (index, cascade) in shadow_cascades.cascades.iter().enumerate() {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render cascade"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: cascade,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(0.0),
                        store: true,
                    }),
                    stencil_ops: None,
                }),
            });

            render_pass.set_pipeline(&self.render_cascade);
            render_pass.set_bind_group(0, &self.bind_group, &[]);
            render_pass.set_bind_group(1, &scene_state.bind_group, &[]);

            let index = index as u32;
            render_pass.set_push_constants(wgpu::ShaderStages::VERTEX, 0, &index.to_le_bytes());

            for (index, draw) in scene_state.primitive_draw_infos.iter().enumerate() {
                let index = index as u32;
                render_pass.draw(draw.indices.clone(), index..index + 1);
            }
        }
    }
}

fn create_bind_group(
    context: &Context,
    shadow_cascades: &ShadowCascades,
    depth_pyramid: &DepthPyramid,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &shadow_cascades.cascade_info,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_pyramid.whole),
                },
            ],
        })
}
