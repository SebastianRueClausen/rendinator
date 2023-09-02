use std::{borrow::Cow, mem};

use glam::Mat4;

use crate::{
    camera::Camera,
    context::Context,
    resources::{self, ConstState, RenderState, SceneState, ShadowCascades, Skybox},
    util,
};

pub struct ShadePhase {
    shade: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ShadePhase {
    pub fn new(
        context: &mut Context,
        scene_state: &SceneState,
        render_state: &RenderState,
        shadow_cascades: &ShadowCascades,
        skybox: &Skybox,
    ) -> ShadePhase {
        let shade_module = context.create_shader_module(
            include_str!("shaders/shade.wgsl"),
            "shaders/shade.wgsl",
            &[],
        );

        let shade_shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("shade"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(shade_module)),
            });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("shade"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                sample_type: wgpu::TextureSampleType::Uint,
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
                                view_dimension: wgpu::TextureViewDimension::Cube,
                                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                                multisampled: false,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::StorageTexture {
                                format: resources::COLOR_BUFFER_FORMAT,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                access: wgpu::StorageTextureAccess::ReadWrite,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Texture {
                                view_dimension: wgpu::TextureViewDimension::D2Array,
                                sample_type: wgpu::TextureSampleType::Depth,
                                multisampled: false,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = create_shade_bind_group(
            context,
            render_state,
            shadow_cascades,
            skybox,
            &bind_group_layout,
        );

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("shade"),
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::COMPUTE,
                        range: 0..mem::size_of::<Mat4>() as u32,
                    }],
                    bind_group_layouts: &[
                        ConstState::bind_group_layout(&context),
                        &scene_state.bind_group_layout,
                        &bind_group_layout,
                    ],
                });

        let shade = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("shade"),
                layout: Some(&pipeline_layout),
                module: &shade_shader,
                entry_point: "shade",
            });

        Self {
            shade,
            bind_group,
            bind_group_layout,
        }
    }

    pub fn resize_surface(
        &mut self,
        context: &Context,
        render_state: &RenderState,
        shadow_cascades: &ShadowCascades,
        skybox: &Skybox,
    ) {
        self.bind_group = create_shade_bind_group(
            context,
            render_state,
            shadow_cascades,
            skybox,
            &self.bind_group_layout,
        );
    }

    pub fn record(
        &self,
        context: &Context,
        camera: &Camera,
        const_state: &ConstState,
        scene_state: &SceneState,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("shade"),
        });

        compute_pass.set_pipeline(&self.shade);
        compute_pass.set_bind_group(0, &const_state.bind_group, &[]);
        compute_pass.set_bind_group(1, &scene_state.bind_group, &[]);
        compute_pass.set_bind_group(2, &self.bind_group, &[]);

        let mut ray_matrix = camera.proj_view();
        ray_matrix.col_mut(3)[0] = 0.0;
        ray_matrix.col_mut(3)[1] = 0.0;
        ray_matrix.col_mut(3)[2] = 0.0;
        ray_matrix = ray_matrix.inverse();

        compute_pass.set_push_constants(0, bytemuck::bytes_of(&ray_matrix));

        let x = util::div_ceil(context.surface_size.width, 8);
        let y = util::div_ceil(context.surface_size.height, 8);

        compute_pass.dispatch_workgroups(x, y, 1);
    }
}

fn create_shade_bind_group(
    context: &Context,
    render_state: &RenderState,
    shadow_cascades: &ShadowCascades,
    skybox: &Skybox,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shade"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&render_state.visibility.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&render_state.depth.view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&skybox.cube_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&render_state.color.view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &shadow_cascades.cascade_info,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&shadow_cascades.cascade_array),
                },
            ],
        })
}
