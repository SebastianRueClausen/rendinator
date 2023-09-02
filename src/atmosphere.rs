use std::borrow::Cow;

use crate::{
    context::Context,
    resources::{self, ConstState, Skybox, SKYBOX_FORMAT},
};

pub struct AtmospherePhase {
    pipeline: wgpu::ComputePipeline,
    bind_group: wgpu::BindGroup,
}

impl AtmospherePhase {
    pub fn new(context: &mut Context, skybox: &Skybox) -> Self {
        let module = context.create_shader_module(
            include_str!("shaders/atmosphere.wgsl"),
            "shaders/atmosphere.wgsl",
            &[],
        );

        let shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("atmosphere"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("atmosphere"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::StorageTexture {
                            access: wgpu::StorageTextureAccess::WriteOnly,
                            format: SKYBOX_FORMAT,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                        },
                        count: None,
                    }],
                });

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("atmosphere"),
                    bind_group_layouts: &[
                        ConstState::bind_group_layout(context),
                        &bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                });

        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("atmosphere"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
            });

        let bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("atmosphere"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&skybox.array_view),
                }],
            });

        Self {
            pipeline,
            bind_group,
        }
    }

    pub fn record(&self, const_state: &ConstState, encoder: &mut wgpu::CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("atmosphere"),
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &const_state.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.bind_group, &[]);

        let x = resources::SKYBOX_SIZE.width / 8;
        let y = resources::SKYBOX_SIZE.height / 8;
        let z = resources::SKYBOX_SIZE.depth_or_array_layers;

        compute_pass.dispatch_workgroups(x, y, z);
    }
}
