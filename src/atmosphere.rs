use std::borrow::Cow;

use crate::{
    context::Context,
    resources::{self, Skybox},
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
        );

        let shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("atmosphere"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            });

        let pipeline = context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("atmosphere"),
                layout: None,
                module: &shader,
                entry_point: "main",
            });

        let bind_group_layout = pipeline.get_bind_group_layout(0);

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

    pub fn record(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("atmosphere"),
        });

        compute_pass.set_pipeline(&self.pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);

        let x = resources::SKYBOX_SIZE.width / 8;
        let y = resources::SKYBOX_SIZE.height / 8;
        let z = resources::SKYBOX_SIZE.depth_or_array_layers;

        compute_pass.dispatch_workgroups(x, y, z);
    }
}
