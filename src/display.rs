use std::borrow::Cow;

use crate::context::Context;

pub struct BlendPhase {
    display: wgpu::RenderPipeline,
    display_bind_group_layout: wgpu::BindGroupLayout,
    display_bind_group: wgpu::BindGroup,
}

impl BlendPhase {
    pub fn new(
        context: &mut Context,
        output_format: wgpu::TextureFormat,
        input: &wgpu::TextureView,
    ) -> Self {
        let display_module =
            context.create_shader_module(include_str!("shaders/blend.wgsl"), "shaders/blend.wgsl");

        let display_shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("blend"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(display_module)),
            });

        let display_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("blend"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    }],
                });

        let display_bind_group =
            create_display_bind_group(context, input, &display_bind_group_layout);

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("blend"),
                    bind_group_layouts: &[&display_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let display = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("blend"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    entry_point: "vertex",
                    module: &display_shader,
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    entry_point: "fragment",
                    module: &display_shader,
                    targets: &[Some(wgpu::ColorTargetState {
                        write_mask: wgpu::ColorWrites::ALL,
                        blend: None,
                        format: output_format,
                    })],
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
            });

        Self {
            display,
            display_bind_group,
            display_bind_group_layout,
        }
    }

    pub fn change_input(&mut self, context: &Context, input: &wgpu::TextureView) {
        self.display_bind_group =
            create_display_bind_group(context, input, &self.display_bind_group_layout);
    }

    pub fn record<'a>(
        &'a self,
        blend_constant: wgpu::Color,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        render_pass.set_pipeline(&self.display);
        render_pass.set_bind_group(0, &self.display_bind_group, &[]);
        render_pass.set_blend_constant(blend_constant);
        render_pass.draw(0..3, 0..1);
    }
}

fn create_display_bind_group(
    context: &Context,
    input: &wgpu::TextureView,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(input),
            }],
        })
}
