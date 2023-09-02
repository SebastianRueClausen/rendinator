use std::borrow::Cow;

use crate::{
    context::{Context, ShaderDefValue},
    resources::{self, ConstState, DepthPyramid, RenderState},
    util,
};

pub struct DepthReducePhase {
    initial_reduce: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    initial_bind_group_layout: wgpu::BindGroupLayout,
    initial_bind_group: wgpu::BindGroup,
    bind_groups: Vec<wgpu::BindGroup>,
}

impl DepthReducePhase {
    pub fn new(
        context: &mut Context,
        render_state: &RenderState,
        depth_pyramid: &DepthPyramid,
    ) -> Self {
        let bind_group_layout = create_bind_group_layout(context, false);
        let initial_bind_group_layout = create_bind_group_layout(context, true);

        let initial_bind_group = create_bind_group(
            context,
            &render_state.depth.view,
            depth_pyramid.mips.first().unwrap(),
            &initial_bind_group_layout,
        );

        let bind_groups: Vec<_> = depth_pyramid
            .mips
            .windows(2)
            .map(|window| create_bind_group(context, &window[0], &window[1], &bind_group_layout))
            .collect();

        let initial_shader = create_shader(context, true);
        let shader = create_shader(context, false);

        let initial_pipeline_layout = create_pipeline_layout(context, &initial_bind_group_layout);
        let pipeline_layout = create_pipeline_layout(context, &bind_group_layout);

        let initial_reduce = create_pipeline(context, &initial_pipeline_layout, initial_shader);
        let reduce = create_pipeline(context, &pipeline_layout, shader);

        Self {
            initial_reduce,
            reduce,
            bind_group_layout,
            initial_bind_group_layout,
            initial_bind_group,
            bind_groups,
        }
    }

    pub fn rezize_surface(
        &mut self,
        context: &Context,
        render_state: &RenderState,
        depth_pyramid: &DepthPyramid,
    ) {
        self.initial_bind_group = create_bind_group(
            context,
            &render_state.depth.view,
            depth_pyramid.mips.first().unwrap(),
            &self.initial_bind_group_layout,
        );

        self.bind_groups = depth_pyramid
            .mips
            .windows(2)
            .map(|window| {
                create_bind_group(context, &window[0], &window[1], &self.bind_group_layout)
            })
            .collect();
    }

    pub fn record(
        &self,
        depth_pyramid: &DepthPyramid,
        const_state: &ConstState,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("depth reduce"),
        });

        compute_pass.set_pipeline(&self.initial_reduce);
        compute_pass.set_bind_group(0, &const_state.bind_group, &[]);
        compute_pass.set_bind_group(1, &self.initial_bind_group, &[]);

        let size = depth_pyramid.texture.size();
        compute_pass.dispatch_workgroups(
            util::div_ceil(size.width, 8),
            util::div_ceil(size.height, 8),
            1,
        );

        compute_pass.set_pipeline(&self.reduce);

        for (depth, bind_group) in self.bind_groups.iter().enumerate() {
            compute_pass.set_bind_group(1, bind_group, &[]);

            let size = size.mip_level_size(depth as u32 + 1, wgpu::TextureDimension::D2);
            compute_pass.dispatch_workgroups(
                util::div_ceil(size.width, 8),
                util::div_ceil(size.height, 8),
                1,
            );
        }
    }
}

fn create_pipeline(
    context: &Context,
    layout: &wgpu::PipelineLayout,
    shader: wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    context
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("depth reduce"),
            layout: Some(layout),
            module: &shader,
            entry_point: "main",
        })
}

fn create_shader(context: &mut Context, initial: bool) -> wgpu::ShaderModule {
    let initial_module = context.create_shader_module(
        include_str!("shaders/depth_reduce.wgsl"),
        "shaders/depth_reduce.wgsl",
        &[("INITIAL_REDUCE", ShaderDefValue::Bool(initial))],
    );

    context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("depth reduce"),
            source: wgpu::ShaderSource::Naga(Cow::Owned(initial_module)),
        })
}

fn create_pipeline_layout(
    context: &Context,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    context
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("depth reduce"),
            bind_group_layouts: &[ConstState::bind_group_layout(context), bind_group_layout],
            push_constant_ranges: &[],
        })
}

fn create_bind_group(
    context: &Context,
    input: &wgpu::TextureView,
    output: &wgpu::TextureView,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("depth reduce"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(input),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(output),
                },
            ],
        })
}

fn create_bind_group_layout(context: &Context, initial: bool) -> wgpu::BindGroupLayout {
    let entries = [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type: if initial {
                    wgpu::TextureSampleType::Depth
                } else {
                    wgpu::TextureSampleType::Float { filterable: true }
                },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: resources::DEPTH_PYRAMID_FORMAT,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        },
    ];

    context
        .device
        .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("depth reduce"),
            entries: &entries,
        })
}
