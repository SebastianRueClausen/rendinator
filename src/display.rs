use std::{array, borrow::Cow, f32::consts::TAU, mem, time::Duration};

use bytemuck::NoUninit;

use crate::{
    context::{self, Context},
    util,
};

#[repr(C)]
#[derive(Clone, Copy, NoUninit)]
struct LuminanceParams {
    min_log_luminance: f32,
    inverse_log_luminance_range: f32,
    log_luminance_range: f32,
    time_coeff: f32,
    pixel_count: u32,
}

pub struct DisplayPhase {
    display: wgpu::RenderPipeline,
    display_bind_group_layout: wgpu::BindGroupLayout,
    display_bind_group: wgpu::BindGroup,
    luminance_bind_group: wgpu::BindGroup,
    luminance_histogram: wgpu::ComputePipeline,
    luminance_average: wgpu::ComputePipeline,
}

impl DisplayPhase {
    pub fn new(
        context: &mut Context,
        output_format: wgpu::TextureFormat,
        input: &wgpu::TextureView,
    ) -> Self {
        let display_module = context.create_shader_module(
            include_str!("shaders/display.wgsl"),
            "shaders/display.wgsl",
            &[],
        );

        let display_shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("display"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(display_module)),
            });

        let display_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("display"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT | wgpu::ShaderStages::COMPUTE,
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

        let entries: [_; 2] = array::from_fn(|binding| wgpu::BindGroupLayoutEntry {
            binding: binding as u32,
            visibility: wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });

        let luminance_bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("luminance"),
                    entries: &entries,
                });

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("display"),
                    bind_group_layouts: &[&display_bind_group_layout, &luminance_bind_group_layout],
                    push_constant_ranges: &[],
                });

        let display = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("display"),
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

        let luminance_histogram = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("average luminance"),
            size: mem::size_of::<[u32; 256]>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let average_luminance = context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("average luminance"),
            size: mem::size_of::<f32>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let luminance_bind_group = context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("luminance"),
                layout: &luminance_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &luminance_histogram,
                            offset: 0,
                            size: None,
                        }),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                            buffer: &average_luminance,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            });

        let (luminance_histogram, luminance_average) = create_luminance_pipelines(
            context,
            &display_bind_group_layout,
            &luminance_bind_group_layout,
        );

        Self {
            display,
            display_bind_group,
            display_bind_group_layout,
            luminance_bind_group,
            luminance_histogram,
            luminance_average,
        }
    }

    pub fn resize_surface(&mut self, context: &Context, input: &wgpu::TextureView) {
        self.display_bind_group =
            create_display_bind_group(context, input, &self.display_bind_group_layout);
    }

    pub fn record(
        &self,
        context: &Context,
        delta_time: Duration,
        frame_buffer: &wgpu::TextureView,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("luminance"),
        });

        compute_pass.set_pipeline(&self.luminance_histogram);
        compute_pass.set_bind_group(0, &self.display_bind_group, &[]);
        compute_pass.set_bind_group(1, &self.luminance_bind_group, &[]);

        let min_log_luminance = -8.0;
        let max_log_luminance = 3.5;

        let log_luminance_range = max_log_luminance - min_log_luminance;
        let time_coeff = f32::clamp(1.0 - (-delta_time.as_secs_f32() * TAU).exp(), 0.0, 1.0);

        let params = LuminanceParams {
            min_log_luminance,
            log_luminance_range,
            inverse_log_luminance_range: log_luminance_range.recip(),
            pixel_count: context.surface_size.width * context.surface_size.height,
            time_coeff,
        };

        compute_pass.set_push_constants(0, bytemuck::bytes_of(&params));

        let x = util::div_ceil(context.surface_size.width, 16);
        let y = util::div_ceil(context.surface_size.height, 16);
        compute_pass.dispatch_workgroups(x, y, 1);

        compute_pass.set_pipeline(&self.luminance_average);
        compute_pass.dispatch_workgroups(1, 1, 1);

        drop(compute_pass);

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("display"),
            depth_stencil_attachment: None,
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &frame_buffer,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
        });

        render_pass.set_pipeline(&self.display);
        render_pass.set_bind_group(0, &self.display_bind_group, &[]);
        render_pass.set_bind_group(1, &self.luminance_bind_group, &[]);
        render_pass.draw(0..3, 0..1);
    }
}

fn create_luminance_pipelines(
    context: &mut Context,
    display_bind_group_layout: &wgpu::BindGroupLayout,
    luminance_bind_group_layout: &wgpu::BindGroupLayout,
) -> (wgpu::ComputePipeline, wgpu::ComputePipeline) {
    let luminance_pipeline_layout =
        context
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("luminance"),
                bind_group_layouts: &[&display_bind_group_layout, &luminance_bind_group_layout],
                push_constant_ranges: &[wgpu::PushConstantRange {
                    stages: wgpu::ShaderStages::COMPUTE,
                    range: 0..mem::size_of::<LuminanceParams>() as u32,
                }],
            });

    let luminance_shader_source = include_str!("shaders/luminance.wgsl");
    let luminance_shader_path = "shaders/luminance.wgsl";

    let luminance_histogram_module = context.create_shader_module(
        luminance_shader_source,
        luminance_shader_path,
        &[("BUILD_HISTOGRAM", context::ShaderDefValue::Bool(true))],
    );

    let luminance_histogram_shader =
        context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("luminance histogram"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(luminance_histogram_module)),
            });

    let luminance_average_module = context.create_shader_module(
        luminance_shader_source,
        luminance_shader_path,
        &[("BUILD_HISTOGRAM", context::ShaderDefValue::Bool(false))],
    );

    let luminance_average_shader =
        context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("average luminance"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(luminance_average_module)),
            });

    let luminance_histogram =
        context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("luminance histogram"),
                layout: Some(&luminance_pipeline_layout),
                module: &luminance_histogram_shader,
                entry_point: "build_histogram",
            });

    let luminance_average =
        context
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("average luminance"),
                layout: Some(&luminance_pipeline_layout),
                module: &luminance_average_shader,
                entry_point: "compute_average",
            });

    (luminance_histogram, luminance_average)
}

fn create_display_bind_group(
    context: &Context,
    display: &wgpu::TextureView,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("display"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(display),
            }],
        })
}
