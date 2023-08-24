use std::{borrow::Cow, mem};

use bytemuck::NoUninit;

use crate::{
    context::Context,
    resources::{self, ConstState, RenderState},
};

bitflags::bitflags! {
    #[repr(C)]
    #[derive(Debug, Clone, Copy, NoUninit)]
    struct BloomFlags: u32 {
        const IS_INITIAL = 1 << 0;
    }
}

pub struct Config {
    pub lf_freq: f64,
    pub lf_curve: f64,
    pub hp_freq: f64,
    pub intensity: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            intensity: 0.15,
            lf_freq: 0.7,
            lf_curve: 0.95,
            hp_freq: 1.0,
        }
    }
}

pub struct BloomPhase {
    upsample: wgpu::RenderPipeline,
    downsample: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    initial_downsample: wgpu::BindGroup,
    bloom_mips: Vec<Mip>,
    config: Config,
}

impl BloomPhase {
    pub fn new(context: &mut Context, render_state: &RenderState) -> Self {
        let module = context.create_shader_module(
            include_str!("shaders/bloom.wgsl"),
            "shaders/bloom.wgsl",
            &[],
        );

        let shader = context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("bloom"),
                source: wgpu::ShaderSource::Naga(Cow::Owned(module)),
            });

        let bind_group_layout =
            context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("bloom"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    }],
                });

        let pipeline_layout =
            context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("bloom"),
                    bind_group_layouts: &[
                        ConstState::bind_group_layout(context),
                        &bind_group_layout,
                    ],
                    push_constant_ranges: &[wgpu::PushConstantRange {
                        stages: wgpu::ShaderStages::FRAGMENT,
                        range: 0..mem::size_of::<u32>() as u32,
                    }],
                });

        let downsample = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("bloom downsample"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vertex",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "downsample",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: resources::COLOR_BUFFER_FORMAT,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
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

        let upsample = context
            .device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("bloom upsample"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: "vertex",
                    buffers: &[],
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: "upsample",
                    targets: &[Some(wgpu::ColorTargetState {
                        format: resources::COLOR_BUFFER_FORMAT,
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Constant,
                                dst_factor: wgpu::BlendFactor::OneMinusConstant,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::Zero,
                                dst_factor: wgpu::BlendFactor::One,
                                operation: wgpu::BlendOperation::Add,
                            },
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
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

        let (initial_downsample, bloom_mips) =
            create_mips(context, render_state, &bind_group_layout);

        let config = Config::default();

        Self {
            bloom_mips,
            initial_downsample,
            bind_group_layout,
            upsample,
            downsample,
            config,
        }
    }

    fn max_mip(&self) -> u32 {
        self.bloom_mips.len() as u32 - 1
    }

    fn initial_downsample(&self, const_state: &ConstState, encoder: &mut wgpu::CommandEncoder) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("initial bloom downsample"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.bloom_mips[0].view,
                resolve_target: None,
                ops: wgpu::Operations::default(),
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&self.downsample);
        render_pass.set_bind_group(0, &const_state.bind_group, &[]);
        render_pass.set_bind_group(1, &self.initial_downsample, &[]);

        render_pass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, &0u32.to_ne_bytes());

        render_pass.draw(0..3, 0..1);
    }

    fn downsample(&self, const_state: &ConstState, encoder: &mut wgpu::CommandEncoder) {
        for window in self.bloom_mips.windows(2) {
            let input = &window[0];
            let output = &window[1];

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom downsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output.view,
                    resolve_target: None,
                    ops: wgpu::Operations::default(),
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.downsample);
            render_pass.set_bind_group(0, &const_state.bind_group, &[]);
            render_pass.set_bind_group(1, &input.bind_group, &[]);
            render_pass.set_push_constants(
                wgpu::ShaderStages::FRAGMENT,
                0,
                &input.level.to_ne_bytes(),
            );

            render_pass.draw(0..3, 0..1);
        }
    }

    fn upsample(&self, const_state: &ConstState, encoder: &mut wgpu::CommandEncoder) {
        for window in self.bloom_mips.windows(2).rev() {
            let input = &window[1];
            let output = &window[0];

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bloom upsample"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: true,
                    },
                })],
                depth_stencil_attachment: None,
            });

            render_pass.set_pipeline(&self.upsample);
            render_pass.set_bind_group(0, &const_state.bind_group, &[]);
            render_pass.set_bind_group(1, &input.bind_group, &[]);
            render_pass.set_push_constants(
                wgpu::ShaderStages::FRAGMENT,
                0,
                &input.level.to_ne_bytes(),
            );

            let blend_constant = blend_constant(input.level, self.max_mip(), &self.config);
            render_pass.set_blend_constant(blend_constant);

            render_pass.draw(0..3, 0..1);
        }
    }

    fn final_upsample(
        &self,
        const_state: &ConstState,
        render_state: &RenderState,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let input = self.bloom_mips.last().unwrap();

        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("final bloom upsample"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &render_state.post.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: true,
                },
            })],
            depth_stencil_attachment: None,
        });

        render_pass.set_pipeline(&self.upsample);
        render_pass.set_bind_group(0, &const_state.bind_group, &[]);
        render_pass.set_bind_group(1, &input.bind_group, &[]);
        render_pass.set_push_constants(wgpu::ShaderStages::FRAGMENT, 0, &0u32.to_ne_bytes());

        let blend_constant = blend_constant(0, self.max_mip(), &self.config);
        render_pass.set_blend_constant(blend_constant);

        render_pass.draw(0..3, 0..1);
    }

    pub fn record(
        &self,
        const_state: &ConstState,
        render_state: &RenderState,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        self.initial_downsample(const_state, encoder);
        self.downsample(const_state, encoder);
        self.upsample(const_state, encoder);
        self.final_upsample(const_state, render_state, encoder);
    }

    pub fn resize_surface(&mut self, context: &Context, render_state: &RenderState) {
        (self.initial_downsample, self.bloom_mips) =
            create_mips(context, render_state, &self.bind_group_layout);
    }
}

struct Mip {
    bind_group: wgpu::BindGroup,
    view: wgpu::TextureView,
    level: u32,
}

fn create_mips(
    context: &Context,
    render_state: &RenderState,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> (wgpu::BindGroup, Vec<Mip>) {
    let size = context.surface_size;
    let mip_level_count = size.width.min(size.height).ilog2().max(2) - 1;

    let bloom_texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bloom"),
        dimension: wgpu::TextureDimension::D2,
        format: resources::COLOR_BUFFER_FORMAT,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
        mip_level_count,
        sample_count: 1,
        view_formats: &[],
        size,
    });

    let mips: Vec<_> = (0..mip_level_count)
        .map(|level| {
            let view = bloom_texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("bloom"),
                base_mip_level: level,
                mip_level_count: Some(1),
                ..Default::default()
            });

            let bind_group = context
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("bloom"),
                    layout: bind_group_layout,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&view),
                    }],
                });

            Mip {
                bind_group,
                view,
                level,
            }
        })
        .collect();

    let initial = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bloom"),
            layout: bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&render_state.post.view),
            }],
        });

    (initial, mips)
}

fn blend_constant(mip: u32, max_mip: u32, config: &Config) -> wgpu::Color {
    let mip = mip as f64;
    let max_mip = max_mip as f64;

    let mut lf_boost =
        (1.0 - (1.0 - (mip / max_mip)).powf(1.0 / (1.0 - config.lf_curve))) * config.lf_freq;
    lf_boost *= 1.0 - config.intensity;

    let high_pass_lq = 1.0 - (((mip / max_mip) - config.hp_freq) / config.hp_freq).clamp(0.0, 1.0);
    let factor = (config.intensity + lf_boost) * high_pass_lq;

    wgpu::Color {
        r: factor,
        g: factor,
        b: factor,
        a: 1.0,
    }
}
