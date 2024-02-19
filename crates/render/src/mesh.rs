use std::mem;

use ash::vk::{self};
use eyre::Result;
use glam::{Mat4, Vec2};

use crate::constants::Constants;
use crate::render_targets::RenderTargets;
use crate::scene::{DrawCommand, Scene};
use crate::{hal, Descriptors};

pub(crate) struct MeshPhase {
    pipeline: hal::Pipeline,
    depth_reduce_pipeline: hal::Pipeline,
    pre_cull_pipeline: hal::Pipeline,
    post_cull_pipeline: hal::Pipeline,
    gbuffer_pipeline: hal::Pipeline,
    descriptor_layout: hal::DescriptorLayout,
    cull_descriptor_layout: hal::DescriptorLayout,
    depth_reduce_descriptor_layout: hal::DescriptorLayout,
    gbuffer_descriptor_layout: hal::DescriptorLayout,
    depth_pyramid: hal::Image,
    depth_sampler: hal::Sampler,
    memory: hal::Memory,
}

impl MeshPhase {
    pub fn new(
        device: &hal::Device,
        swapchain: &hal::Swapchain,
        render_targets: &RenderTargets,
    ) -> Result<Self> {
        let descriptor_layout = hal::DescriptorLayoutBuilder::default()
            .binding(vk::DescriptorType::UNIFORM_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .array_binding(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1024)
            .build(device)?;
        let depth_reduce_descriptor_layout =
            hal::DescriptorLayoutBuilder::default()
                .binding(vk::DescriptorType::STORAGE_IMAGE)
                .binding(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .build(device)?;
        let cull_descriptor_layout = hal::DescriptorLayoutBuilder::default()
            .binding(vk::DescriptorType::UNIFORM_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(device)?;
        let gbuffer_descriptor_layout =
            create_gbuffer_descriptor_layout(device)?;

        let vertex_shader = hal::Shader::new(
            device,
            &hal::ShaderRequest {
                stage: vk::ShaderStageFlags::VERTEX,
                source: vk_shader_macros::include_glsl!(
                    "src/shaders/mesh/vert.glsl",
                    kind: vert,
                ),
            },
        )?;
        let fragment_shader = hal::Shader::new(
            device,
            &hal::ShaderRequest {
                stage: vk::ShaderStageFlags::FRAGMENT,
                source: vk_shader_macros::include_glsl!(
                    "src/shaders/mesh/frag.glsl",
                    kind: frag,
                ),
            },
        )?;
        let pipeline_layout = hal::PipelineLayout {
            descriptors: &[&descriptor_layout],
            push_constant: None,
        };
        let specializations = hal::Specializations::default();
        let pipeline = hal::Pipeline::graphics(
            device,
            &pipeline_layout,
            &hal::GraphicsPipelineRequest {
                color_attachments: &[
                    hal::ColorAttachment {
                        format: swapchain.format,
                        blend: Some(hal::Blend {
                            src: vk::BlendFactor::ONE,
                            dst: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                        }),
                    },
                    hal::ColorAttachment {
                        format: render_targets.visibility.format,
                        blend: None,
                    },
                ],
                depth_format: Some(render_targets.depth.format),
                cull_mode: vk::CullModeFlags::BACK,
                shaders: &[
                    hal::ShaderStage {
                        shader: &vertex_shader,
                        specializations: &specializations,
                    },
                    hal::ShaderStage {
                        shader: &fragment_shader,
                        specializations: &specializations,
                    },
                ],
            },
        )?;
        fragment_shader.destroy(device);
        vertex_shader.destroy(device);

        let (depth_pyramid, memory) = create_depth_pyramid(device, swapchain)?;
        let depth_sampler = create_depth_sampler(device)?;

        let depth_reduce_pipeline = create_depth_reduce_pipeline(
            device,
            &depth_reduce_descriptor_layout,
        )?;
        let pre_cull_pipeline = create_cull_pipeline(
            device,
            &cull_descriptor_layout,
            DrawPhase::Pre,
        )?;
        let post_cull_pipeline = create_cull_pipeline(
            device,
            &cull_descriptor_layout,
            DrawPhase::Post,
        )?;
        let gbuffer_pipeline =
            create_gbuffer_pipeline(device, &gbuffer_descriptor_layout)?;
        Ok(Self {
            pipeline,
            descriptor_layout,
            depth_reduce_descriptor_layout,
            depth_reduce_pipeline,
            cull_descriptor_layout,
            pre_cull_pipeline,
            post_cull_pipeline,
            gbuffer_descriptor_layout,
            gbuffer_pipeline,
            depth_pyramid,
            memory,
            depth_sampler,
        })
    }

    pub fn destroy(&self, device: &hal::Device) {
        self.descriptor_layout.destroy(device);
        self.depth_reduce_descriptor_layout.destroy(device);
        self.cull_descriptor_layout.destroy(device);
        self.gbuffer_descriptor_layout.destroy(device);
        self.pipeline.destroy(device);
        self.depth_reduce_pipeline.destroy(device);
        self.pre_cull_pipeline.destroy(device);
        self.post_cull_pipeline.destroy(device);
        self.gbuffer_pipeline.destroy(device);
        self.depth_pyramid.destroy(device);
        self.memory.free(device);
        self.depth_sampler.destroy(device);
    }
}

#[derive(Clone, Copy)]
enum DrawPhase {
    Pre,
    Post,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct CullData {
    total_draw_count: u32,
}

fn create_depth_pyramid(
    device: &hal::Device,
    swapchain: &hal::Swapchain,
) -> Result<(hal::Image, hal::Memory)> {
    let vk::Extent2D { width, height } = swapchain.extent;
    // Find previous power of 2.
    let width = (width / 2).next_power_of_two();
    let height = (height / 2).next_power_of_two();
    let mip_level_count = (width.max(height) as f32).log2().ceil() as u32;
    let extent = vk::Extent3D { width, height, depth: 1 };
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::STORAGE
        | vk::ImageUsageFlags::TRANSFER_SRC;
    let mut image = hal::Image::new(
        device,
        &hal::ImageRequest {
            format: vk::Format::R32_SFLOAT,
            mip_level_count,
            extent,
            usage,
        },
    )?;
    let memory = hal::image_memory(
        device,
        &image,
        vk::MemoryPropertyFlags::DEVICE_LOCAL,
    )?;
    for base_mip_level in 0..mip_level_count {
        image.add_view(
            device,
            hal::ImageViewRequest { mip_level_count: 1, base_mip_level },
        )?;
    }
    image.add_view(
        device,
        hal::ImageViewRequest { mip_level_count, base_mip_level: 0 },
    )?;
    Ok((image, memory))
}

fn create_depth_sampler(device: &hal::Device) -> Result<hal::Sampler> {
    hal::Sampler::new(
        device,
        &hal::SamplerRequest {
            filter: vk::Filter::LINEAR,
            address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            reduction_mode: Some(vk::SamplerReductionMode::MIN),
            max_anisotropy: None,
        },
    )
}

fn create_depth_reduce_pipeline(
    device: &hal::Device,
    descriptor_layout: &hal::DescriptorLayout,
) -> Result<hal::Pipeline> {
    let layout = hal::PipelineLayout {
        descriptors: &[descriptor_layout],
        push_constant: Some(vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            size: mem::size_of::<Vec2>() as u32,
            offset: 0,
        }),
    };
    let shader = hal::Shader::new(
        device,
        &hal::ShaderRequest {
            stage: vk::ShaderStageFlags::COMPUTE,
            source: vk_shader_macros::include_glsl!(
                "src/shaders/mesh/depth_reduce.comp.glsl",
                kind: comp,
            ),
        },
    )?;
    let specializations = hal::Specializations::default();
    let shader_stage =
        hal::ShaderStage { shader: &shader, specializations: &specializations };
    let pipeline = hal::Pipeline::compute(device, &layout, shader_stage)?;
    shader.destroy(device);
    Ok(pipeline)
}

fn create_gbuffer_descriptor_layout(
    device: &hal::Device,
) -> Result<hal::DescriptorLayout> {
    hal::DescriptorLayoutBuilder::default()
        // Constant buffer.
        .binding(vk::DescriptorType::UNIFORM_BUFFER)
        // Scene buffers.
        .binding(vk::DescriptorType::STORAGE_BUFFER)
        .binding(vk::DescriptorType::STORAGE_BUFFER)
        .binding(vk::DescriptorType::STORAGE_BUFFER)
        .binding(vk::DescriptorType::STORAGE_BUFFER)
        .binding(vk::DescriptorType::STORAGE_BUFFER)
        // Visibility buffer.
        .binding(vk::DescriptorType::STORAGE_IMAGE)
        // G-buffers.
        .binding(vk::DescriptorType::STORAGE_IMAGE)
        .binding(vk::DescriptorType::STORAGE_IMAGE)
        .binding(vk::DescriptorType::STORAGE_IMAGE)
        // Textures.
        .array_binding(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1024)
        .build(device)
}

fn create_gbuffer_pipeline(
    device: &hal::Device,
    descriptor_layout: &hal::DescriptorLayout,
) -> Result<hal::Pipeline> {
    let layout = hal::PipelineLayout {
        descriptors: &[descriptor_layout],
        push_constant: Some(vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            size: mem::size_of::<Mat4>() as u32,
            offset: 0,
        }),
    };
    let shader = hal::Shader::new(
        device,
        &hal::ShaderRequest {
            stage: vk::ShaderStageFlags::COMPUTE,
            source: vk_shader_macros::include_glsl!(
                "src/shaders/mesh/gbuffer.comp.glsl",
                kind: comp,
            ),
        },
    )?;
    let specializations = hal::Specializations::default();
    let shader_stage =
        hal::ShaderStage { shader: &shader, specializations: &specializations };
    let pipeline = hal::Pipeline::compute(device, &layout, shader_stage)?;
    shader.destroy(device);
    Ok(pipeline)
}

fn create_cull_pipeline(
    device: &hal::Device,
    descriptor_layout: &hal::DescriptorLayout,
    phase: DrawPhase,
) -> Result<hal::Pipeline> {
    let layout = hal::PipelineLayout {
        descriptors: &[descriptor_layout],
        push_constant: Some(vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            size: mem::size_of::<CullData>() as u32,
            offset: 0,
        }),
    };
    let shader = hal::Shader::new(
        device,
        &hal::ShaderRequest {
            stage: vk::ShaderStageFlags::COMPUTE,
            source: vk_shader_macros::include_glsl!(
                "src/shaders/mesh/draw_cull.comp.glsl",
                kind: comp,
            ),
        },
    )?;
    let specializations = match phase {
        DrawPhase::Pre => hal::Specializations::default(),
        DrawPhase::Post => {
            hal::Specializations::default().entry(&1u32.to_le_bytes())
        }
    };
    let shader_stage =
        hal::ShaderStage { shader: &shader, specializations: &specializations };
    let pipeline = hal::Pipeline::compute(device, &layout, shader_stage)?;
    shader.destroy(device);
    Ok(pipeline)
}

pub(super) fn create_descriptor(
    device: &hal::Device,
    mesh_phase: &MeshPhase,
    constants: &Constants,
    scene: &Scene,
    data: &mut hal::DescriptorData,
) -> hal::Descriptor {
    let texture_views = scene.textures.iter().map(hal::Image::full_view);
    data.builder(device, &mesh_phase.descriptor_layout)
        .uniform_buffer(&constants.buffer)
        .storage_buffer(&scene.vertices)
        .storage_buffer(&scene.meshes)
        .storage_buffer(&scene.instances)
        .storage_buffer(&scene.draw_commands)
        .storage_buffer(&scene.materials)
        .tlas(&scene.tlas)
        .combined_image_samplers(&scene.texture_sampler, texture_views)
        .build()
}

pub(super) fn create_depth_reduce_descriptors(
    device: &hal::Device,
    mesh_phase: &MeshPhase,
    render_targets: &RenderTargets,
    data: &mut hal::DescriptorData,
) -> Vec<hal::Descriptor> {
    let layout = &mesh_phase.depth_reduce_descriptor_layout;
    (0..mesh_phase.depth_pyramid.mip_level_count)
        .map(|mip_level| {
            let input = if let Some(input_level) = mip_level.checked_sub(1) {
                mesh_phase.depth_pyramid.view(&hal::ImageViewRequest {
                    mip_level_count: 1,
                    base_mip_level: input_level,
                })
            } else {
                render_targets.depth.full_view()
            };
            let output =
                mesh_phase.depth_pyramid.view(&hal::ImageViewRequest {
                    mip_level_count: 1,
                    base_mip_level: mip_level,
                });
            data.builder(device, layout)
                .storage_image(&output)
                .combined_image_sampler(&mesh_phase.depth_sampler, input)
                .build()
        })
        .collect()
}

pub(crate) fn create_cull_descriptor(
    device: &hal::Device,
    mesh_phase: &MeshPhase,
    constants: &Constants,
    scene: &Scene,
    data: &mut hal::DescriptorData,
) -> hal::Descriptor {
    let depth_pyramid = mesh_phase.depth_pyramid.full_view();
    data.builder(device, &mesh_phase.cull_descriptor_layout)
        .uniform_buffer(&constants.buffer)
        .storage_buffer(&scene.meshes)
        .storage_buffer(&scene.draws)
        .storage_buffer(&scene.instances)
        .storage_buffer(&scene.draw_commands)
        .storage_buffer(&scene.draw_count)
        .combined_image_sampler(&mesh_phase.depth_sampler, depth_pyramid)
        .build()
}

pub(crate) fn create_gbuffer_descriptor(
    device: &hal::Device,
    mesh_phase: &MeshPhase,
    constants: &Constants,
    render_targets: &RenderTargets,
    scene: &Scene,
    data: &mut hal::DescriptorData,
) -> hal::Descriptor {
    let texture_views = scene.textures.iter().map(hal::Image::full_view);
    data.builder(device, &mesh_phase.gbuffer_descriptor_layout)
        .uniform_buffer(&constants.buffer)
        .storage_buffer(&scene.instances)
        .storage_buffer(&scene.draws)
        .storage_buffer(&scene.indices)
        .storage_buffer(&scene.vertices)
        .storage_buffer(&scene.materials)
        .storage_image(&render_targets.visibility.full_view())
        .storage_image(&render_targets.gbuffer0.full_view())
        .storage_image(&render_targets.gbuffer1.full_view())
        .storage_image(&render_targets.gbuffer2.full_view())
        .combined_image_samplers(&scene.texture_sampler, texture_views)
        .build()
}

fn cull<'a>(
    device: &hal::Device,
    command_buffer: &mut hal::CommandBuffer<'a>,
    descriptors: &Descriptors,
    mesh_phase: &'a MeshPhase,
    phase: DrawPhase,
    scene: &Scene,
) {
    let pipeline = match phase {
        DrawPhase::Pre => &mesh_phase.pre_cull_pipeline,
        DrawPhase::Post => &mesh_phase.post_cull_pipeline,
    };

    let cull_data = CullData { total_draw_count: scene.total_draw_count };

    if let DrawPhase::Post = phase {
        command_buffer.pipeline_barriers(
            device,
            &[],
            &[hal::BufferBarrier {
                buffer: &scene.draw_count,
                src: hal::Access::INDIRECT_READ,
                dst: hal::Access::TRANSFER_DST,
            }],
        );
    }

    command_buffer
        .fill_buffer(device, &scene.draw_count, 0)
        .pipeline_barriers(
            device,
            &[hal::ImageBarrier {
                image: &mesh_phase.depth_pyramid,
                new_layout: vk::ImageLayout::GENERAL,
                mip_levels: hal::MipLevels::All,
                src: match phase {
                    DrawPhase::Pre => hal::Access::NONE,
                    DrawPhase::Post => hal::Access::COMPUTE_WRITE,
                },
                dst: hal::Access::COMPUTE_READ,
            }],
            &[
                hal::BufferBarrier {
                    buffer: &scene.draw_count,
                    src: hal::Access::TRANSFER_DST,
                    dst: hal::Access::COMPUTE_WRITE,
                },
                hal::BufferBarrier {
                    buffer: &scene.draw_commands,
                    src: hal::Access::INDIRECT_READ
                        | hal::Access {
                            stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                            access: vk::AccessFlags2::SHADER_READ,
                        },
                    dst: hal::Access::COMPUTE_WRITE,
                },
            ],
        )
        .bind_pipeline(device, pipeline)
        .bind_descriptor(device, pipeline, &descriptors.cull)
        .push_constants(device, pipeline, bytemuck::bytes_of(&cull_data))
        .dispatch(device, scene.total_draw_count.div_ceil(64), 1, 1);
}

pub(super) fn depth_reduce<'a>(
    device: &hal::Device,
    command_buffer: &mut hal::CommandBuffer<'a>,
    descriptors: &Descriptors,
    mesh_phase: &'a MeshPhase,
) {
    let pipeline = &mesh_phase.depth_reduce_pipeline;
    let vk::Extent3D { width, height, .. } = mesh_phase.depth_pyramid.extent;

    for (output_level, descriptor) in
        descriptors.depth_reduce.iter().enumerate()
    {
        let width = (width >> output_level).max(1);
        let height = (height >> output_level).max(1);

        let extent = Vec2 { x: width as f32, y: height as f32 };

        command_buffer
            .bind_pipeline(device, pipeline)
            .bind_descriptor(device, pipeline, descriptor)
            .push_constants(device, pipeline, bytemuck::bytes_of(&extent))
            .dispatch(device, width.div_ceil(32), height.div_ceil(32), 1)
            .pipeline_barriers(
                device,
                &[hal::ImageBarrier {
                    image: &mesh_phase.depth_pyramid,
                    new_layout: vk::ImageLayout::GENERAL,
                    mip_levels: hal::MipLevels::Levels {
                        base: output_level as u32,
                        count: 1,
                    },
                    src: hal::Access {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    },
                    dst: hal::Access {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_SAMPLED_READ,
                    },
                }],
                &[],
            );
    }
}

fn draw<'a>(
    device: &hal::Device,
    command_buffer: &mut hal::CommandBuffer<'a>,
    swapchain_image: &'a hal::Image,
    descriptors: &Descriptors,
    mesh_phase: &'a MeshPhase,
    render_targets: &'a RenderTargets,
    scene: &Scene,
    phase: DrawPhase,
) {
    let extent = vk::Extent2D {
        width: swapchain_image.extent.width,
        height: swapchain_image.extent.height,
    };

    let (depht_load, color_load) = match phase {
        DrawPhase::Pre => {
            let depth_load = hal::Load::Clear(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });
            let color_load = hal::Load::Clear(vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            });
            (depth_load, color_load)
        }
        DrawPhase::Post => (hal::Load::Load, hal::Load::Load),
    };

    command_buffer
        .bind_pipeline(device, &mesh_phase.pipeline)
        .set_viewport(
            device,
            &[vk::Viewport {
                x: 0.0,
                y: extent.height as f32,
                width: extent.width as f32,
                height: -(extent.height as f32),
                min_depth: 0.0,
                max_depth: 1.0,
            }],
        )
        .set_scissor(
            device,
            &[vk::Rect2D { offset: vk::Offset2D { x: 0, y: 0 }, extent }],
        )
        .bind_index_buffer(device, &scene.indices)
        .bind_descriptor(device, &mesh_phase.pipeline, &descriptors.mesh_phase)
        .begin_rendering(
            device,
            &hal::BeginRendering {
                depth_attachment: Some(hal::Attachment {
                    view: render_targets
                        .depth
                        .view(&hal::ImageViewRequest::BASE),
                    load: depht_load,
                }),
                color_attachments: &[
                    hal::Attachment {
                        view: swapchain_image
                            .view(&hal::ImageViewRequest::BASE),
                        load: color_load,
                    },
                    hal::Attachment {
                        view: render_targets
                            .visibility
                            .view(&hal::ImageViewRequest::BASE),
                        load: color_load,
                    },
                ],
                extent,
            },
        )
        .draw_indexed_indirect_count(
            device,
            &scene.draw_commands,
            &scene.draw_count,
            scene.total_draw_count,
            mem::size_of::<DrawCommand>() as u32,
            0,
        )
        .end_rendering(device);
}

fn generate_gbuffer<'a>(
    device: &hal::Device,
    command_buffer: &mut hal::CommandBuffer<'a>,
    descriptors: &Descriptors,
    mesh_phase: &'a MeshPhase,
    render_targets: &'a RenderTargets,
    constants: &Constants,
) {
    let vk::Extent3D { width, height, .. } = render_targets.visibility.extent;
    let width = width.div_ceil(32);
    let height = height.div_ceil(32);

    let mut ray_matrix = constants.data.proj_view;
    ray_matrix.col_mut(3)[0] = 0.0;
    ray_matrix.col_mut(3)[1] = 0.0;
    ray_matrix.col_mut(3)[2] = 0.0;
    ray_matrix = ray_matrix.inverse();

    command_buffer
        .bind_pipeline(device, &mesh_phase.gbuffer_pipeline)
        .bind_descriptor(
            device,
            &mesh_phase.gbuffer_pipeline,
            &descriptors.gbuffer,
        )
        .push_constants(
            device,
            &mesh_phase.gbuffer_pipeline,
            bytemuck::bytes_of(&ray_matrix),
        )
        .dispatch(device, width, height, 1);
}

pub(super) fn render<'a>(
    device: &hal::Device,
    command_buffer: &mut hal::CommandBuffer<'a>,
    swapchain_image: &'a hal::Image,
    descriptors: &Descriptors,
    mesh_phase: &'a MeshPhase,
    render_targets: &'a RenderTargets,
    scene: &Scene,
    constants: &Constants,
) {
    cull(
        device,
        command_buffer,
        descriptors,
        mesh_phase,
        DrawPhase::Pre,
        scene,
    );

    let buffer_barriers = [
        hal::BufferBarrier {
            buffer: &scene.draw_count,
            src: hal::Access::COMPUTE_WRITE,
            dst: hal::Access::INDIRECT_READ,
        },
        hal::BufferBarrier {
            buffer: &scene.draw_commands,
            src: hal::Access::COMPUTE_WRITE,
            dst: hal::Access::INDIRECT_READ,
        },
    ];

    let image_barriers = [
        hal::ImageBarrier {
            image: &render_targets.depth,
            new_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            mip_levels: hal::MipLevels::All,
            src: hal::Access::NONE,
            dst: hal::Access::DEPTH_BUFFER_RENDER,
        },
        hal::ImageBarrier {
            image: &swapchain_image,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            mip_levels: hal::MipLevels::All,
            src: hal::Access::NONE,
            dst: hal::Access::COLOR_BUFFER_RENDER,
        },
        hal::ImageBarrier {
            image: &render_targets.visibility,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            mip_levels: hal::MipLevels::All,
            src: hal::Access::NONE,
            dst: hal::Access::COLOR_BUFFER_RENDER,
        },
    ];

    command_buffer.pipeline_barriers(device, &image_barriers, &buffer_barriers);

    draw(
        device,
        command_buffer,
        swapchain_image,
        descriptors,
        mesh_phase,
        render_targets,
        scene,
        DrawPhase::Pre,
    );

    command_buffer.pipeline_barriers(
        device,
        &[hal::ImageBarrier {
            image: &render_targets.depth,
            new_layout: vk::ImageLayout::GENERAL,
            mip_levels: hal::MipLevels::All,
            src: hal::Access::DEPTH_BUFFER_RENDER,
            dst: hal::Access::DEPTH_BUFFER_READ,
        }],
        &[],
    );

    depth_reduce(device, command_buffer, descriptors, mesh_phase);

    cull(
        device,
        command_buffer,
        descriptors,
        mesh_phase,
        DrawPhase::Post,
        scene,
    );

    command_buffer.pipeline_barriers(device, &image_barriers, &buffer_barriers);

    draw(
        device,
        command_buffer,
        swapchain_image,
        descriptors,
        mesh_phase,
        render_targets,
        scene,
        DrawPhase::Post,
    );

    fn gbuffer_barrier<'a>(gbuffer: &'a hal::Image) -> hal::ImageBarrier<'a> {
        hal::ImageBarrier {
            image: gbuffer,
            new_layout: vk::ImageLayout::GENERAL,
            mip_levels: hal::MipLevels::All,
            src: hal::Access::NONE,
            dst: hal::Access::COMPUTE_WRITE,
        }
    }

    command_buffer.pipeline_barriers(
        device,
        &[
            hal::ImageBarrier {
                image: &render_targets.visibility,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                mip_levels: hal::MipLevels::All,
                src: hal::Access::COLOR_BUFFER_RENDER,
                dst: hal::Access::COMPUTE_READ,
            },
            gbuffer_barrier(&render_targets.gbuffer0),
            gbuffer_barrier(&render_targets.gbuffer1),
            gbuffer_barrier(&render_targets.gbuffer2),
        ],
        &[],
    );

    generate_gbuffer(
        device,
        command_buffer,
        descriptors,
        mesh_phase,
        render_targets,
        constants,
    );
}
