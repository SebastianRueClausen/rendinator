use std::mem;

use ash::vk::{self};
use eyre::Result;
use glam::Vec2;

use crate::command::{
    Access, Attachment, BeginRendering, BufferBarrier, CommandBuffer,
    ImageBarrier, Load, MipLevels,
};
use crate::constants::Constants;
use crate::descriptor::{
    Descriptor, DescriptorData, DescriptorLayout, DescriptorLayoutBuilder,
};
use crate::device::Device;
use crate::render_targets::RenderTargets;
use crate::resources::{
    image_memory, Image, ImageRequest, ImageViewRequest, Memory, Sampler,
    SamplerRequest,
};
use crate::scene::{DrawCommand, Scene};
use crate::shader::{
    GraphicsPipelineRequest, Pipeline, PipelineLayout, Shader, ShaderRequest,
    ShaderStage, Specializations,
};
use crate::swapchain::Swapchain;
use crate::{render_targets, Descriptors};

pub(crate) struct MeshPhase {
    pipeline: Pipeline,
    depth_reduce_pipeline: Pipeline,
    pre_cull_pipeline: Pipeline,
    post_cull_pipeline: Pipeline,
    descriptor_layout: DescriptorLayout,
    cull_descriptor_layout: DescriptorLayout,
    depth_reduce_descriptor_layout: DescriptorLayout,
    depth_pyramid: Image,
    depth_sampler: Sampler,
    memory: Memory,
}

impl MeshPhase {
    pub fn new(device: &Device, swapchain: &Swapchain) -> Result<Self> {
        let descriptor_layout = DescriptorLayoutBuilder::default()
            .binding(vk::DescriptorType::UNIFORM_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .array_binding(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1024)
            .build(device)?;
        let depth_reduce_descriptor_layout = DescriptorLayoutBuilder::default()
            .binding(vk::DescriptorType::STORAGE_IMAGE)
            .binding(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(device)?;
        let cull_descriptor_layout = DescriptorLayoutBuilder::default()
            .binding(vk::DescriptorType::UNIFORM_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .build(device)?;
        let vertex_shader = Shader::new(
            device,
            &ShaderRequest {
                stage: vk::ShaderStageFlags::VERTEX,
                source: vk_shader_macros::include_glsl!(
                    "src/shaders/mesh/vert.glsl",
                    kind: vert,
                ),
            },
        )?;
        let fragment_shader = Shader::new(
            device,
            &ShaderRequest {
                stage: vk::ShaderStageFlags::FRAGMENT,
                source: vk_shader_macros::include_glsl!(
                    "src/shaders/mesh/frag.glsl",
                    kind: frag,
                ),
            },
        )?;
        let pipeline_layout = PipelineLayout {
            descriptors: &[&descriptor_layout],
            push_constant: None,
        };
        let specializations = Specializations::default();
        let pipeline = Pipeline::graphics(
            device,
            &pipeline_layout,
            &GraphicsPipelineRequest {
                color_formats: &[swapchain.format],
                depth_format: Some(render_targets::DEPTH_FORMAT),
                cull_mode: vk::CullModeFlags::BACK,
                shaders: &[
                    ShaderStage {
                        shader: &vertex_shader,
                        specializations: &specializations,
                    },
                    ShaderStage {
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
        Ok(Self {
            pipeline,
            descriptor_layout,
            depth_reduce_descriptor_layout,
            depth_reduce_pipeline,
            cull_descriptor_layout,
            pre_cull_pipeline,
            post_cull_pipeline,
            depth_pyramid,
            memory,
            depth_sampler,
        })
    }

    pub fn destroy(&self, device: &Device) {
        self.descriptor_layout.destroy(device);
        self.depth_reduce_descriptor_layout.destroy(device);
        self.cull_descriptor_layout.destroy(device);
        self.pipeline.destroy(device);
        self.depth_reduce_pipeline.destroy(device);
        self.pre_cull_pipeline.destroy(device);
        self.post_cull_pipeline.destroy(device);
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
    device: &Device,
    swapchain: &Swapchain,
) -> Result<(Image, Memory)> {
    let vk::Extent2D { width, height } = swapchain.extent;
    // Find previous power of 2.
    let width = (width / 2).next_power_of_two();
    let height = (height / 2).next_power_of_two();
    let mip_level_count = (width.max(height) as f32).log2().ceil() as u32;
    let extent = vk::Extent3D { width, height, depth: 1 };
    let usage = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::STORAGE
        | vk::ImageUsageFlags::TRANSFER_SRC;
    let mut image = Image::new(
        device,
        &ImageRequest {
            format: vk::Format::R32_SFLOAT,
            mip_level_count,
            extent,
            usage,
        },
    )?;
    let memory =
        image_memory(device, &image, vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    for base_mip_level in 0..mip_level_count {
        image.add_view(
            device,
            ImageViewRequest { mip_level_count: 1, base_mip_level },
        )?;
    }
    image.add_view(
        device,
        ImageViewRequest { mip_level_count, base_mip_level: 0 },
    )?;
    Ok((image, memory))
}

fn create_depth_sampler(device: &Device) -> Result<Sampler> {
    Sampler::new(
        device,
        &SamplerRequest {
            filter: vk::Filter::LINEAR,
            address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            reduction_mode: Some(vk::SamplerReductionMode::MIN),
            max_anisotropy: None,
        },
    )
}

fn create_depth_reduce_pipeline(
    device: &Device,
    descriptor_layout: &DescriptorLayout,
) -> Result<Pipeline> {
    let layout = PipelineLayout {
        descriptors: &[descriptor_layout],
        push_constant: Some(vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            size: mem::size_of::<Vec2>() as u32,
            offset: 0,
        }),
    };
    let shader = Shader::new(
        device,
        &ShaderRequest {
            stage: vk::ShaderStageFlags::COMPUTE,
            source: vk_shader_macros::include_glsl!(
                "src/shaders/mesh/depth_reduce.comp.glsl",
                kind: comp,
            ),
        },
    )?;
    let specializations = Specializations::default();
    let shader_stage =
        ShaderStage { shader: &shader, specializations: &specializations };
    let pipeline = Pipeline::compute(device, &layout, shader_stage)?;
    shader.destroy(device);
    Ok(pipeline)
}

fn create_cull_pipeline(
    device: &Device,
    descriptor_layout: &DescriptorLayout,
    phase: DrawPhase,
) -> Result<Pipeline> {
    let layout = PipelineLayout {
        descriptors: &[descriptor_layout],
        push_constant: Some(vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::COMPUTE,
            size: mem::size_of::<CullData>() as u32,
            offset: 0,
        }),
    };
    let shader = Shader::new(
        device,
        &ShaderRequest {
            stage: vk::ShaderStageFlags::COMPUTE,
            source: vk_shader_macros::include_glsl!(
                "src/shaders/mesh/draw_cull.comp.glsl",
                kind: comp,
            ),
        },
    )?;
    let specializations = match phase {
        DrawPhase::Pre => Specializations::default(),
        DrawPhase::Post => {
            Specializations::default().entry(&1u32.to_le_bytes())
        }
    };
    let shader_stage =
        ShaderStage { shader: &shader, specializations: &specializations };
    let pipeline = Pipeline::compute(device, &layout, shader_stage)?;
    shader.destroy(device);
    Ok(pipeline)
}

pub(super) fn create_descriptor(
    device: &Device,
    mesh_phase: &MeshPhase,
    constants: &Constants,
    scene: &Scene,
    data: &mut DescriptorData,
) -> Descriptor {
    let texture_views = scene.textures.iter().map(Image::full_view);
    data.builder(device, &mesh_phase.descriptor_layout)
        .uniform_buffer(&constants.buffer)
        .storage_buffer(&scene.vertices)
        .storage_buffer(&scene.meshes)
        .storage_buffer(&scene.instances)
        .storage_buffer(&scene.draw_commands)
        .storage_buffer(&scene.materials)
        .combined_image_samplers(&scene.texture_sampler, texture_views)
        .build()
}

pub(super) fn create_depth_reduce_descriptors(
    device: &Device,
    mesh_phase: &MeshPhase,
    render_targets: &RenderTargets,
    data: &mut DescriptorData,
) -> Vec<Descriptor> {
    let layout = &mesh_phase.depth_reduce_descriptor_layout;
    (0..mesh_phase.depth_pyramid.mip_level_count)
        .map(|mip_level| {
            let input = if let Some(input_level) = mip_level.checked_sub(1) {
                mesh_phase.depth_pyramid.view(&ImageViewRequest {
                    mip_level_count: 1,
                    base_mip_level: input_level,
                })
            } else {
                render_targets.depth.full_view()
            };
            let output = mesh_phase.depth_pyramid.view(&ImageViewRequest {
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
    device: &Device,
    mesh_phase: &MeshPhase,
    constants: &Constants,
    scene: &Scene,
    data: &mut DescriptorData,
) -> Descriptor {
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

fn cull<'a>(
    device: &Device,
    command_buffer: &mut CommandBuffer<'a>,
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
            &[BufferBarrier {
                buffer: &scene.draw_count,
                src: Access::INDIRECT_READ,
                dst: Access::TRANSFER_DST,
            }],
        );
    }

    command_buffer
        .fill_buffer(device, &scene.draw_count, 0)
        .pipeline_barriers(
            device,
            &[ImageBarrier {
                image: &mesh_phase.depth_pyramid,
                new_layout: vk::ImageLayout::GENERAL,
                mip_levels: MipLevels::All,
                src: match phase {
                    DrawPhase::Pre => Access::NONE,
                    DrawPhase::Post => Access::COMPUTE_WRITE,
                },
                dst: Access::COMPUTE_READ,
            }],
            &[
                BufferBarrier {
                    buffer: &scene.draw_count,
                    src: Access::TRANSFER_DST,
                    dst: Access::COMPUTE_WRITE,
                },
                BufferBarrier {
                    buffer: &scene.draw_commands,
                    src: Access::INDIRECT_READ
                        | Access {
                            stage: vk::PipelineStageFlags2::VERTEX_SHADER,
                            access: vk::AccessFlags2::SHADER_READ,
                        },
                    dst: Access::COMPUTE_WRITE,
                },
            ],
        )
        .bind_pipeline(device, pipeline)
        .bind_descriptor(device, pipeline, &descriptors.cull)
        .push_constants(device, pipeline, bytemuck::bytes_of(&cull_data))
        .dispatch(device, scene.total_draw_count.div_ceil(64), 1, 1);
}

pub(super) fn depth_reduce<'a>(
    device: &Device,
    command_buffer: &mut CommandBuffer<'a>,
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
                &[ImageBarrier {
                    image: &mesh_phase.depth_pyramid,
                    new_layout: vk::ImageLayout::GENERAL,
                    mip_levels: MipLevels::Levels {
                        base: output_level as u32,
                        count: 1,
                    },
                    src: Access {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_STORAGE_WRITE,
                    },
                    dst: Access {
                        stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                        access: vk::AccessFlags2::SHADER_SAMPLED_READ,
                    },
                }],
                &[],
            );
    }
}

fn draw<'a>(
    device: &Device,
    command_buffer: &mut CommandBuffer<'a>,
    swapchain_image: &'a Image,
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
            let depth_load = Load::Clear(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 0.0,
                    stencil: 0,
                },
            });
            let color_load = Load::Clear(vk::ClearValue {
                color: vk::ClearColorValue { float32: [0.0; 4] },
            });
            (depth_load, color_load)
        }
        DrawPhase::Post => (Load::Load, Load::Load),
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
            &BeginRendering {
                depth_attachment: Some(Attachment {
                    view: render_targets.depth.view(&ImageViewRequest::BASE),
                    load: depht_load,
                }),
                color_attachments: &[Attachment {
                    view: swapchain_image.view(&ImageViewRequest::BASE),
                    load: color_load,
                }],
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

pub(super) fn render<'a>(
    device: &Device,
    command_buffer: &mut CommandBuffer<'a>,
    swapchain_image: &'a Image,
    descriptors: &Descriptors,
    mesh_phase: &'a MeshPhase,
    render_targets: &'a RenderTargets,
    scene: &Scene,
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
        BufferBarrier {
            buffer: &scene.draw_count,
            src: Access::COMPUTE_WRITE,
            dst: Access::INDIRECT_READ,
        },
        BufferBarrier {
            buffer: &scene.draw_commands,
            src: Access::COMPUTE_WRITE,
            dst: Access::INDIRECT_READ,
        },
    ];

    let image_barriers = [
        ImageBarrier {
            image: &render_targets.depth,
            new_layout: vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL,
            mip_levels: MipLevels::All,
            src: Access::NONE,
            dst: Access::DEPTH_BUFFER_RENDER,
        },
        ImageBarrier {
            image: &swapchain_image,
            new_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            mip_levels: MipLevels::All,
            src: Access::NONE,
            dst: Access::COLOR_BUFFER_RENDER,
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
        &[ImageBarrier {
            image: &render_targets.depth,
            new_layout: vk::ImageLayout::GENERAL,
            mip_levels: MipLevels::All,
            src: Access::DEPTH_BUFFER_RENDER,
            dst: Access::DEPTH_BUFFER_READ,
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
}
