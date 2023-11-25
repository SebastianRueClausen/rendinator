use std::mem;

use ash::vk::{self};
use eyre::Result;

use crate::command::{Attachment, BeginRendering, CommandBuffer, Load};
use crate::constants::Constants;
use crate::descriptor::{
    Descriptor, DescriptorBuilder, DescriptorData, DescriptorLayout,
    DescriptorLayoutBuilder,
};
use crate::device::Device;
use crate::render_targets::RenderTargets;
use crate::resources::{Image, ImageViewRequest};
use crate::scene::{self, Scene};
use crate::shader::{
    GraphicsPipelineRequest, Pipeline, PipelineLayout, Shader, ShaderRequest,
};
use crate::swapchain::Swapchain;
use crate::{render_targets, Descriptors};

pub(crate) struct MeshPhase {
    pub pipeline: Pipeline,
    pub descriptor_layout: DescriptorLayout,
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
        let pipeline = Pipeline::graphics(
            device,
            &pipeline_layout,
            &GraphicsPipelineRequest {
                color_formats: &[swapchain.format],
                depth_format: Some(render_targets::DEPTH_FORMAT),
                shaders: &[&vertex_shader, &fragment_shader],
                cull_mode: vk::CullModeFlags::BACK,
            },
        )?;
        fragment_shader.destroy(device);
        vertex_shader.destroy(device);
        Ok(Self { pipeline, descriptor_layout })
    }

    pub fn destroy(&self, device: &Device) {
        self.descriptor_layout.destroy(device);
        self.pipeline.destroy(device);
    }
}

pub(super) fn create_descriptor(
    device: &Device,
    mesh_phase: &MeshPhase,
    constants: &Constants,
    scene: &Scene,
    data: &mut DescriptorData,
) -> Descriptor {
    let texture_views = scene.textures.iter().map(Image::full_view);
    DescriptorBuilder::new(device, &mesh_phase.descriptor_layout, data)
        .uniform_buffer(&constants.buffer)
        .storage_buffer(&scene.vertices)
        .storage_buffer(&scene.meshes)
        .storage_buffer(&scene.instances)
        .storage_buffer(&scene.draws)
        .storage_buffer(&scene.materials)
        .combined_image_samplers(&scene.texture_sampler, texture_views)
        .set()
}

pub(super) fn render(
    device: &Device,
    command_buffer: &mut CommandBuffer,
    swapchain_image: &Image,
    descriptors: &Descriptors,
    mesh_phase: &MeshPhase,
    render_targets: &RenderTargets,
    scene: &Scene,
) {
    let extent = vk::Extent2D {
        width: swapchain_image.extent.width,
        height: swapchain_image.extent.height,
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
                    load: Load::Clear(vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: 0,
                        },
                    }),
                }),
                color_attachments: &[Attachment {
                    view: swapchain_image.view(&ImageViewRequest::BASE),
                    load: Load::Clear(vk::ClearValue {
                        color: vk::ClearColorValue { float32: [0.0; 4] },
                    }),
                }],
                extent,
            },
        )
        .draw_indexed_indirect(
            device,
            &scene.draws,
            scene.draw_count,
            mem::size_of::<scene::Draw>() as u32,
            0,
        )
        .end_rendering(device);
}
