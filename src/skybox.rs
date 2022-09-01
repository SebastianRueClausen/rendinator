use anyhow::Result;
use glam::{Mat4, Vec3};
use ash::vk;

use crate::core::*;
use crate::resource::*;

use std::mem;

pub struct Skybox {
    pub cube_map: CubeMap,
    pub pipeline: GraphicsPipeline,
    pub descriptor: DescriptorSet, 
}

impl Skybox {
    pub fn new(renderer: &Renderer, pool: &ResourcePool, skybox: &asset::Skybox) -> Result<Self> {
        let image = Image::new(renderer, pool, vk::MemoryPropertyFlags::DEVICE_LOCAL, &ImageReq {
            extent: vk::Extent3D { width: skybox.width(), height: skybox.height(), depth: 1 },
            format: vk::Format::R8G8B8A8_SRGB,
            kind: ImageKind::CubeMap,
        })?;

        let staging = {
            let size: usize = skybox.images
                .iter()
                .map(|image| image.data.len())
                .sum();

            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let buffer = Buffer::new(renderer, pool, memory_flags, &BufferReq {
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                size: size as u64
            })?;

            let mapped = buffer.get_mapped()?;

            skybox.images.iter().fold(0, |start, image| {
                let end = start + image.data.len() as u64;
                mapped.fill_range(start..end, &image.data);

                end
            });

            buffer
        };

        renderer.transfer_with(|recorder| {
            recorder.transition_image_layout(&image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            recorder.copy_buffer_to_image(&staging, &image);
            recorder.transition_image_layout(&image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        })?;

        let cube_map = CubeMap::new(renderer, pool, image)?;

        let layout = pool.alloc(DescriptorSetLayout::new(renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: None,
            },
        ])?);

        let sampler = pool.alloc(TextureSampler::new(renderer)?);

        let descriptor = DescriptorSet::new_per_frame(&renderer, layout.clone(), &[
            DescriptorBinding::Image(sampler, [
                cube_map.image.clone(), 
                cube_map.image.clone(), 
            ]),
        ])?;

        let push_consts = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(mem::size_of::<Mat4>() as u32)
            .offset(0)
            .build()];

        let layout = pool.alloc(PipelineLayout::new(renderer, &push_consts, &[layout.clone()])?);

        let depth_stencil_info = &vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(true)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

        let vertex_code = include_bytes_aligned_as!(u32, "../assets/shaders/skybox.vert.spv");
        let fragment_code = include_bytes_aligned_as!(u32, "../assets/shaders/skybox.frag.spv");

        let vertex_module = ShaderModule::new(&renderer, "main", vertex_code)?;
        let fragment_module = ShaderModule::new(&renderer, "main", fragment_code)?;

        let cull_mode = vk::CullModeFlags::FRONT;

        let pipeline = GraphicsPipeline::new(&renderer, GraphicsPipelineReq {
            vertex_attributes: &[vk::VertexInputAttributeDescription {
                format: vk::Format::R32G32B32_SFLOAT,
                binding: 0,
                location: 0,
                offset: 0,
            }],
            vertex_bindings: &[vk::VertexInputBindingDescription {
                binding: 0,
                stride: mem::size_of::<Vec3>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
            vertex_shader: &vertex_module,
            fragment_shader: &fragment_module,
            depth_stencil_info,
            cull_mode,
            layout,
        })?;

        Ok(Self { cube_map, descriptor, pipeline })
    }
}
