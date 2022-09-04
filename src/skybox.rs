use anyhow::Result;
use glam::{Mat4, Vec3, UVec3};
use ash::vk;

use crate::core::*;
use crate::resource::*;
use crate::light::Lights;

use std::mem;

struct Generator {
    pipeline: ComputePipeline,
    descriptor: DescriptorSet,

    size: u32,
}

impl Generator {
    fn new(
        renderer: &Renderer,
        image_view: Res<ImageView>,
        sampler: Res<TextureSampler>,
        lights: &Lights,
        pool: &ResourcePool,
    ) -> Result<Self> {
        let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
        ])?);

        let descriptor = DescriptorSet::new_single(&renderer, layout.clone(), &[
            DescriptorBinding::Image(sampler.clone(), [image_view.clone()]),
            DescriptorBinding::Buffer([lights.light_buffer.clone()]),
        ])?;

        let code = include_bytes_aligned_as!(u32, "../assets/shaders/skybox.comp.spv");
        let shader = ShaderModule::new(&renderer, "main", code)?;

        let layout = pool.alloc(
            PipelineLayout::new(&renderer, &[], &[layout])?
        );

        let pipeline = ComputePipeline::new(&renderer, layout, &shader)?;
        let size = image_view.image().extent(0).width;

        Ok(Self { pipeline, descriptor, size })
    }

    fn generate(&self, renderer: &Renderer) -> Result<()> {
        renderer.compute_with(|recorder| {
            recorder.bind_descriptor_sets(
                0,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline.layout(),
                &[&self.descriptor],
            );

            let dim = UVec3::new(self.size / 8, self.size / 8, 6);
            recorder.dispatch(&self.pipeline, dim);
        })
    }
}

pub struct Skybox {
    pub cube_map: CubeMap,
    pub pipeline: GraphicsPipeline,
    pub descriptor: DescriptorSet, 

    #[allow(dead_code)]
    generator: Generator,
}

impl Skybox {
    pub fn new(renderer: &Renderer, lights: &Lights, pool: &ResourcePool) -> Result<Self> {
        let size = 64;

        let image = Image::new(renderer, pool, vk::MemoryPropertyFlags::DEVICE_LOCAL, &ImageReq {
            mip_levels: 1,
            extent: vk::Extent3D { width: size, height: size, depth: 1 },
            format: vk::Format::R16G16B16A16_SFLOAT,
            kind: ImageKind::CubeMap,
        })?;

        renderer.transfer_with(|recorder| {
            recorder.transition_image_layout(&image, vk::ImageLayout::GENERAL);
        })?;

        let array_view =
            ImageView::new(renderer, pool, image.clone(), vk::ImageViewType::TYPE_2D_ARRAY)?;

        let sampler = pool.alloc(TextureSampler::new(renderer)?);
        let generator = Generator::new(renderer, array_view, sampler.clone(), lights, pool)?;

        generator.generate(renderer)?;

        renderer.transfer_with(|recorder| {
            recorder.transition_image_layout(&image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        })?;

        let cube_view = ImageView::new(renderer, pool, image.clone(), vk::ImageViewType::CUBE)?;
        let cube_map = CubeMap::new(renderer, pool, cube_view.clone())?;

        let layout = pool.alloc(DescriptorSetLayout::new(renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: None,
            },
        ])?);

        let descriptor = DescriptorSet::new_per_frame(&renderer, layout.clone(), &[
            DescriptorBinding::Image(sampler, [
                cube_view.clone(), 
                cube_view.clone(), 
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

        Ok(Self { cube_map, descriptor, pipeline, generator })
    }
}
