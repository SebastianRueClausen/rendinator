use anyhow::Result;
use glam::{Mat3, Mat4, Vec3};
use ash::vk;

use crate::core::*;
use crate::resource::*;
use crate::camera::Camera;
use crate::light::Lights;

use std::mem;

struct Generator {
    pipeline: Res<ComputePipeline>,
    descriptor: Res<DescriptorSet>,

    size: u32,
}

impl Generator {
    fn new(
        renderer: &Renderer,
        image_view: Res<ImageView>,
        sampler: Res<TextureSampler>,
        lights: &Lights,
    ) -> Result<Self> {
        let pool = &renderer.static_pool;
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

        let descriptor = DescriptorSet::new(&renderer, pool, layout.clone(), &[
            DescriptorBinding::Image(
                sampler.clone(),
                vk::ImageLayout::GENERAL,
                image_view.clone(),
            ),
            DescriptorBinding::Buffer(lights.light_buffer.clone()),
        ])?;

        let code = include_bytes_aligned_as!(u32, "../assets/shaders/skybox.comp.spv");
        let shader = ShaderModule::new(&renderer, "main", code)?;

        let layout = pool.alloc(
            PipelineLayout::new(&renderer, &[], &[layout])?
        );

        let pipeline = pool.alloc(ComputePipeline::new(&renderer, layout, &shader)?);
        let size = image_view.image().extent(0).width;

        Ok(Self { pipeline, descriptor, size })
    }

    fn generate(&self, renderer: &Renderer) -> Result<()> {
        renderer.compute_with(|recorder| {
            recorder.bind_descriptor_sets(&DescriptorBindReq {
                bind_point: vk::PipelineBindPoint::COMPUTE,
                layout: self.pipeline.layout(),
                descriptors: &[self.descriptor.clone()],
            });

            recorder.dispatch(self.pipeline.clone(), [
                self.size / 8, self.size / 8, 6,
            ]);
        })
    }
}

pub struct Skybox {
    pub cube_map: CubeMap,
    pub pipeline: Res<GraphicsPipeline>,
    pub descriptor: Res<DescriptorSet>, 

    #[allow(dead_code)]
    generator: Generator,
}

impl Skybox {
    pub fn new(
        renderer: &Renderer,
        render_target_info: RenderTargetInfo,
        lights: &Lights,
    ) -> Result<Self> {
        let pool = &renderer.static_pool;
        let size = 64;

        let image = Image::new(renderer, pool, vk::MemoryPropertyFlags::DEVICE_LOCAL, &ImageReq {
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            aspect_flags: vk::ImageAspectFlags::COLOR,
            extent: vk::Extent3D { width: size, height: size, depth: 1 },
            format: vk::Format::R16G16B16A16_SFLOAT,
            kind: ImageKind::CubeMap,
            mip_levels: 1,
        })?;

        renderer.transfer_with(|recorder| {
            recorder.transition_image_layout(image.clone(), vk::ImageLayout::GENERAL);
        })?;

        let array_view = ImageView::new(renderer, pool, &ImageViewReq {
            view_type: vk::ImageViewType::TYPE_2D_ARRAY,
            mips: image.mip_levels(),
            image: image.clone(),
        })?;

        let sampler = pool.alloc(
            TextureSampler::new(renderer, vk::SamplerReductionMode::WEIGHTED_AVERAGE)?
        );

        let generator = Generator::new(renderer, array_view, sampler.clone(), lights)?;

        generator.generate(renderer)?;

        renderer.transfer_with(|recorder| {
            recorder.transition_image_layout(image.clone(), vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        })?;

        let cube_view = ImageView::new(renderer, pool, &ImageViewReq {
            view_type: vk::ImageViewType::CUBE,
            mips: image.mip_levels(),
            image: image.clone(),
        })?;

        let cube_map = CubeMap::new(renderer, pool, cube_view.clone())?;

        let layout = pool.alloc(DescriptorSetLayout::new(renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: None,
            },
        ])?);

        let descriptor = DescriptorSet::new(&renderer, pool, layout.clone(), &[
            DescriptorBinding::Image(
                sampler,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                cube_view.clone(),
            )
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

        let pipeline = pool.alloc(GraphicsPipeline::new(&renderer, GraphicsPipelineReq {
            render_target_info, 

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
        })?);

        Ok(Self { cube_map, descriptor, pipeline, generator })
    }

    pub fn draw(&self, camera: &Camera, recorder: &CommandRecorder) {
        recorder.bind_vertex_buffer(self.cube_map.vertex_buffer.clone());
        recorder.bind_graphics_pipeline(self.pipeline.clone());

        recorder.bind_descriptor_sets(&DescriptorBindReq {
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            layout: self.pipeline.layout(),
            descriptors: &[self.descriptor.clone()],
        });

        let transform = camera.proj * Mat4::from_mat3(Mat3::from_mat4(camera.view));

        recorder.push_constants(
            self.pipeline.layout(),
            vk::ShaderStageFlags::VERTEX,
            0,
            &transform,
        );

        recorder.draw(36, 0);
    }
}
