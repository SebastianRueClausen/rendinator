use anyhow::Result;
use ash::vk;

use std::mem;

use crate::command::*;
use crate::core::*;
use crate::resource::*;
use crate::camera::Camera;
use crate::light::Lights;

use rendi_math::prelude::*;

const CUBE_VERTICES: [Vec3; 36] = [
    Vec3::new(-1.0, 1.0, 1.0),
    Vec3::new(-1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(-1.0, 1.0, 1.0),

    Vec3::new(-1.0, -1.0, -1.0),
    Vec3::new(-1.0, -1.0, 1.0),
    Vec3::new(-1.0, 1.0, 1.0),
    Vec3::new(-1.0, 1.0, 1.0),
    Vec3::new(-1.0, 1.0, -1.0),
    Vec3::new(-1.0, -1.0, -1.0),

    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(1.0, -1.0, 1.0),

    Vec3::new(-1.0, -1.0, -1.0),
    Vec3::new(-1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, -1.0, -1.0),
    Vec3::new(-1.0, -1.0, -1.0),

    Vec3::new(-1.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, 1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(1.0, 1.0, -1.0),
    Vec3::new(-1.0, 1.0, -1.0),
    Vec3::new(-1.0, 1.0, 1.0),

    Vec3::new(-1.0, -1.0, 1.0),
    Vec3::new(-1.0, -1.0, -1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(1.0, -1.0, 1.0),
    Vec3::new(-1.0, -1.0, -1.0),
    Vec3::new(1.0, -1.0, -1.0),
];

pub struct CubeMap {
    pub image_view: Res<ImageView>,
    pub vertex_buffer: Res<Buffer>,
}

impl CubeMap {
    pub fn new(
        renderer: &Renderer,
        pool: &ResourcePool,
        image_view: Res<ImageView>,
    ) -> Result<Self> {
        let vertex_data: &[u8] = bytemuck::cast_slice(&CUBE_VERTICES);

        let staging = {
            let buffer = pool.create_buffer(MemoryLocation::Cpu, &BufferInfo {
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                size: vertex_data.len() as u64
            })?;

            buffer.get_mapped()?.fill(vertex_data);

            buffer
        };

        let vertex_buffer = {
            pool.create_buffer(MemoryLocation::Gpu, &BufferInfo {
                usage: vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::VERTEX_BUFFER,
                size: vertex_data.len() as u64
            })?
        };

        renderer.transfer_with(|recorder|
            recorder.copy_buffers(staging.clone(), vertex_buffer.clone())
        )?;

        Ok(Self { image_view, vertex_buffer })
    }
}

struct Generator {
    pipeline: Res<ComputePipeline>,
    descriptor: Res<DescSet>,

    size: u32,
}

impl Generator {
    fn new(
        renderer: &Renderer,
        image_view: Res<ImageView>,
        sampler: Res<Sampler>,
        lights: &Lights,
    ) -> Result<Self> {
        let pool = &renderer.static_pool;
        let layout = pool.create_desc_layout(&[
            DescLayoutSlot {
                binding: 0,
                ty: vk::DescriptorType::STORAGE_IMAGE,
                count: rendi_shader::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 1,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                count: rendi_shader::DescCount::Single,
            },
        ])?;

        let descriptor = pool.create_desc_set(layout.clone(), &[
            DescBinding::Image(sampler.clone(), vk::ImageLayout::GENERAL, image_view.clone()),
            DescBinding::Buffer(lights.info_buffer.clone()),
        ])?;

        let code = include_bytes_aligned_as!(u32, "../assets/shaders/skybox.comp.spv");
        let shader = pool.create_shader_module("main", code)?;
        let prog = pool.create_compute_prog(shader)?;

        let pipeline = pool.create_compute_pipeline(prog, &[])?;
        let size = image_view.image().extent(0).width;

        Ok(Self { pipeline, descriptor, size })
    }

    fn generate(&self, renderer: &Renderer) -> Result<()> {
        renderer.compute_with(|recorder| {
            recorder.bind_descs(&DescBindInfo {
                bind_point: vk::PipelineBindPoint::COMPUTE,
                layout: self.pipeline.layout(),
                descs: &[self.descriptor.clone()],
            });

            recorder.dispatch(self.pipeline.clone(), [
                self.size / 8, self.size / 8, 6,
            ]);
        })
    }
}

pub struct Skybox {
    pub cube_map: CubeMap,
    pub pipeline: Res<RasterPipeline>,
    pub descriptor: Res<DescSet>, 

    #[allow(dead_code)]
    generator: Generator,
}

impl Skybox {
    pub fn new(renderer: &Renderer, target_info: RenderTargetInfo, lights: &Lights) -> Result<Self> {
        let pool = &renderer.static_pool;
        let size = 64;

        let image = pool.create_image(MemoryLocation::Gpu, &ImageInfo {
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE,
            aspect_flags: vk::ImageAspectFlags::COLOR,
            extent: vk::Extent3D { width: size, height: size, depth: 1 },
            format: vk::Format::R16G16B16A16_SFLOAT,
            kind: ImageKind::CubeMap,
            mip_levels: 1,
        })?;

        renderer.transfer_with(|recorder| {
            recorder.image_barrier(&ImageBarrierInfo {
                flags: vk::DependencyFlags::BY_REGION,
                mips: image.mip_levels(),
                new_layout: vk::ImageLayout::GENERAL,
                src_stage: vk::PipelineStageFlags2::empty(),
                dst_stage: vk::PipelineStageFlags2::empty(),
                src_mask: vk::AccessFlags2::empty(),
                dst_mask: vk::AccessFlags2::empty(),
                image: image.clone(),
            });
        })?;

        let array_view = pool.create_image_view(&ImageViewInfo {
            view_type: vk::ImageViewType::TYPE_2D_ARRAY,
            mips: image.mip_levels(),
            image: image.clone(),
        })?;

        let sampler = pool.create_sampler()?;
        let generator = Generator::new(renderer, array_view, sampler.clone(), lights)?;

        generator.generate(renderer)?;

        renderer.transfer_with(|recorder| {
            recorder.image_barrier(&ImageBarrierInfo {
                flags: vk::DependencyFlags::BY_REGION,
                mips: image.mip_levels(),
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_stage: vk::PipelineStageFlags2::empty(),
                dst_stage: vk::PipelineStageFlags2::empty(),
                src_mask: vk::AccessFlags2::empty(),
                dst_mask: vk::AccessFlags2::empty(),
                image: image.clone(),
            });
        })?;

        let cube_view = pool.create_image_view(&ImageViewInfo {
            view_type: vk::ImageViewType::CUBE,
            mips: image.mip_levels(),
            image: image.clone(),
        })?;

        let cube_map = CubeMap::new(renderer, pool, cube_view.clone())?;

        let layout = pool.create_desc_layout(&[
            DescLayoutSlot {
                binding: 0,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: rendi_shader::DescCount::Single,
            },
        ])?;

        let descriptor = pool.create_desc_set(layout.clone(), &[
            DescBinding::Image(
                sampler, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL, cube_view.clone(),
            )
        ])?;

        let depth_stencil_info = &vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
            .depth_write_enable(false)
            .depth_test_enable(true);

        let vert_code = include_bytes_aligned_as!(u32, "../assets/shaders/skybox.vert.spv");
        let frag_code = include_bytes_aligned_as!(u32, "../assets/shaders/skybox.frag.spv");

        let vert_shader = pool.create_shader_module("main", vert_code)?;
        let frag_shader = pool.create_shader_module("main", frag_code)?;

        let prog = pool.create_raster_prog(vert_shader, frag_shader)?;

        let cull_mode = vk::CullModeFlags::FRONT;

        let vertex_attributes = &[VertexAttribute {
            format: vk::Format::R32G32B32_SFLOAT,
            size: mem::size_of::<Vec3>() as vk::DeviceSize,
        }];

        let push_consts = &[PushConstRange {
            size: mem::size_of::<Mat4>() as vk::DeviceSize,
            stage: vk::ShaderStageFlags::VERTEX,
        }];

        let pipeline = pool.create_raster_pipeline(RasterPipelineInfo {
            render_target_info: target_info, 
            vertex_attributes,
            depth_stencil_info,
            push_consts,
            cull_mode,
            prog,
        })?;

        Ok(Self {
            cube_map,
            descriptor,
            pipeline,
            generator,
        })
    }
}

pub fn draw(skybox: &Skybox, camera: &Camera, recorder: &DrawRecorder) {
    recorder.bind_vertex_buffer(skybox.cube_map.vertex_buffer.clone());
    recorder.bind_raster_pipeline(skybox.pipeline.clone());

    recorder.bind_descs(&DescBindInfo {
        bind_point: vk::PipelineBindPoint::GRAPHICS,
        layout: skybox.pipeline.layout(),
        descs: &[skybox.descriptor.clone()],
    });

    let transform = camera.proj * Mat4::from_mat3(Mat3::from_mat4(camera.view));

    recorder.push_consts(skybox.pipeline.layout(), &[PushConst {
        stage: vk::ShaderStageFlags::VERTEX,
        bytes: bytemuck::bytes_of(&transform),
    }]);

    recorder.draw(36, 0);
}
