use glam::{Vec3, Vec2, Mat4};
use anyhow::Result;
use ash::vk;

use std::{array, mem};

use crate::light::Lights;
use crate::core::*;
use crate::resource::*;
use crate::camera::CameraUniforms;
use crate::skybox::Skybox;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct InstanceData {
    #[allow(dead_code)]
    transform: Mat4,

    #[allow(dead_code)]
    inverse_transpose_transform: Mat4,
}

/// Draw command used in draw indirect.
///
/// These correspond to a single primitive.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DrawCommand {
    command: vk::DrawIndexedIndirectCommand,

    albedo_map: u32,
    specular_map: u32,
    normal_map: u32,
}

unsafe impl bytemuck::NoUninit for DrawCommand {}

#[repr(align(16))]
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Primitive {
    center: Vec3,
    radius: f32,

    /// The vertex offset of the indices in the index buffer.
    vertex_offset: u32,

    /// The first index in the index buffer belonging to this primitive.
    first_index: u32,

    /// The amount of indices used by this primitive.
    index_count: u32,

    /// The instance index.
    instance: u32,

    /// The index of the albedo texture.
    albedo_map: u32,

    /// The index of the specular texture.
    specular_map: u32,

    /// The index of the normal texture.
    normal_map: u32,
}

unsafe impl bytemuck::NoUninit for Primitive {}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
pub struct DrawCount {
    /// The amount of draw commands.
    pub command_count: u32, 

    /// The amount of primitives.
    pub primitive_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
pub struct FrustrumInfo {
    pub z_near: f32,
    pub z_far: f32,

    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
}

pub struct Scene {
    pub render_pipeline: GraphicsPipeline,

    pub light_descriptor: DescriptorSet,
    pub descriptor: DescriptorSet,

    pub cull_descriptor: DescriptorSet,
    pub cull_pipeline: ComputePipeline,

    pub primitive_count: u32,

    /// Vertex buffer containing all vertex data for all objects in the scene.
    pub vertex_buffer: Res<Buffer>,

    /// Index buffer containing all vertex data for objects in the scene.
    pub index_buffer: Res<Buffer>,

    /// Buffer containing [`InstanceData`] for each model.
    pub instance_buffer: Res<Buffer>,

    /// [`DrawCommand`] for each primitive that should be drawn.
    pub draw_buffers: [Res<Buffer>; FRAMES_IN_FLIGHT],

    /// [`DrawCount`] for each frame in flight.
    pub draw_count_buffers: [Res<Buffer>; FRAMES_IN_FLIGHT],

    /// Buffer containing [`Primitive`] for every primitive in the scene.
    pub primitive_buffer: Res<Buffer>,

    /// The format of the indices in `index_buffer`.
    pub index_format: asset::IndexFormat,
}

impl Scene {
    pub fn from_scene_asset(
        renderer: &Renderer,
        pool: &ResourcePool,
        camera_uniforms: &CameraUniforms,
        skybox: &Skybox,
        lights: &Lights,
        scene: &asset::Scene,
    ) -> Result<Self> {
        let instance_data: Vec<_> = scene.instances
            .iter()
            .map(|instance| InstanceData {
                transform: instance.transform,
                inverse_transpose_transform: instance
                    .transform
                    .inverse()
                    .transpose()
            })
            .collect();

        let primitives: Vec<Primitive> = scene.instances
            .iter()
            .enumerate()
            .flat_map(|(i, instance)| {
                scene.meshes[instance.mesh].primitives
                    .iter()
                    .map(move |prim| {
                        let material = &scene.materials[prim.material];

                        Primitive {
                            center: prim.bounding_sphere.center,
                            radius: prim.bounding_sphere.radius,
                            albedo_map: material.albedo_map as u32,
                            normal_map: material.normal_map as u32,
                            specular_map: material.specular_map as u32,

                            vertex_offset: prim.vertex_start,
                            first_index: prim.index_start,
                            index_count: prim.index_count,
                            instance: i as u32,
                        }
                    })
            })
            .collect();

        let draw_count = DrawCount {
            // Filled out in cull shader.
            command_count: 0,
            primitive_count: primitives.len() as u32,
        };

        //
        // Create staging buffers.
        //

        let vertex_data = bytemuck::cast_slice(scene.vertices.as_slice());
        let index_data = bytemuck::cast_slice(scene.indices.as_slice());
        let instance_data = bytemuck::cast_slice(instance_data.as_slice());
        let primitive_data = bytemuck::cast_slice(primitives.as_slice());
        let draw_count_data = bytemuck::bytes_of(&draw_count);

        let staging_pool = ResourcePool::new();

        let memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let primitive_staging = Buffer::new(renderer, &staging_pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: primitive_data.len() as vk::DeviceSize,
        })?;

        let instance_staging = Buffer::new(renderer, &staging_pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: instance_data.len() as vk::DeviceSize,
        })?;

        let vertex_staging = Buffer::new(renderer, &staging_pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: vertex_data.len() as vk::DeviceSize,
        })?;

        let index_staging = Buffer::new(renderer, &staging_pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: index_data.len() as vk::DeviceSize,
        })?;

        let draw_count_staging = Buffer::new(renderer, &staging_pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: draw_count_data.len() as vk::DeviceSize,
        })?;

        primitive_staging.get_mapped()?.fill(primitive_data);
        instance_staging.get_mapped()?.fill(instance_data);
        vertex_staging.get_mapped()?.fill(vertex_data);
        index_staging.get_mapped()?.fill(index_data);
        draw_count_staging.get_mapped()?.fill(draw_count_data);

        //
        // Create device buffers and copy staging buffers.
        //

        let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        let instance_buffer = Buffer::new(renderer, pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            size: instance_data.len() as vk::DeviceSize,
        })?;

        let primitive_buffer = Buffer::new(renderer, pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            size: primitive_data.len() as vk::DeviceSize,
        })?;

        let vertex_buffer = Buffer::new(renderer, pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::VERTEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            size: vertex_data.len() as vk::DeviceSize,
        })?;

        let index_buffer = Buffer::new(renderer, pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            size: index_data.len() as vk::DeviceSize,
        })?;

        let draw_count_buffers: [_; FRAMES_IN_FLIGHT] = array::try_from_fn(|_| {
            Buffer::new(renderer, pool, memory_flags, &BufferReq {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDIRECT_BUFFER,
                size: mem::size_of::<DrawCount>() as vk::DeviceSize,
            })
        })?;

        renderer.transfer_with(|recorder| {
            recorder.copy_buffers(&instance_staging, &instance_buffer);
            recorder.copy_buffers(&primitive_staging, &primitive_buffer);
            recorder.copy_buffers(&vertex_staging, &vertex_buffer);
            recorder.copy_buffers(&index_staging, &index_buffer);

            for buffer in &draw_count_buffers {
                recorder.copy_buffers(&draw_count_staging, &buffer);
            }
        })?;

        let draw_buffers: [_; FRAMES_IN_FLIGHT] = array::try_from_fn(|_| {
            Buffer::new(renderer, pool, memory_flags, &BufferReq {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDIRECT_BUFFER,
                size: (primitives.len() * mem::size_of::<DrawCommand>()) as vk::DeviceSize,
            })
        })?;

        //
        // Create staging buffer for textures and upload the raw texture data.
        //

        let memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let staging: Result<Vec<Vec<_>>> = scene.textures
            .iter()
            .map(|texture| {
                texture.mips
                    .iter()
                    .map(|data| {
                        let buffer = Buffer::new(renderer, &staging_pool, memory_flags, &BufferReq {
                            usage: vk::BufferUsageFlags::TRANSFER_SRC,
                            size: data.len() as vk::DeviceSize,
                        })?;

                        buffer.get_mapped()?.fill(data);

                        Ok(buffer)
                    })
                    .collect()
            })
            .collect();

        let staging = staging?;

        let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        let images: Result<Vec<_>> = scene.textures
            .iter()
            .map(|texture| {
                Image::new(renderer, pool, memory_flags, &ImageReq {
                    mip_levels: texture.mip_levels(),
                    format: texture.format.into(),
                    kind: ImageKind::Texture,
                    extent: vk::Extent3D {
                        width: texture.width,
                        height: texture.height,
                        depth: 1,
                    },
                })
            })
            .collect();

        let images = images?;

        renderer.transfer_with(|recorder| {
            for image in images.iter() {
                recorder.transition_image_layout(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            }

            for (levels, dst) in staging.iter().zip(images.iter()) {
                for (level, src) in levels.iter().enumerate() {
                    recorder.copy_buffer_to_image(src, dst, level as u32);
                }
            }

            for image in images.iter() {
                recorder.transition_image_layout(image, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            }
        })?;

        let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: None,
            },
        ])?);

        let light_descriptor = DescriptorSet::new_per_frame(&renderer, layout, &[
            DescriptorBinding::Buffer([
                lights.cluster_info.buffer.clone(), 
                lights.cluster_info.buffer.clone(), 
            ]),
            DescriptorBinding::Buffer([
                lights.light_buffer.clone(),
                lights.light_buffer.clone(),
            ]),
            DescriptorBinding::Buffer(
                lights.light_mask_buffers.clone()
            ),
        ])?;

        let sampler = pool.alloc(TextureSampler::new(&renderer)?);

        let descriptor_layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::VERTEX,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::VERTEX,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: Some(images.len() as u32),
            },
        ])?);

        let descriptor = DescriptorSet::new_per_frame(&renderer, descriptor_layout.clone(), &[
            DescriptorBinding::Buffer(camera_uniforms.view_buffers.clone()),
            DescriptorBinding::Buffer([
                camera_uniforms.proj_buffer.clone(),
                camera_uniforms.proj_buffer.clone(),
            ]),
            DescriptorBinding::Buffer([
                instance_buffer.clone(),
                instance_buffer.clone(),
            ]),
            DescriptorBinding::Buffer(
                draw_buffers.clone(),
            ),
            DescriptorBinding::Image(sampler.clone(), [
                skybox.cube_map.image.clone(),
                skybox.cube_map.image.clone(),
            ]),
            DescriptorBinding::VariableImageArray(sampler.clone(), [
                &images,
                &images,
            ]),
        ])?;

        let render_pipeline = {
            let vertex_code = include_bytes_aligned_as!(u32, "../assets/shaders/pbr.vert.spv");
            let fragment_code = include_bytes_aligned_as!(u32, "../assets/shaders/pbr.frag.spv");

            let vertex_module = ShaderModule::new(&renderer, "main", vertex_code)?;
            let fragment_module = ShaderModule::new(&renderer, "main", fragment_code)?;

            let depth_stencil_info = &vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_test_enable(true)
                .depth_write_enable(true)
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);

            let layout = pool.alloc(PipelineLayout::new(&renderer, &[], &[
                descriptor_layout.clone(),
                light_descriptor.layout.clone(),
            ])?);

            let pipeline = GraphicsPipeline::new(&renderer, GraphicsPipelineReq {
                vertex_attributes: &[
                    vk::VertexInputAttributeDescription {
                        format: vk::Format::R32G32B32_SFLOAT,
                        binding: 0,
                        location: 0,
                        offset: 0,
                    },
                    vk::VertexInputAttributeDescription {
                        format: vk::Format::R32G32B32_SFLOAT,
                        binding: 0,
                        location: 1,
                        offset: mem::size_of::<Vec3>() as u32,
                    },
                    vk::VertexInputAttributeDescription {
                        format: vk::Format::R32G32_SFLOAT,
                        binding: 0,
                        location: 2,
                        offset: mem::size_of::<[Vec3; 2]>() as u32
                    },
                    vk::VertexInputAttributeDescription {
                        format: vk::Format::R32G32B32A32_SFLOAT,
                        binding: 0,
                        location: 3,
                        offset: (mem::size_of::<[Vec3; 2]>() + mem::size_of::<Vec2>())as u32,
                    },
                ],
                vertex_bindings: &[vk::VertexInputBindingDescription {
                    binding: 0,
                    stride: mem::size_of::<asset::Vertex>() as u32,
                    input_rate: vk::VertexInputRate::VERTEX,
                }],
                cull_mode: vk::CullModeFlags::BACK,
                vertex_shader: &vertex_module,
                fragment_shader: &fragment_module,
                depth_stencil_info,
                layout,
            })?;

            pipeline
        };

        let descriptor_layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
        ])?);

        let cull_descriptor = DescriptorSet::new_per_frame(&renderer, descriptor_layout.clone(), &[
            DescriptorBinding::Buffer(draw_count_buffers.clone()),
            DescriptorBinding::Buffer(draw_buffers.clone()),
            DescriptorBinding::Buffer([
                primitive_buffer.clone(),
                primitive_buffer.clone(),
            ]),
            DescriptorBinding::Buffer([
                instance_buffer.clone(),
                instance_buffer.clone(),
            ]),
        ])?;

        let cull_pipeline = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/draw_cull.comp.spv");

            let shader = ShaderModule::new(&renderer, "main", code)?;

            let push_consts = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(mem::size_of::<FrustrumInfo>() as u32)
                .offset(0)
                .build()];

            let layout = pool.alloc(PipelineLayout::new(&renderer, &push_consts, &[
                descriptor_layout.clone(),
            ])?);

            ComputePipeline::new(&renderer, layout, &shader)?
        };

        let index_format = scene.index_format;
        let primitive_count = primitives.len() as u32;

        Ok(Self {
            cull_descriptor,
            cull_pipeline,
            primitive_count,
            light_descriptor,
            descriptor,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            draw_buffers,
            draw_count_buffers,
            primitive_buffer,
            instance_buffer,
            index_format,
        })
    }
}
