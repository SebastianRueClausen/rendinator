use glam::{Vec3, Vec2, Mat4};
use anyhow::Result;
use ash::vk;

use std::{array, mem};

use crate::light::Lights;
use crate::core::*;
use crate::resource::{self, *};
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
            command_count: 10,
            primitive_count: primitives.len() as u32,
        };

        let primitive_buffer_size = (primitives.len() * mem::size_of::<Primitive>()) as u64;
        let draw_buffer_size = (primitives.len() * mem::size_of::<DrawCommand>()) as u64;
        let draw_count_buffer_size = mem::size_of::<DrawCount>() as u64;

        let instance_buffer_size = (instance_data.len() * mem::size_of::<InstanceData>()) as u64;
        let vertex_buffer_size = (scene.vertices.len() * mem::size_of::<asset::Vertex>()) as u64;
        let index_buffer_size = scene.indices.len() as u64;

        let staging = {
            let reqs = vec![
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: vertex_buffer_size },
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: index_buffer_size },
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: instance_buffer_size },
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: primitive_buffer_size },
            ];

            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let (buffers, block) = resource::create_buffers(
                &renderer,
                &pool,
                &reqs,
                memory_flags,
                4,
            )?;

            let mapped = MappedMemory::new(block.clone())?;

            let mut data = vec![
                bytemuck::cast_slice(scene.vertices.as_slice()),
                bytemuck::cast_slice(scene.indices.as_slice()),
                bytemuck::cast_slice(instance_data.as_slice()),
                bytemuck::cast_slice(primitives.as_slice()),
            ];

            for _ in 0..FRAMES_IN_FLIGHT {
                data.push(bytemuck::bytes_of(&draw_count))
            }

            for (src, dst) in data.iter().zip(buffers.iter()) {
                mapped.get_buffer_data(dst).copy_from_slice(&src);
            }

            buffers
        };

        let buffers = {
            let mut reqs = vec![
                BufferReq {
                    usage: vk::BufferUsageFlags::VERTEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    size: vertex_buffer_size,
                },
                BufferReq {
                    usage: vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    size: index_buffer_size,
                },
                // Instance buffer.
                BufferReq {
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    size: instance_buffer_size, 
                },
                // Primitive buffer.
                BufferReq {
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    size: primitive_buffer_size, 
                },
            ];

            for _ in 0..FRAMES_IN_FLIGHT {
                reqs.push(BufferReq {
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    size: draw_count_buffer_size, 
                });
            }
            
            for _ in 0..FRAMES_IN_FLIGHT {
                reqs.push(BufferReq {
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    size: draw_buffer_size, 
                });
            }

            let (buffers, _) = resource::create_buffers(
                &renderer,
                &pool,
                &reqs,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                4,
            )?;

            buffers
        };

        renderer.transfer_with(|recorder| {
            for (src, dst) in staging[0..4].iter().zip(buffers.iter()) {
                recorder.copy_buffers(src, dst);
            }
        })?;

        let mut buffers = buffers.into_iter();

        let vertex_buffer = buffers.next().unwrap();
        let index_buffer = buffers.next().unwrap();
        let instance_buffer = buffers.next().unwrap();
        let primitive_buffer = buffers.next().unwrap();

        let draw_count_buffers: [_; 2] = array::from_fn(|_| buffers.next().unwrap());
        let draw_buffers: [_; 2] = array::from_fn(|_| buffers.next().unwrap());

        let staging = {
            let create_infos: Vec<_> = scene.textures
                .iter()
                .map(|tex| BufferReq {
                    usage: vk::BufferUsageFlags::TRANSFER_SRC,
                    size: tex.data.len() as u64,
                })
                .collect();

            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

            let (staging, block) = resource::create_buffers(
                &renderer,
                &pool,
                &create_infos,
                memory_flags,
                4,
            )?;

            let mapped = MappedMemory::new(block.clone())?;
           
            scene.textures
                .iter()
                .map(|tex| tex.data.as_slice())
                .zip(staging.iter())
                .for_each(|(data, buffer)| {
                    mapped.get_buffer_data(buffer).copy_from_slice(&data);
                });

            staging
        };

        let mut images = {
            let image_reqs: Vec<_> = scene.textures
                .iter()
                .map(|tex| ImageReq {
                    format: tex.format.into(),
                    kind: ImageKind::Texture,
                    extent: vk::Extent3D {
                        width: tex.width,
                        height: tex.height,
                        depth: 1,
                    },
                })
                .collect();

            let (images, _) = resource::create_images(
                &renderer,
                &pool,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                &image_reqs,
            )?;

            images
        };

        renderer.transfer_with(|recorder| {
            for image in images.iter_mut() {
                recorder.transition_image_layout(image, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            }

            for (src, dst) in staging.iter().zip(images.iter()) {
                recorder.copy_buffer_to_image(src, dst);
            }

            for image in images.iter_mut() {
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
                lights.lights_buf.clone(),
                lights.lights_buf.clone(),
            ]),
            DescriptorBinding::Buffer(
                lights.light_mask_bufs.clone()
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
            DescriptorBinding::Buffer([
                camera_uniforms.view_uniform(0).clone(),
                camera_uniforms.view_uniform(1).clone(),
            ]),
            DescriptorBinding::Buffer([
                camera_uniforms.proj_uniform().clone(),
                camera_uniforms.proj_uniform().clone(),
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
