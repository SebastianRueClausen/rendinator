use glam::{Vec3, Vec2, Mat4};
use anyhow::Result;
use ash::vk;

use std::mem;
use std::ops::Range;

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

#[repr(C)]
#[derive(Clone, Copy)]
pub struct DrawCommand {
    command: vk::DrawIndexedIndirectCommand,

    albedo_map: u32,
    specular_map: u32,
    normal_map: u32,
}

unsafe impl bytemuck::NoUninit for DrawCommand {}

pub struct Instance {
    draw_commands: Range<usize>,
}

pub struct Scene {
    pub render_pipeline: GraphicsPipeline,

    pub light_descriptor: DescriptorSet,
    pub descriptor: DescriptorSet,

    pub draw_count: u32,

    /// The instances to be rendered.
    ///
    /// Their index is determined by their index in the vector.
    pub instances: Vec<Instance>,

    /// Vertex buffer containing all vertex data for all objects in the scene.
    pub vertex_buffer: Res<Buffer>,

    /// Index buffer containing all vertex data for objects in the scene.
    pub index_buffer: Res<Buffer>,

    /// Buffer containing [`InstanceData`] for each model.
    pub instance_buffer: Res<Buffer>,

    pub draw_buffer: Res<Buffer>,

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

        let mut draw_commands = Vec::new();

        let instances: Vec<_> = scene.instances
            .iter()
            .enumerate()
            .map(|(i, instance)| {
                let mesh = &scene.meshes[instance.mesh];
                let first = draw_commands.len();

                draw_commands.extend(
                    mesh.primitives
                        .iter()
                        .map(|prim| {
                            let material = &scene.materials[prim.material];
                            DrawCommand {
                                albedo_map: material.albedo_map as u32,
                                normal_map: material.normal_map as u32,
                                specular_map: material.specular_map as u32,
                                command: vk::DrawIndexedIndirectCommand {
                                    first_index: prim.index_start,
                                    index_count: prim.index_count,

                                    vertex_offset: prim.vertex_start as i32,

                                    first_instance: i as u32,
                                    instance_count: 1,
                                },
                            }
                        })
                );

                Instance { draw_commands: first..draw_commands.len() }
            })
            .collect();

        let draw_buffer_size = (draw_commands.len() * mem::size_of::<DrawCommand>()) as u64;
        let instance_buffer_size = (instance_data.len() * mem::size_of::<InstanceData>()) as u64;
        let vertex_buffer_size = (scene.vertices.len() * mem::size_of::<asset::Vertex>()) as u64;
        let index_buffer_size = scene.indices.len() as u64;

        let staging = {
            let reqs = [
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: vertex_buffer_size },
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: index_buffer_size },
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: instance_buffer_size },
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: draw_buffer_size },
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

            let vertex_data: &[u8] = bytemuck::cast_slice(scene.vertices.as_slice());
            let instance_data: &[u8] = bytemuck::cast_slice(instance_data.as_slice());
            let draw_data: &[u8] = bytemuck::cast_slice(draw_commands.as_slice());
            let index_data = scene.indices.as_slice();

            for (data, buffer) in [vertex_data, index_data, instance_data, draw_data]
                .iter()
                .zip(buffers.iter())
            {
                mapped.get_buffer_data(buffer).copy_from_slice(&data);
            }

            buffers
        };

        let buffers = {
            let reqs = [
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
                BufferReq {
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    size: instance_buffer_size, 
                },
                BufferReq {
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST
                        | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    size: draw_buffer_size, 
                },
            ];

            let (buffers, _) = resource::create_buffers(
                &renderer,
                &pool,
                &reqs,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                4,
            )?;

            buffers
        };

        renderer.device.transfer_with(|recorder| {
            for (src, dst) in staging.iter().zip(buffers.iter()) {
                recorder.copy_buffers(src, dst);
            }
        })?;

        let vertex_buffer = buffers[0].clone();
        let index_buffer = buffers[1].clone();
        let instance_buffer = buffers[2].clone();
        let draw_buffer = buffers[3].clone();

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

        renderer.device.transfer_with(|recorder| {
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
            DescriptorBinding::Buffer([
                draw_buffer.clone(),
                draw_buffer.clone(),
            ]),
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

        let index_format = scene.index_format;
        let draw_count = draw_commands.len() as u32;

        Ok(Self {
            light_descriptor,
            descriptor,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            draw_buffer,
            draw_count,
            instance_buffer,
            instances,
            index_format,
        })
    }
}
