use glam::{Vec3, Vec2, Mat4};
use anyhow::Result;
use ash::vk;

use std::mem;
use std::ops::Index;

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

pub struct Model {
    pub index_start: u32,
    pub index_count: u32,
    pub vertex_offset: i32,
    pub material: usize,
}

pub struct Instance {
    /// The index of the model.
    pub model: usize,
}

pub struct Material {
    pub base_color: usize,
    pub normal: usize,
    pub metallic_roughness: usize,
    pub descriptor: DescriptorSet,
}

pub struct Materials {
    pub images: Vec<Res<Image>>, 
    pub sampler: Res<TextureSampler>,
    materials: Vec<Material>,
}

impl Index<usize> for Materials {
    type Output = Material;

    fn index(&self, idx: usize) -> &Self::Output {
        &self.materials[idx]
    }
}

pub struct Scene {
    pub render_pipeline: GraphicsPipeline,
    pub light_descriptor: DescriptorSet,
    pub materials: Materials,
    pub models: Vec<Model>,

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
        let instances: Vec<_> = scene.instances
            .iter()
            .map(|instance| InstanceData {
                transform: instance.transform,
                inverse_transpose_transform: instance
                    .transform
                    .inverse()
                    .transpose()
            })
            .collect();

        let vertex_size = (scene.vertices.len() * mem::size_of::<asset::Vertex>()) as u64;
        let instance_size = (instances.len() * mem::size_of::<InstanceData>()) as u64;
        let index_size = scene.indices.len() as u64;

        let staging = {
            let reqs = [
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: vertex_size },
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: index_size },
                BufferReq { usage: vk::BufferUsageFlags::TRANSFER_SRC, size: instance_size },
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
            let index_data = scene.indices.as_slice();

            let vertex_data: &[u8] = bytemuck::cast_slice(scene.vertices.as_slice());
            let instance_data: &[u8] = bytemuck::cast_slice(instances.as_slice());

            for (data, buffer) in [vertex_data, index_data, instance_data]
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
                    size: vertex_size,
                },
                BufferReq {
                    usage: vk::BufferUsageFlags::INDEX_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    size: index_size,
                },
                BufferReq {
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    size: instance_size, 
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

        let staging = {
            let create_infos: Vec<_> = scene.materials
                .iter()
                .flat_map(|mat| [
                    BufferReq {
                        usage: vk::BufferUsageFlags::TRANSFER_SRC,
                        size: mat.base_color.data.len() as u64,
                    },
                    BufferReq {
                        usage: vk::BufferUsageFlags::TRANSFER_SRC,
                        size: mat.normal.data.len() as u64,
                    },
                    BufferReq {
                        usage: vk::BufferUsageFlags::TRANSFER_SRC,
                        size: mat.metallic_roughness.data.len() as u64,
                    },
                ])
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
           
            scene.materials
                .iter()
                .flat_map(|mat| [
                    mat.base_color.data.as_slice(),
                    mat.normal.data.as_slice(),
                    mat.metallic_roughness.data.as_slice(),
                ])
                .zip(staging.iter())
                .for_each(|(data, buffer)| {
                    mapped.get_buffer_data(buffer).copy_from_slice(&data);
                });

            staging
        };

        let mut images = {
            let image_reqs: Vec<_> = scene.materials
                .iter()
                .flat_map(|mat| [
                    ImageReq {
                        format: mat.base_color.format.into(),
                        kind: ImageKind::Texture,
                        extent: vk::Extent3D {
                            width: mat.base_color.width,
                            height: mat.base_color.height,
                            depth: 1,
                        },
                    },
                    ImageReq {
                        format: mat.normal.format.into(),
                        kind: ImageKind::Texture,
                        extent: vk::Extent3D {
                            width: mat.normal.width,
                            height: mat.normal.height,
                            depth: 1,
                        },
                    },
                    ImageReq {
                        format: mat.metallic_roughness.format.into(),
                        kind: ImageKind::Texture,
                        extent: vk::Extent3D {
                            width: mat.metallic_roughness.width,
                            height: mat.metallic_roughness.height,
                            depth: 1,
                        },
                    },
                ])
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

        let instances: Vec<_> = scene.instances
            .iter()
            .map(|instance| Instance {
                model: instance.mesh,  
            })
            .collect();

        let models: Vec<_> = scene.meshes
            .iter()
            .map(|mesh| Model {
                vertex_offset: mesh.vertex_start as i32,
                index_start: mesh.index_start,
                index_count: mesh.index_count,
                material: mesh.material,
            })
            .collect();

        let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
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
            },
            LayoutBinding {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::VERTEX,
            },
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
            },
        ])?);

        let materials: Result<Vec<_>> = scene.materials
            .iter()
            .enumerate()
            .map(|(base, _)| {
                let base = base * 3;

                let base_color_index = base;
                let normal_index = base + 1;
                let metallic_roughness_index = base + 2;

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
                    DescriptorBinding::Image(sampler.clone(), [
                        skybox.cube_map.image.clone(),
                        skybox.cube_map.image.clone(),
                    ]),
                    DescriptorBinding::Image(sampler.clone(), [
                        images[base_color_index].clone(),
                        images[base_color_index].clone(),
                    ]),
                    DescriptorBinding::Image(sampler.clone(), [
                        images[normal_index].clone(),
                        images[normal_index].clone(),
                    ]),
                    DescriptorBinding::Image(sampler.clone(), [
                        images[metallic_roughness_index].clone(),
                        images[metallic_roughness_index].clone(),
                    ]),
                ])?;

                Ok(Material {
                    base_color: base_color_index,
                    metallic_roughness: metallic_roughness_index,
                    normal: normal_index,
                    descriptor,
                })
            })
            .collect();

        let materials = materials?;

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

        let materials = Materials { images, sampler, materials };
        let index_format = scene.index_format;

        Ok(Self {
            light_descriptor,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            instance_buffer,
            instances,
            models,
            materials,
            index_format,
        })
    }
}
