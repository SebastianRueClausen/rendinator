use glam::{Vec4, Mat4};
use anyhow::Result;
use ash::vk;

use std::mem;

use crate::light::Lights;
use crate::core::*;
use crate::resource::*;
use crate::camera::{Camera, CameraUniforms};

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
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct DrawCommand {
    // NOTE: This is the same layout as `vk::DrawIndexedIndirectCommand` and thus the order must
    // not be changed.
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    vertex_offset: u32,
    first_instance: u32,

    albedo_map: u32,
    specular_map: u32,
    normal_map: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy, bytemuck::NoUninit)]
struct Primitive {
    center: Vec4,
    radius: f32,

    _pad: u32,

    lods: [Lod; MAX_LOD_COUNT],

    /// The instance index.
    instance: u32,

    /// The vertex offset of the indices in the index buffer.
    vertex_offset: u32,

    /// The amount lods in `lods`.
    lod_count: u32,

    /// The index of the albedo texture.
    albedo_map: u32,

    /// The index of the specular texture.
    specular_map: u32,

    /// The index of the normal texture.
    normal_map: u32,
}

#[repr(C)]
#[derive(Default, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct Lod {
    first_index: u32,
    index_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct DrawCount {
    /// The amount of draw commands.
    pub command_count: u32, 

    /// The amount of primitives.
    pub primitive_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct CullInfo {
    z_near: f32,
    z_far: f32,

    frust_left: f32,
    frust_right: f32,
    frust_top: f32,
    frust_bottom: f32,

    lod_base: f32,
    lod_step: f32,
}

pub struct Scene {
    pub render_pipeline: Res<GraphicsPipeline>,
    pub cull_pipeline: Res<ComputePipeline>,

    pub descriptor: Res<DescriptorSet>,

    pub primitive_count: u32,

    /// Vertex buffer containing all vertex data for all objects in the scene.
    pub vertex_buffer: Res<Buffer>,

    /// Index buffer containing all vertex data for objects in the scene.
    pub index_buffer: Res<Buffer>,

    /// Buffer containing [`InstanceData`] for each model.
    pub instance_buffer: Res<Buffer>,

    /// [`DrawCommand`] for each primitive that should be drawn.
    pub draw_buffers: PerFrame<Res<Buffer>>,

    /// [`DrawCount`] for each frame in flight.
    pub draw_count_buffers: PerFrame<Res<Buffer>>,

    /// Buffer containing [`Primitive`] for every primitive in the scene.
    pub primitive_buffer: Res<Buffer>,
}

impl Scene {
    pub fn from_scene_asset(
        renderer: &Renderer,
        pool: &ResourcePool,
        camera_uniforms: &CameraUniforms,
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
                        let mut lods: [Lod; MAX_LOD_COUNT] = Default::default();
                        let mut lod_count = 0;

                        for (i, lod) in prim.lods.iter().take(MAX_LOD_COUNT).enumerate() {
                            lod_count += 1;

                            lods[i] = Lod {
                                first_index: lod.index_start,
                                index_count: lod.index_count,
                            };
                        }

                        let material = &scene.materials[prim.material];

                        Primitive {
                            center: prim.bounding_sphere.center.extend(1.0),
                            radius: prim.bounding_sphere.radius,

                            albedo_map: material.albedo_map as u32,
                            specular_map: material.specular_map as u32,
                            normal_map: material.normal_map as u32,

                            vertex_offset: prim.vertex_start,
                            instance: i as u32,

                            lod_count,
                            lods,

                            ..Primitive::default()
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
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            size: vertex_data.len() as vk::DeviceSize,
        })?;

        let index_buffer = Buffer::new(renderer, pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            size: index_data.len() as vk::DeviceSize,
        })?;

        let draw_count_buffers = PerFrame::try_from_fn(|_| {
            Buffer::new(renderer, pool, memory_flags, &BufferReq {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDIRECT_BUFFER,
                size: mem::size_of::<DrawCount>() as vk::DeviceSize,
            })
        })?;

        renderer.transfer_with(|recorder| {
            let buffers = [
                (instance_staging.clone(), instance_buffer.clone()),
                (primitive_staging.clone(), primitive_buffer.clone()),
                (vertex_staging.clone(), vertex_buffer.clone()),
                (index_staging.clone(), index_buffer.clone()),
            ];

            for (staging, buffer) in buffers {
                recorder.copy_buffers(staging, buffer);
            }

            for buffer in &draw_count_buffers {
                recorder.copy_buffers(draw_count_staging.clone(), buffer.clone());
            }
        })?;

        let draw_buffers = PerFrame::try_from_fn(|_| {
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
                recorder.transition_image_layout(
                    image.clone(),
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );
            }

            for (levels, dst) in staging.iter().zip(images.iter()) {
                for (level, src) in levels.iter().enumerate() {
                    recorder.copy_buffer_to_image(src.clone(), dst.clone(), level as u32);
                }
            }

            for image in images.iter() {
                recorder.transition_image_layout(
                    image.clone(),
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                );
            }
        })?;

        let views: Result<Vec<_>> = images
            .into_iter()
            .map(|image| {
                ImageView::new(renderer, pool, image, vk::ImageViewType::TYPE_2D)
            })
            .collect();
    
        let views = views?;

        let sampler = pool.alloc(TextureSampler::new(&renderer)?);

        let descriptor_layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
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
                stage: vk::ShaderStageFlags::VERTEX,
                array_count: None,
            },
            LayoutBinding {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: Some(views.len() as u32),
            },
        ])?);

        let descriptor = pool.alloc(DescriptorSet::new_per_frame(&renderer, descriptor_layout.clone(), &[
            DescriptorBinding::Buffer([instance_buffer.clone(), instance_buffer.clone()]),
            DescriptorBinding::Buffer(draw_buffers.clone().into()),
            DescriptorBinding::Buffer([primitive_buffer.clone(), primitive_buffer.clone()]),
            DescriptorBinding::Buffer(draw_count_buffers.clone().into()),
            DescriptorBinding::Buffer([vertex_buffer.clone(), vertex_buffer.clone()]),
            DescriptorBinding::VariableImageArray(sampler.clone(), [&views, &views]),
        ])?);

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
                camera_uniforms.descriptor.layout(),
                lights.descriptor.layout(),
                descriptor.layout(),
            ])?);

            pool.alloc(GraphicsPipeline::new(&renderer, GraphicsPipelineReq {
                vertex_attributes: &[],
                vertex_bindings: &[],
                cull_mode: vk::CullModeFlags::BACK,
                vertex_shader: &vertex_module,
                fragment_shader: &fragment_module,
                depth_stencil_info,
                layout,
            })?)
        };

        let cull_pipeline = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/draw_cull.comp.spv");

            let shader = ShaderModule::new(&renderer, "main", code)?;

            let push_consts = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(mem::size_of::<CullInfo>() as u32)
                .offset(0)
                .build()];

            let layout = pool.alloc(PipelineLayout::new(&renderer, &push_consts, &[
                descriptor.layout(),
            ])?);

            pool.alloc(ComputePipeline::new(&renderer, layout, &shader)?)
        };

        let primitive_count = primitives.len() as u32;

        Ok(Self {
            cull_pipeline,
            primitive_count,
            descriptor,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            draw_buffers,
            draw_count_buffers,
            primitive_buffer,
            instance_buffer,
        })
    }

    pub fn prepare_draw_buffers(
        &self,
        frame_index: FrameIndex,
        camera: &Camera,
        recorder: &CommandRecorder,
    ) {
        recorder.update_buffer(
            self.draw_count_buffers[frame_index].clone(),
            &DrawCount {
                command_count: 0,
                primitive_count: self.primitive_count,
            },
        );

        recorder.buffer_barrier(&BufferBarrierReq {
            buffer: self.draw_count_buffers[frame_index].clone(),
            src_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_mask: vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
            src_stage: vk::PipelineStageFlags2::TRANSFER,
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        });

        recorder.bind_descriptor_sets(&DescriptorBindReq {
            frame_index: Some(frame_index),
            bind_point: vk::PipelineBindPoint::COMPUTE,
            layout: self.cull_pipeline.layout(),
            descriptors: &[self.descriptor.clone()],
        });

        fn normalize_plane(plane: Vec4) -> Vec4 {
            plane / plane.truncate().length()
        }

        let horizontal = normalize_plane(camera.proj.row(3) + camera.proj.row(0));
        let vertical = normalize_plane(camera.proj.row(3) + camera.proj.row(1));

        let cull_info = CullInfo {
            z_near: camera.z_near,
            z_far: camera.z_far,

            frust_left: horizontal.x,
            frust_right: horizontal.y,
            frust_top: vertical.y,
            frust_bottom: vertical.z,

            lod_base: 10.0,
            lod_step: 2.0,
        };

        recorder.push_constants(
            self.cull_pipeline.layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            &cull_info,
        );

        recorder.dispatch(self.cull_pipeline.clone(), [
            self.primitive_count.div_ceil(64), 1, 1,
        ]);

        recorder.buffer_barrier(&BufferBarrierReq {
            buffer: self.draw_buffers[frame_index].clone(),
            src_mask: vk::AccessFlags2::SHADER_WRITE,
            dst_mask: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            dst_stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
        });
    }

    pub fn draw(
        &self,
        frame_index: FrameIndex,
        camera_uniforms: &CameraUniforms,
        lights: &Lights,
        recorder: &CommandRecorder,
    ) {
        recorder.bind_index_buffer(self.index_buffer.clone(), vk::IndexType::UINT32);
        recorder.bind_graphics_pipeline(self.render_pipeline.clone());

        recorder.bind_descriptor_sets(&DescriptorBindReq {
            frame_index: Some(frame_index),
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            layout: self.render_pipeline.layout(),
            descriptors: &[
                camera_uniforms.descriptor.clone(),
                lights.descriptor.clone(),
                self.descriptor.clone(),
            ],
        });
        
        recorder.draw_indexed_indirect_count(&IndexedIndirectDrawReq {
            draw_command_size: mem::size_of::<DrawCommand>() as vk::DeviceSize,
            draw_buffer: self.draw_buffers[frame_index].clone(),
            count_buffer: self.draw_count_buffers[frame_index].clone(),
            max_draw_count: self.primitive_count,
            count_offset: 0,
            draw_offset: 0,
        });
    }
}

const MAX_LOD_COUNT: usize = 8;
