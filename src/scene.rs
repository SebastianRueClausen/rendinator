use glam::{Vec4, Mat4, UVec2};
use anyhow::Result;
use ash::vk;

use std::mem;

use crate::RenderTargets;
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
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
struct DrawCount {
    /// The amount of draw commands.
    pub command_count: u32, 

    /// The amount of primitives.
    pub primitive_count: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct CullInfo {
    frustrum_planes: [Vec4; 6],

    z_near: f32,
    z_far: f32,

    lod_base: f32,
    lod_step: f32,

    pyramid_width: f32,
    pyramid_height: f32,

    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct DepthReduceInfo {
    image_size: UVec2, 
    target: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct DepthResolveInfo {
    image_size: UVec2, 
    sample: u32,
}

struct DepthPyramid {
    /// Depth staging image.
    ///
    /// This is used to resolve the multisampled depth image into. And it's the first level of the
    /// depth pyramid.
    depth_staging: Res<ImageView>,

    /// The depth pyramid image. It has the dimensions of `depth_image` rounded down to the
    /// previous power of 2.
    pyramid: Res<Image>,

    /// [`ImageView`] of all mips of `pyramid`.
    mips: Vec<Res<ImageView>>,

    width: u32,
    height: u32,
}

impl DepthPyramid {
    fn new(
        renderer: &Renderer,
        render_targets: &RenderTargets,
        frame_index: FrameIndex,
    ) -> Result<Self> {
        let depth_image = render_targets.depth_images[frame_index].clone();
        let depth_extent = depth_image.image().extent(0);

        let width = prev_pow2(depth_extent.width);
        let height = prev_pow2(depth_extent.height);

        let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;

        let pyramid = {
            let mip_levels = (height.max(height) as f32).log2().floor() as u32;

            Image::new(renderer, &renderer.pool, memory_flags, &ImageReq {
                extent: vk::Extent3D { width, height, depth: 1 },
                aspect_flags: vk::ImageAspectFlags::COLOR,
                format: vk::Format::R32_SFLOAT,
                kind: ImageKind::Texture,
                mip_levels,
                usage,
            })?
        };

        let depth_staging = {
            let image = Image::new(renderer, &renderer.pool, memory_flags, &ImageReq {
                format: vk::Format::R32_SFLOAT,
                aspect_flags: vk::ImageAspectFlags::COLOR,
                kind: ImageKind::Texture,
                extent: depth_extent,
                mip_levels: 1,
                usage,
            })?;

            ImageView::new(renderer, &renderer.pool, &ImageViewReq {
                view_type: vk::ImageViewType::TYPE_2D,
                image: image.clone(),
                mips: image.mip_levels(),
            })?
        };

        renderer.transfer_with(|recorder| {
            recorder.image_barrier(&ImageBarrierReq {
                flags: vk::DependencyFlags::BY_REGION,
                src_stage: vk::PipelineStageFlags2::empty(),
                dst_stage: vk::PipelineStageFlags2::empty(),
                src_mask: vk::AccessFlags2::empty(),
                dst_mask: vk::AccessFlags2::empty(),
                new_layout: vk::ImageLayout::GENERAL,
                mips: pyramid.mip_levels(),
                image: pyramid.clone(),
            });

            recorder.image_barrier(&ImageBarrierReq {
                flags: vk::DependencyFlags::BY_REGION,
                src_stage: vk::PipelineStageFlags2::empty(),
                dst_stage: vk::PipelineStageFlags2::empty(),
                src_mask: vk::AccessFlags2::empty(),
                dst_mask: vk::AccessFlags2::empty(),
                new_layout: vk::ImageLayout::GENERAL,
                mips: depth_staging.image().mip_levels(),
                image: depth_staging.image().clone(),
            });
        })?;

        let mut mips = vec![depth_staging.clone()];

        for level in pyramid.mip_levels() {
            mips.push(
                ImageView::new(renderer, &renderer.pool, &ImageViewReq {
                    view_type: vk::ImageViewType::TYPE_2D,
                    mips: level..level + 1,
                    image: pyramid.clone(),
                })?
            );
        }

        Ok(Self { pyramid, mips, depth_staging, width, height })
    }
}

pub struct Scene {
    render_pipeline: Res<GraphicsPipeline>,
    cull_pipeline: Res<ComputePipeline>,

    descriptor: Res<DescriptorSet>,

    primitive_count: u32,

    depth_pyramids: PerFrame<DepthPyramid>,

    sampled_pyramid: Res<DescriptorSet>,    
    storage_pyramid: Res<DescriptorSet>,    

    depth_resolve_descriptor: Res<DescriptorSet>,

    depth_reduce: Res<ComputePipeline>,
    depth_resolve: Res<ComputePipeline>,

    /// Index buffer containing all vertex data for objects in the scene.
    index_buffer: Res<Buffer>,

    /// [`DrawCommand`] for each primitive that should be drawn.
    draw_buffers: PerFrame<Res<Buffer>>,

    /// [`DrawCount`] for each frame in flight.
    draw_count_buffers: PerFrame<Res<Buffer>>,

    draw_count_host_buffers: PerFrame<Res<Buffer>>,
}

impl Scene {
    pub fn from_scene_asset(
        renderer: &Renderer,
        render_targets: &RenderTargets,
        camera_uniforms: &CameraUniforms,
        lights: &Lights,
        scene: &asset::Scene,
    ) -> Result<Self> {
        let pool = &renderer.static_pool;

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

                        let asset::Material { albedo_map, specular_map, normal_map, .. } =
                            scene.materials[prim.material];

                        let (albedo_map, specular_map, normal_map) = (
                            albedo_map as u32, specular_map as u32, normal_map as u32,
                        );

                        Primitive {
                            center: prim.bounding_sphere.center.extend(1.0),
                            radius: prim.bounding_sphere.radius,
                            vertex_offset: prim.vertex_start,
                            instance: i as u32,

                            albedo_map,
                            specular_map,
                            normal_map,

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
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC,
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
                    aspect_flags: vk::ImageAspectFlags::COLOR,
                    usage: vk::ImageUsageFlags::TRANSFER_DST
                        | vk::ImageUsageFlags::SAMPLED,
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
                ImageView::new(renderer, pool, &ImageViewReq {
                    view_type: vk::ImageViewType::TYPE_2D,
                    mips: image.mip_levels(),
                    image: image.clone(),
                })
            })
            .collect();
    
        let views = views?;

        let sampler = pool.alloc(
            TextureSampler::new(&renderer, vk::SamplerReductionMode::WEIGHTED_AVERAGE)?
        );

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
            DescriptorBinding::VariableImageArray(
                sampler.clone(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                [&views, &views]
            ),
        ])?);

        let render_pipeline = {
            let vertex_code = include_bytes_aligned_as!(u32, "../assets/shaders/pbr.vert.spv");
            let fragment_code = include_bytes_aligned_as!(u32, "../assets/shaders/pbr.frag.spv");

            let vertex_module = ShaderModule::new(&renderer, "main", vertex_code)?;
            let fragment_module = ShaderModule::new(&renderer, "main", fragment_code)?;

            let depth_stencil_info = &vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_write_enable(true)
                .depth_test_enable(true);

            let layout = pool.alloc(PipelineLayout::new(&renderer, &[], &[
                camera_uniforms.descriptor.layout(),
                lights.descriptor.layout(),
                descriptor.layout(),
            ])?);

            pool.alloc(GraphicsPipeline::new(&renderer, GraphicsPipelineReq {
                color_format: render_targets.color_format(),
                depth_format: render_targets.depth_format(),
                sample_count: render_targets.sample_count(),
                cull_mode: vk::CullModeFlags::BACK,
                fragment_shader: &fragment_module,
                vertex_shader: &vertex_module,
                vertex_attributes: &[],
                vertex_bindings: &[],
                depth_stencil_info,
                layout,
            })?)
        };

        let depth_pyramids = PerFrame::try_from_fn(|frame_index| {
            DepthPyramid::new(renderer, render_targets, frame_index)
        })?;

        let sampled_pyramid = create_sampled_pyramid_descriptor(renderer, &depth_pyramids)?;

        let sampler = pool.alloc(
            TextureSampler::new(renderer, vk::SamplerReductionMode::WEIGHTED_AVERAGE)?
        );

        let storage_pyramid =
            create_storage_pyramid_descriptor(renderer, sampler.clone(), &depth_pyramids)?;

        let depth_resolve_descriptor =
            create_depth_resolve_descriptor(renderer, render_targets, &depth_pyramids, sampler.clone())?;

        let cull_pipeline = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/draw_cull.comp.spv");
            let shader = ShaderModule::new(&renderer, "main", code)?;

            let push_consts = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(mem::size_of::<CullInfo>() as u32)
                .offset(0)
                .build()];

            let layout = pool.alloc(PipelineLayout::new(&renderer, &push_consts, &[
                camera_uniforms.descriptor.layout(),
                descriptor.layout(),
                sampled_pyramid.layout(),
            ])?);

            pool.alloc(ComputePipeline::new(&renderer, layout, &shader)?)
        };

        let depth_reduce = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/depth_reduce.comp.spv");
            let shader = ShaderModule::new(&renderer, "main", code)?;

            let push_consts = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(mem::size_of::<DepthReduceInfo>() as u32)
                .offset(0)
                .build()];

            let layout = pool.alloc(PipelineLayout::new(&renderer, &push_consts, &[
                sampled_pyramid.layout(),
                storage_pyramid.layout(),
            ])?);

            pool.alloc(ComputePipeline::new(&renderer, layout, &shader)?)
        };

        let depth_resolve = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/depth_resolve.comp.spv");
            let shader = ShaderModule::new(&renderer, "main", code)?;

            let push_consts = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(mem::size_of::<DepthResolveInfo>() as u32)
                .offset(0)
                .build()];

            let layout = pool.alloc(PipelineLayout::new(&renderer, &push_consts, &[
                depth_resolve_descriptor.layout(),
            ])?);

            pool.alloc(ComputePipeline::new(&renderer, layout, &shader)?)
        };

        let draw_count_host_buffers = PerFrame::try_from_fn(|_| {
            let memory_flags =
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            Buffer::new(renderer, pool, memory_flags, &BufferReq {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                size: mem::size_of::<DrawCount>() as vk::DeviceSize,
            })
        })?;

        let primitive_count = primitives.len() as u32;

        Ok(Self {
            depth_resolve,
            depth_resolve_descriptor,
            sampled_pyramid,
            storage_pyramid,
            depth_pyramids,
            depth_reduce,
            cull_pipeline,
            primitive_count,
            descriptor,
            render_pipeline,
            index_buffer,
            draw_buffers,
            draw_count_buffers,
            draw_count_host_buffers,
        })
    }

    pub fn handle_resize(&mut self, renderer: &Renderer, render_targets: &RenderTargets) -> Result<()> {
        self.depth_pyramids = PerFrame::try_from_fn(|frame_index| {
            DepthPyramid::new(renderer, render_targets, frame_index)
        })?;

        self.sampled_pyramid = create_sampled_pyramid_descriptor(renderer, &self.depth_pyramids)?;

        let sampler = renderer.pool.alloc(
            TextureSampler::new(renderer, vk::SamplerReductionMode::WEIGHTED_AVERAGE)?
        );

        self.storage_pyramid =
            create_storage_pyramid_descriptor(renderer, sampler.clone(), &self.depth_pyramids)?;

        self.depth_resolve_descriptor = create_depth_resolve_descriptor(
            renderer,
            render_targets,
            &self.depth_pyramids,
            sampler.clone(),
        )?;

        Ok(())
    }

    pub fn prepare_draw_buffers(
        &self,
        frame_index: FrameIndex,
        render_targets: &RenderTargets,
        camera_uniforms: &CameraUniforms,
        camera: &Camera,
        recorder: &CommandRecorder,
    ) {
        let draw_count = DrawCount {
            command_count: 0,
            primitive_count: self.primitive_count,
        };

        recorder.update_buffer(self.draw_count_buffers[frame_index].clone(), &draw_count);

        recorder.buffer_barrier(&BufferBarrierReq {
            buffer: self.draw_count_buffers[frame_index].clone(),
            src_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_mask: vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
            src_stage: vk::PipelineStageFlags2::TRANSFER,
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        });

        //
        // Frustrum cull and generate draw buffers.
        //

        let pyramid = &self.depth_pyramids[frame_index];
        let depth_image = &render_targets.depth_images[frame_index];

        recorder.bind_descriptor_sets(&DescriptorBindReq {
            frame_index: Some(frame_index),
            bind_point: vk::PipelineBindPoint::COMPUTE,
            layout: self.cull_pipeline.layout(),
            descriptors: &[
                camera_uniforms.descriptor.clone(),
                self.descriptor.clone(),
                self.sampled_pyramid.clone(),
            ],
        });

        let cull_info = CullInfo {
            z_near: camera.z_near,
            z_far: camera.z_far,

            frustrum_planes: camera.frustrum_planes(),

            pyramid_width: pyramid.width as f32,
            pyramid_height: pyramid.height as f32,

            lod_base: 10.0,
            lod_step: 2.0,

            _pad1: 0.0,
            _pad2: 0.0,
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

        //
        // Resolve from depth image to depth staging image.
        //
       
        recorder.image_barrier(&ImageBarrierReq {
            flags: vk::DependencyFlags::BY_REGION,
            src_stage: vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            src_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_mask: vk::AccessFlags2::SHADER_READ,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image: depth_image.image().clone(),
            mips: depth_image.image().mip_levels(),
        });

        recorder.image_barrier(&ImageBarrierReq {
            flags: vk::DependencyFlags::BY_REGION,
            src_stage: vk::PipelineStageFlags2::empty(),
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            src_mask: vk::AccessFlags2::empty(),
            dst_mask: vk::AccessFlags2::SHADER_WRITE,
            new_layout: vk::ImageLayout::GENERAL,
            image: pyramid.depth_staging.image().clone(),
            mips: pyramid.depth_staging.image().mip_levels(),
        });

        recorder.bind_descriptor_sets(&DescriptorBindReq {
            frame_index: Some(frame_index),
            bind_point: vk::PipelineBindPoint::COMPUTE,
            layout: self.depth_resolve.layout(),
            descriptors: &[self.depth_resolve_descriptor.clone()],
        });

        let vk::Extent3D { width, height, .. } = pyramid.depth_staging.image().extent(0);
        let info = DepthResolveInfo {
            image_size: UVec2::new(width, height),
            // `vk::SampleCountFlags` as raw maps to the amount of samples.
            sample: depth_image.image().sample_count().as_raw(),
        };

        recorder.push_constants(
            self.depth_resolve.layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            &info,
        );
       
        recorder.dispatch(self.depth_resolve.clone(), [width.div_ceil(16), height.div_ceil(16), 1]);

        recorder.image_barrier(&ImageBarrierReq {
            flags: vk::DependencyFlags::BY_REGION,
            src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            src_mask: vk::AccessFlags2::SHADER_WRITE,
            dst_mask: vk::AccessFlags2::SHADER_READ,
            new_layout: vk::ImageLayout::GENERAL,
            image: pyramid.depth_staging.image().clone(),
            mips: pyramid.depth_staging.image().mip_levels(),
        });

        //
        // Reduce from each level to the next in depth pyramid.
        //

        recorder.bind_descriptor_sets(&DescriptorBindReq {
            frame_index: Some(frame_index),
            bind_point: vk::PipelineBindPoint::COMPUTE,
            layout: self.depth_reduce.layout(),
            descriptors: &[
                self.sampled_pyramid.clone(),
                self.storage_pyramid.clone(),
            ],
        });

        for target in pyramid.pyramid.mip_levels() {
            let vk::Extent3D { width, height, .. } = pyramid.pyramid.extent(target);
            let info = DepthReduceInfo {
                image_size: UVec2::new(width, height),
                target,
            };

            let layout = self.depth_reduce.layout();

            recorder.push_constants(layout, vk::ShaderStageFlags::COMPUTE, 0, &info);
            recorder.dispatch(self.depth_reduce.clone(), [width / 32, height / 32, 1]);

            recorder.image_barrier(&ImageBarrierReq {
                flags: vk::DependencyFlags::BY_REGION,
                src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                src_mask: vk::AccessFlags2::SHADER_WRITE,
                dst_mask: vk::AccessFlags2::SHADER_READ,
                new_layout: vk::ImageLayout::GENERAL,
                image: pyramid.pyramid.clone(),
                mips: target..target + 1,
            });
        }
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

    pub fn pyramid_debug(
        &self,
        frame_index: FrameIndex,
        swapchain_image: Res<Image>,
        recorder: &CommandRecorder,
        level: u32,
    ) {
        let pyramid = &self.depth_pyramids[frame_index];

        recorder.blit_image(&ImageBlitReq {
            src: pyramid.pyramid.clone(),
            dst: swapchain_image,
            filter: vk::Filter::NEAREST,
            src_mip: level,
            dst_mip: 0,
        });
    }

    pub fn primitives_drawn(&self, renderer: &Renderer, frame_index: FrameIndex) -> Result<u32> {
        let src = self.draw_count_buffers[frame_index].clone();
        let dst = self.draw_count_host_buffers[frame_index].clone();

        renderer.transfer_with(|recorder| {
            recorder.copy_buffers(src.clone(), dst.clone());
        })?;

        let mapped = dst.get_mapped()?;
        let count: &DrawCount = bytemuck::from_bytes(mapped.as_slice());

        Ok(count.command_count)
    }
}

fn create_sampled_pyramid_descriptor(
    renderer: &Renderer,
    depth_pyramids: &PerFrame<DepthPyramid>,
) -> Result<Res<DescriptorSet>> {
    let pool = &renderer.pool;

    let sampler = pool.alloc(
        TextureSampler::new(renderer, vk::SamplerReductionMode::MIN)?
    );

    let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
        LayoutBinding {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            stage: vk::ShaderStageFlags::COMPUTE,
            array_count: Some(depth_pyramids[FrameIndex::Uno].mips.len() as u32),
        },
    ])?);

    Ok(pool.alloc(
        DescriptorSet::new_per_frame(&renderer, layout, &[
            DescriptorBinding::VariableImageArray(sampler, vk::ImageLayout::GENERAL, [
                &depth_pyramids[FrameIndex::Uno].mips,
                &depth_pyramids[FrameIndex::Dos].mips,
            ]),
        ])?
    ))
}

fn create_storage_pyramid_descriptor(
    renderer: &Renderer,
    sampler: Res<TextureSampler>,
    depth_pyramids: &PerFrame<DepthPyramid>,
) -> Result<Res<DescriptorSet>> {
    let pool = &renderer.pool;

    let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
        LayoutBinding {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            stage: vk::ShaderStageFlags::COMPUTE,
            array_count: Some(
                depth_pyramids[FrameIndex::Uno].mips.len() as u32 - 1
            ),
        },
    ])?);

    let image_layout = vk::ImageLayout::GENERAL;

    let set = pool.alloc(DescriptorSet::new_per_frame(&renderer, layout, &[
        DescriptorBinding::VariableImageArray(sampler.clone(), image_layout, [
            &depth_pyramids[FrameIndex::Uno].mips[1..],
            &depth_pyramids[FrameIndex::Dos].mips[1..],
        ]),
    ])?);

    Ok(set)
}

fn create_depth_resolve_descriptor(
    renderer: &Renderer,
    render_targets: &RenderTargets,
    depth_pyramids: &PerFrame<DepthPyramid>,
    sampler: Res<TextureSampler>,
) -> Result<Res<DescriptorSet>> {
    let pool = &renderer.pool;

    let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[
        LayoutBinding {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            stage: vk::ShaderStageFlags::COMPUTE,
            array_count: None,
        },
        LayoutBinding {
            ty: vk::DescriptorType::STORAGE_IMAGE,
            stage: vk::ShaderStageFlags::COMPUTE,
            array_count: None,
        },
    ])?);

    let image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

    let set = pool.alloc(DescriptorSet::new_per_frame(&renderer, layout.clone(), &[
        DescriptorBinding::Image(sampler.clone(), image_layout, [
            render_targets.depth_images[FrameIndex::Uno].clone(),
            render_targets.depth_images[FrameIndex::Dos].clone(),
        ]),
        DescriptorBinding::Image(sampler.clone(), vk::ImageLayout::GENERAL, [
            depth_pyramids[FrameIndex::Uno].depth_staging.clone(),
            depth_pyramids[FrameIndex::Dos].depth_staging.clone(),
        ]),
    ])?);

    Ok(set)
}

const MAX_LOD_COUNT: usize = 8;

// Get previous power of 2.
fn prev_pow2(mut val: u32) -> u32 {
    val = val | (val >> 1);
    val = val | (val >> 2);
    val = val | (val >> 4);
    val = val | (val >> 8);
    val = val | (val >> 16);
    val - (val >> 1)
}
