use glam::{Vec4, Mat4, UVec2};
use anyhow::Result;
use ash::vk;

use std::mem;

use crate::light::Lights;
use crate::core::*;
use crate::resource::*;
use crate::command::*;
use crate::camera::{self, Proj, View, CameraUniforms};

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
    stagings: PerFrame<Res<ImageView>>,

    /// The depth pyramid image. It has the dimensions of `depth_image` rounded down to the
    /// previous power of 2.
    pyramids: PerFrame<Res<Image>>,

    /// Min sampled image array of each level in `mips`.
    sampled: PerFrame<Res<DescSet>>,

    /// Storage image array of each level in `mips`, besides level 0.
    storage: PerFrame<Res<DescSet>>,

    /// Descriptor of `depth_staging` and the depth image.
    resolve_descs: PerFrame<Res<DescSet>>,

    reduce: Res<ComputePipeline>,
    resolve: Res<ComputePipeline>,

    // TODO: Remove.
    width: u32,
    height: u32,
}

impl DepthPyramid {
    fn new(renderer: &Renderer, depth_images: &PerFrame<Res<ImageView>>) -> Result<Self> {
        let pool = &renderer.pool;

        let depth_extent = depth_images.any().image().extent(0);

        let width = prev_pow2(depth_extent.width);
        let height = prev_pow2(depth_extent.height);

        let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;

        let mip_levels = (height.max(height) as f32).log2().floor() as u32;

        let pyramids = PerFrame::try_from_fn(|_| {
            renderer.pool.create_image(memory_flags, &ImageInfo {
                extent: vk::Extent3D { width, height, depth: 1 },
                aspect_flags: vk::ImageAspectFlags::COLOR,
                format: vk::Format::R32_SFLOAT,
                kind: ImageKind::Texture,
                mip_levels,
                usage,
            })
        })?;

        let stagings = PerFrame::try_from_fn(|_| {
            let image = renderer.pool.create_image(memory_flags, &ImageInfo {
                format: vk::Format::R32_SFLOAT,
                aspect_flags: vk::ImageAspectFlags::COLOR,
                kind: ImageKind::Texture,
                extent: depth_extent,
                mip_levels: 1,
                usage,
            })?;

            renderer.pool.create_image_view(&ImageViewInfo {
                view_type: vk::ImageViewType::TYPE_2D,
                image: image.clone(),
                mips: image.mip_levels(),
            })
        })?;

        renderer.transfer_with(|recorder| {
            for (pyramid, staging) in pyramids.iter().zip(stagings.iter()) {
                recorder.image_barrier(&ImageBarrierInfo {
                    flags: vk::DependencyFlags::BY_REGION,
                    src_stage: vk::PipelineStageFlags2::empty(),
                    dst_stage: vk::PipelineStageFlags2::empty(),
                    src_mask: vk::AccessFlags2::empty(),
                    dst_mask: vk::AccessFlags2::empty(),
                    new_layout: vk::ImageLayout::GENERAL,
                    mips: pyramid.mip_levels(),
                    image: pyramid.clone(),
                });

                recorder.image_barrier(&ImageBarrierInfo {
                    flags: vk::DependencyFlags::BY_REGION,
                    src_stage: vk::PipelineStageFlags2::empty(),
                    dst_stage: vk::PipelineStageFlags2::empty(),
                    src_mask: vk::AccessFlags2::empty(),
                    dst_mask: vk::AccessFlags2::empty(),
                    new_layout: vk::ImageLayout::GENERAL,
                    mips: staging.image().mip_levels(),
                    image: staging.image().clone(),
                });
            }
        })?;

        let mips = PerFrame::try_from_fn(|frame_index| {
            let mut mips = vec![stagings[frame_index].clone()];
            let pyramid = pyramids[frame_index].clone();

            for level in pyramid.mip_levels() {
                mips.push(
                    renderer.pool.create_image_view(&ImageViewInfo {
                        view_type: vk::ImageViewType::TYPE_2D,
                        mips: level..level + 1,
                        image: pyramid.clone(),
                    })?
                );
            }

            Ok(mips)
        })?;

        let min_sampler = pool.create_sampler(vk::SamplerReductionMode::MIN)?;

        let layout = pool.create_desc_layout(&[
            DescLayoutSlot {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: Some(mip_levels + 1),
            },
        ])?;

        let sampled = PerFrame::try_from_fn(|frame_index| {
            pool.create_desc_set(layout.clone(), &[
                DescBinding::ImageArray(
                    min_sampler.clone(),
                    vk::ImageLayout::GENERAL,
                    &mips[frame_index],
                ),
            ])
        })?;

        let sampler = pool.create_sampler(vk::SamplerReductionMode::WEIGHTED_AVERAGE)?;

        let layout = pool.create_desc_layout(&[
            DescLayoutSlot {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: Some(mip_levels),
            },
        ])?;

        let storage = PerFrame::try_from_fn(|frame_index| {
            pool.create_desc_set(layout.clone(), &[
                DescBinding::ImageArray(
                    sampler.clone(),
                    vk::ImageLayout::GENERAL,
                    &mips[frame_index][1..],
                ),
            ])
        })?;

        let layout = pool.create_desc_layout(&[
            DescLayoutSlot {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
            DescLayoutSlot {
                ty: vk::DescriptorType::STORAGE_IMAGE,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
        ])?;

        let image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

        let resolve_descs = PerFrame::try_from_fn(|frame_index| {
            pool.create_desc_set(layout.clone(), &[
                DescBinding::Image(sampler.clone(), image_layout,
                    depth_images[frame_index].clone(),
                ),
                DescBinding::Image(sampler.clone(), vk::ImageLayout::GENERAL,
                    stagings[frame_index].clone(),
                ),
            ])
        })?;

        let resolve = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/depth_resolve.comp.spv");
            let shader = pool.create_shader_module("main", code)?;

            let push_consts = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(mem::size_of::<DepthResolveInfo>() as u32)
                .offset(0)
                .build()];

            let layout = pool.create_pipeline_layout(&push_consts, &[
                resolve_descs.any().layout(),
            ])?;

            pool.create_compute_pipeline(layout, shader)?
        };

        let reduce = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/depth_reduce.comp.spv");
            let shader = pool.create_shader_module("main", code)?;

            let push_consts = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(mem::size_of::<DepthReduceInfo>() as u32)
                .offset(0)
                .build()];

            let layout = pool.create_pipeline_layout(&push_consts, &[
                sampled.any().layout(),
                storage.any().layout(),
            ])?;

            pool.create_compute_pipeline(layout, shader)?
        };


        Ok(Self {
            resolve,
            reduce,
            pyramids,
            stagings,
            resolve_descs,
            storage,
            sampled,
            width,
            height,
        })
    }

    fn update(
        &self,
        frame_index: FrameIndex,
        depth_images: &PerFrame<Res<ImageView>>,
        recorder: &CommandRecorder,
    ) {
        let depth_image = depth_images[frame_index].clone();

        recorder.image_barrier(&ImageBarrierInfo {
            flags: vk::DependencyFlags::BY_REGION,
            src_stage: vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            src_mask: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            dst_mask: vk::AccessFlags2::SHADER_READ,
            new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            image: depth_image.image().clone(),
            mips: depth_image.image().mip_levels(),
        });

        recorder.image_barrier(&ImageBarrierInfo {
            flags: vk::DependencyFlags::BY_REGION,
            src_stage: vk::PipelineStageFlags2::empty(),
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            src_mask: vk::AccessFlags2::empty(),
            dst_mask: vk::AccessFlags2::SHADER_WRITE,
            new_layout: vk::ImageLayout::GENERAL,
            image: self.stagings[frame_index].image().clone(),
            mips: self.stagings[frame_index].image().mip_levels(),
        });

        recorder.bind_descs(&DescBindInfo {
            bind_point: vk::PipelineBindPoint::COMPUTE,
            layout: self.resolve.layout(),
            descs: &[
                self.resolve_descs[frame_index].clone()
            ],
        });

        let vk::Extent3D { width, height, .. } =
            self.stagings[frame_index].image().extent(0);

        let info = DepthResolveInfo {
            image_size: UVec2::new(width, height),
            // `vk::SampleCountFlags` as raw maps to the amount of samples.
            sample: depth_image.image().sample_count().as_raw(),
        };

        recorder.push_consts(
            self.resolve.layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&info),
        );
       
        recorder.dispatch(self.resolve.clone(), [width.div_ceil(16), height.div_ceil(16), 1]);

        recorder.image_barrier(&ImageBarrierInfo {
            flags: vk::DependencyFlags::BY_REGION,
            src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            src_mask: vk::AccessFlags2::SHADER_WRITE,
            dst_mask: vk::AccessFlags2::SHADER_READ,
            new_layout: vk::ImageLayout::GENERAL,
            image: self.stagings[frame_index].image().clone(),
            mips: self.stagings[frame_index].image().mip_levels(),
        });

        //
        // Reduce from each level to the next in depth pyramid.
        //

        recorder.bind_descs(&DescBindInfo {
            bind_point: vk::PipelineBindPoint::COMPUTE,
            layout: self.reduce.layout(),
            descs: &[
                self.sampled[frame_index].clone(),
                self.storage[frame_index].clone(),
            ],
        });

        let pyramid = &self.pyramids[frame_index];

        for target in pyramid.mip_levels() {
            let vk::Extent3D { width, height, .. } = pyramid.extent(target);
            let info = DepthReduceInfo {
                image_size: UVec2::new(width, height),
                target,
            };

            let layout = self.reduce.layout();

            recorder.push_consts(
                layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(&info),
            );

            recorder.dispatch(self.reduce.clone(), [width / 32, height / 32, 1]);

            recorder.image_barrier(&ImageBarrierInfo {
                flags: vk::DependencyFlags::BY_REGION,
                src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
                src_mask: vk::AccessFlags2::SHADER_WRITE,
                dst_mask: vk::AccessFlags2::SHADER_READ,
                new_layout: vk::ImageLayout::GENERAL,
                image: pyramid.clone(),
                mips: target..target + 1,
            });
        }
    }
}

pub struct ForwardPass {
    descs: PerFrame<Res<DescSet>>,

    pub depth_images: PerFrame<Res<ImageView>>,
    pub color_images: PerFrame<Res<ImageView>>,

    render: Res<GraphicsPipeline>, 
    cull: Res<ComputePipeline>,

    depth_pyramid: DepthPyramid,

    /// Contains [`DrawCommand`] used for draw indirect count commands.
    draw_buffers: PerFrame<Res<Buffer>>,

    /// Contains [`DrawCount`] used for draw indirect count commands.
    draw_count_buffers: PerFrame<Res<Buffer>>,

    /// Small host buffer to copy `draw_count_buffers` into to access amount of primitives drawn.
    draw_count_host_buffers: PerFrame<Res<Buffer>>,

    render_target_info: RenderTargetInfo,
    primitive_count: u32,
}

impl ForwardPass {
    pub fn new(
        renderer: &Renderer,
        camera_uniforms: &CameraUniforms,
        scene: &Scene,
        lights: &Lights,
    ) -> Result<Self> {
        let pool = &renderer.pool;

        let extent = renderer.swapchain.extent_3d();
        let samples = renderer.device.sample_count();

        let depth_images = create_depth_images(renderer, extent, samples)?;
        let color_images = create_forward_color_images(renderer, extent, samples)?;

        let depth_pyramid = DepthPyramid::new(renderer, &depth_images)?;

        let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;
        let draw_count_buffers = PerFrame::try_from_fn(|_| {
            pool.create_buffer(memory_flags, &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                size: mem::size_of::<DrawCount>() as vk::DeviceSize,
            })
        })?;

        let draw_count_host_buffers = PerFrame::try_from_fn(|_| {
            let memory_flags = vk::MemoryPropertyFlags::HOST_VISIBLE
                | vk::MemoryPropertyFlags::HOST_COHERENT;

            pool.create_buffer(memory_flags, &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                size: mem::size_of::<DrawCount>() as vk::DeviceSize,
            })
        })?;

        renderer.transfer_with(|recorder| {
            let draw_count = DrawCount {
                command_count: 0,
                primitive_count: scene.primitive_count(),
            };

            for buffer in &draw_count_buffers {
                recorder.update_buffer(buffer.clone(), &draw_count);
            }
        })?;

        let draw_buffers = PerFrame::try_from_fn(|_| {
            pool.create_buffer(memory_flags, &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDIRECT_BUFFER,
                size: (scene.primitives.len() * mem::size_of::<DrawCommand>()) as vk::DeviceSize,
            })
        })?;

        let sampler = pool.create_sampler(vk::SamplerReductionMode::WEIGHTED_AVERAGE)?;

        let layout = pool.create_desc_layout(&[
            DescLayoutSlot {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
            DescLayoutSlot {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
            DescLayoutSlot {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
            DescLayoutSlot {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::COMPUTE,
                array_count: None,
            },
            DescLayoutSlot {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                stage: vk::ShaderStageFlags::VERTEX,
                array_count: None,
            },
            DescLayoutSlot {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                stage: vk::ShaderStageFlags::FRAGMENT,
                array_count: Some(scene.textures.len() as u32),
            },
        ])?;

        let descs = PerFrame::try_from_fn(|frame_index| {
            pool.create_desc_set(layout.clone(), &[
                DescBinding::Buffer(scene.instance_buffer.clone()),
                DescBinding::Buffer(draw_buffers[frame_index].clone()),
                DescBinding::Buffer(scene.primitive_buffer.clone()),
                DescBinding::Buffer(draw_count_buffers[frame_index].clone()),
                DescBinding::Buffer(scene.vertex_buffer.clone()),
                DescBinding::ImageArray(
                    sampler.clone(),
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    &scene.textures,
                ),
            ])
        })?;

        let render_target_info = RenderTargetInfo {
            color_format: color_images.any().image().format(),
            depth_format: depth_images.any().image().format(),
            sample_count: samples,
        };

        let render = {
            let vertex_code = include_bytes_aligned_as!(u32, "../assets/shaders/pbr.vert.spv");
            let fragment_code = include_bytes_aligned_as!(u32, "../assets/shaders/pbr.frag.spv");

            let vertex_shader = pool.create_shader_module("main", vertex_code)?;
            let fragment_shader = pool.create_shader_module("main", fragment_code)?;

            let depth_stencil_info = &vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_write_enable(true)
                .depth_test_enable(true);

            let layout = pool.create_pipeline_layout(&[], &[
                camera_uniforms.descs.any().layout(),
                lights.descs.any().layout(),
                descs.any().layout(),
            ])?;

            pool.create_graphics_pipeline(&renderer, GraphicsPipelineInfo {
                render_target_info,
                cull_mode: vk::CullModeFlags::BACK,
                fragment_shader,
                vertex_shader,
                vertex_attributes: &[],
                vertex_bindings: &[],
                depth_stencil_info,
                layout,
            })?
        };

        let cull = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/draw_cull.comp.spv");
            let shader = pool.create_shader_module("main", code)?;

            let push_consts = [vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .size(mem::size_of::<CullInfo>() as u32)
                .offset(0)
                .build()];

            let layout = pool.create_pipeline_layout(&push_consts, &[
                camera_uniforms.descs.any().layout(),
                descs.any().layout(),
                depth_pyramid.sampled.any().layout(),
            ])?;

            pool.create_compute_pipeline(layout, shader)?
        };

        Ok(Self {
            primitive_count: scene.primitive_count(),
            depth_images,
            color_images,
            depth_pyramid,
            draw_buffers,
            draw_count_buffers,
            draw_count_host_buffers,
            descs,
            render_target_info,
            cull,
            render,
        })
    }

    pub fn render_target_info(&self) -> RenderTargetInfo {
        self.render_target_info
    }

    pub fn prepare_draw_buffers(
        &self,
        frame_index: FrameIndex,
        proj: &Proj,
        view: &View,
        camera_uniforms: &CameraUniforms,
        recorder: &CommandRecorder,
    ) {
        let draw_count = DrawCount {
            command_count: 0,
            primitive_count: self.primitive_count,
        };

        recorder.update_buffer(self.draw_count_buffers[frame_index].clone(), &draw_count);

        recorder.buffer_barrier(&BufferBarrierInfo {
            buffer: self.draw_count_buffers[frame_index].clone(),
            src_mask: vk::AccessFlags2::TRANSFER_WRITE,
            dst_mask: vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
            src_stage: vk::PipelineStageFlags2::TRANSFER,
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        });

        //
        // Frustrum cull and generate draw buffers.
        //

        recorder.bind_descs(&DescBindInfo {
            bind_point: vk::PipelineBindPoint::COMPUTE,
            layout: self.cull.layout(),
            descs: &[
                camera_uniforms.descs[frame_index].clone(),
                self.descs[frame_index].clone(),
                self.depth_pyramid.sampled[frame_index].clone(),
            ],
        });

        let cull_info = CullInfo {
            z_near: proj.z_near,
            z_far: proj.z_far,

            frustrum_planes: camera::frustrum_planes(proj, view),

            pyramid_width: self.depth_pyramid.width as f32,
            pyramid_height: self.depth_pyramid.height as f32,

            lod_base: 10.0,
            lod_step: 2.0,

            _pad1: 0.0,
            _pad2: 0.0,
        };

        recorder.push_consts(
            self.cull.layout(),
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(&cull_info),
        );

        recorder.dispatch(self.cull.clone(), [
            self.primitive_count.div_ceil(64), 1, 1,
        ]);

        recorder.buffer_barrier(&BufferBarrierInfo {
            buffer: self.draw_buffers[frame_index].clone(),
            src_mask: vk::AccessFlags2::SHADER_WRITE,
            dst_mask: vk::AccessFlags2::INDIRECT_COMMAND_READ,
            src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            dst_stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
        });

        self.depth_pyramid.update(frame_index, &self.depth_images, recorder);
    }

    pub fn draw(
        &self,
        frame_index: FrameIndex,
        scene: &Scene,
        camera_uniforms: &CameraUniforms,
        lights: &Lights,
        recorder: &DrawRecorder,
    ) {
        recorder.bind_index_buffer(scene.index_buffer.clone(), vk::IndexType::UINT32);
        recorder.bind_graphics_pipeline(self.render.clone());

        recorder.bind_descs(&DescBindInfo {
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            layout: self.render.layout(),
            descs: &[
                camera_uniforms.descs[frame_index].clone(),
                lights.descs[frame_index].clone(),
                self.descs[frame_index].clone(),
            ],
        });
        
        recorder.draw_indexed_indirect_count(&IndexedIndirectDrawInfo {
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
        let pyramid = &self.depth_pyramid.pyramids[frame_index];

        recorder.blit_image(&ImageBlitInfo {
            src: pyramid.clone(),
            dst: swapchain_image,
            filter: vk::Filter::NEAREST,
            src_mip: level,
            dst_mip: 0,
        });
    }

    pub fn primitives_drawn(&self, renderer: &Renderer, frame_index: FrameIndex) -> Result<u32> {
        let src = self.draw_count_buffers[frame_index].clone();
        let dst = self.draw_count_host_buffers[frame_index].clone();

        renderer.transfer_with(|recorder| recorder.copy_buffers(src.clone(), dst.clone()))?;

        let mapped = dst.get_mapped()?;
        let count: &DrawCount = bytemuck::from_bytes(mapped.as_slice());

        Ok(count.command_count)
    }

    pub fn handle_resize(&mut self, renderer: &Renderer) -> Result<()> {
        let extent = renderer.swapchain.extent_3d();
        let samples = renderer.device.sample_count();

        self.depth_images = create_depth_images(renderer, extent, samples)?;
        self.color_images = create_forward_color_images(renderer, extent, samples)?;

        self.depth_pyramid = DepthPyramid::new(renderer, &self.depth_images)?;

        Ok(())
    }
}

fn create_depth_images(
    renderer: &Renderer,
    extent: vk::Extent3D,
    samples: vk::SampleCountFlags,
) -> Result<PerFrame<Res<ImageView>>> {
    let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

    PerFrame::try_from_fn(|_| {
        let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::SAMPLED;

        let image = renderer.pool.create_image(memory_flags, &ImageInfo {
            aspect_flags: vk::ImageAspectFlags::DEPTH,
            format: DEPTH_IMAGE_FORMAT,
            kind: ImageKind::RenderTarget {
                queue: renderer.graphics_queue(),
                samples,
            },
            mip_levels: 1,
            extent,
            usage,
        })?;

        renderer.pool.create_image_view(&ImageViewInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            mips: image.mip_levels(),
            image,
        })
    })
}

fn create_forward_color_images(
    renderer: &Renderer,
    extent: vk::Extent3D,
    samples: vk::SampleCountFlags,
) -> Result<PerFrame<Res<ImageView>>> {
    let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

    PerFrame::try_from_fn(|_| {
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC;

        let image = renderer.pool.create_image(memory_flags, &ImageInfo {
            aspect_flags: vk::ImageAspectFlags::COLOR,
            format: renderer.swapchain.format(),
            kind: ImageKind::RenderTarget {
                queue: renderer.graphics_queue(),
                samples,
            },
            mip_levels: 1,
            extent,
            usage,
        })?;

        renderer.pool.create_image_view(&ImageViewInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            mips: 0..1,
            image,
        })
    })
}

pub struct Scene {
    primitives: Vec<Primitive>,
    textures: Vec<Res<ImageView>>,
    index_buffer: Res<Buffer>,
    vertex_buffer: Res<Buffer>, 
    instance_buffer: Res<Buffer>,
    primitive_buffer: Res<Buffer>,
}

impl Scene {
    pub fn from_scene_asset(renderer: &Renderer, scene: &asset::Scene) -> Result<Self> {
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

        let staging_pool = ResourcePool::new(renderer.device.clone());

        let memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let primitive_data = bytemuck::cast_slice(primitives.as_slice());
        let primitive_staging = staging_pool.create_buffer(memory_flags, &BufferInfo {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: primitive_data.len() as vk::DeviceSize,
        })?;

        let instance_data = bytemuck::cast_slice(instance_data.as_slice());
        let instance_staging = staging_pool.create_buffer(memory_flags, &BufferInfo {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: instance_data.len() as vk::DeviceSize,
        })?;

        let vertex_data = bytemuck::cast_slice(scene.vertices.as_slice());
        let vertex_staging = staging_pool.create_buffer(memory_flags, &BufferInfo {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: vertex_data.len() as vk::DeviceSize,
        })?;

        let index_data = bytemuck::cast_slice(scene.indices.as_slice());
        let index_staging = staging_pool.create_buffer(memory_flags, &BufferInfo {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: index_data.len() as vk::DeviceSize,
        })?;

        primitive_staging.get_mapped()?.fill(primitive_data);
        instance_staging.get_mapped()?.fill(instance_data);
        vertex_staging.get_mapped()?.fill(vertex_data);
        index_staging.get_mapped()?.fill(index_data);


        let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        let instance_buffer = pool.create_buffer(memory_flags, &BufferInfo {
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            size: instance_data.len() as vk::DeviceSize,
        })?;

        let primitive_buffer = pool.create_buffer(memory_flags, &BufferInfo {
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            size: primitive_data.len() as vk::DeviceSize,
        })?;

        let vertex_buffer = pool.create_buffer(memory_flags, &BufferInfo {
            usage: vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            size: vertex_data.len() as vk::DeviceSize,
        })?;

        let index_buffer = pool.create_buffer(memory_flags, &BufferInfo {
            usage: vk::BufferUsageFlags::INDEX_BUFFER
                | vk::BufferUsageFlags::TRANSFER_DST,
            size: index_data.len() as vk::DeviceSize,
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
        })?;

        let memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let staging: Result<Vec<Vec<_>>> = scene.textures
            .iter()
            .map(|texture| {
                texture.mips
                    .iter()
                    .map(|data| {
                        let buffer = staging_pool.create_buffer(memory_flags, &BufferInfo {
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
                pool.create_image(memory_flags, &ImageInfo {
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
                recorder.image_barrier(&ImageBarrierInfo {
                    flags: vk::DependencyFlags::BY_REGION,
                    mips: 0..image.mip_level_count(),
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_stage: vk::PipelineStageFlags2::empty(),
                    dst_stage: vk::PipelineStageFlags2::TRANSFER,
                    src_mask: vk::AccessFlags2::empty(),
                    dst_mask: vk::AccessFlags2::TRANSFER_WRITE,
                    image: image.clone(),
                });
            }

            for (levels, dst) in staging.iter().zip(images.iter()) {
                for (level, src) in levels.iter().enumerate() {
                    recorder.copy_buffer_to_image(src.clone(), dst.clone(), level as u32);
                }
            }

            for image in images.iter() {
                recorder.image_barrier(&ImageBarrierInfo {
                    flags: vk::DependencyFlags::BY_REGION,
                    mips: 0..image.mip_level_count(),
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_stage: vk::PipelineStageFlags2::TRANSFER,
                    dst_stage: vk::PipelineStageFlags2::empty(),
                    src_mask: vk::AccessFlags2::TRANSFER_WRITE,
                    dst_mask: vk::AccessFlags2::empty(),
                    image: image.clone(),
                });
            }
        })?;

        let textures: Result<Vec<_>> = images
            .into_iter()
            .map(|image| {
                pool.create_image_view(&ImageViewInfo {
                    view_type: vk::ImageViewType::TYPE_2D,
                    mips: image.mip_levels(),
                    image: image.clone(),
                })
            })
            .collect();
    
        let textures = textures?;

        Ok(Self {
            primitives,
            textures,
            index_buffer,
            vertex_buffer,
            instance_buffer,
            primitive_buffer,
        })
    }

    pub fn primitive_count(&self) -> u32 {
        self.primitives.len() as u32
    }
}

const MAX_LOD_COUNT: usize = 8;
const DEPTH_IMAGE_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

// Get previous power of 2.
fn prev_pow2(mut val: u32) -> u32 {
    val = val | (val >> 1);
    val = val | (val >> 2);
    val = val | (val >> 4);
    val = val | (val >> 8);
    val = val | (val >> 16);
    val - (val >> 1)
}
