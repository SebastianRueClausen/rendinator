use anyhow::Result;
use ash::vk;

use std::array;
use std::mem;

use crate::camera::*;
use crate::command::*;
use crate::core::*;
use crate::frame::{FrameIndex, PerFrame};
use crate::light::{self, DirLight, Lights, PointLight};
use crate::resource::*;

use rendi_math::prelude::*;
use rendi_res::Res;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct InstanceData {
    #[allow(dead_code)]
    transform: Mat4,
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
    /// Center of the bounding sphere of the primitive.
    center: Vec4,

    /// The radius of a bounding sphere around the primitive.
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
    /// All the bounding planes of the frustrum in world space.
    frustrum_planes: [Vec4; 6],

    lod_base: f32,
    lod_step: f32,

    /// Width of the bottom pyramid level.
    pyramid_width: f32,

    /// Height of the bottom pyramid level.
    pyramid_height: f32,

    /// The number of mip levels of the depth pyramid.
    pyramid_mip_count: u32,

    /// The [`RenderingPhase`].
    phase: u32,

    padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct DepthReduceInfo {
    /// The size of the target pyramid level level.
    image_size: UVec2,

    /// The index of the target pyramid level.
    target: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::NoUninit)]
struct DepthResolveInfo {
    /// The size of the both of depth image and depth staging image.
    image_size: UVec2,

    /// The number of samples in the depth image.
    samples: u32,
}

struct DepthPyramid {
    /// Depth staging image.
    ///
    /// This is used to resolve the multisampled depth image into.
    stagings: PerFrame<Res<ImageView>>,

    /// The depth pyramid image. It has the dimensions of `depth_image` rounded down to the
    /// previous power of 2.
    pyramids: PerFrame<Res<Image>>,

    /// Descriptor of sampled image views for each mip level of the pyramid.
    ///
    /// The first image view is from `stagings` since it's used to reduce into the first actual
    /// level of the pyramid.
    sampled: PerFrame<Res<DescSet>>,

    /// Storage image array of each level in `mips`.
    ///
    /// Unlike `sampled`, the first image is not from `stagings`.
    storage: PerFrame<Res<DescSet>>,

    /// Descriptor of `depth_staging` and the depth image.
    resolve_descs: PerFrame<Res<DescSet>>,

    /// Compute shader to reduce one level of the pyramid into the next.
    reduce: Res<ComputePipeline>,

    /// Compute shader to resolve a multisampled depth image into `stagings`.
    resolve: Res<ComputePipeline>,

    /// Width of the top level of the pyramid.
    width: u32,

    /// Height of the top level of the pyramid.
    height: u32,

    /// The number of mip levels in the pyramid, not including the staging image.
    mip_levels: u32,
}

impl DepthPyramid {
    fn new(renderer: &Renderer, depth_images: &PerFrame<Res<ImageView>>) -> Result<Self> {
        let pool = &renderer.pool;

        let depth_extent = depth_images.any().image().extent(0);

        let width = prev_pow2(depth_extent.width);
        let height = prev_pow2(depth_extent.height);

        let memory_location = MemoryLocation::Gpu;

        let usage = vk::ImageUsageFlags::STORAGE
            | vk::ImageUsageFlags::SAMPLED
            | vk::ImageUsageFlags::TRANSFER_SRC;

        let mip_levels = (height.max(height) as f32).log2().floor() as u32;

        let pyramids = PerFrame::try_from_fn(|_| {
            renderer.pool.create_image(
                memory_location,
                &ImageInfo {
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    },
                    aspect_flags: vk::ImageAspectFlags::COLOR,
                    format: vk::Format::R32_SFLOAT,
                    kind: ImageKind::Texture,
                    mip_levels,
                    usage,
                },
            )
        })?;

        let stagings = PerFrame::try_from_fn(|_| {
            let image = renderer.pool.create_image(
                memory_location,
                &ImageInfo {
                    format: vk::Format::R32_SFLOAT,
                    aspect_flags: vk::ImageAspectFlags::COLOR,
                    kind: ImageKind::Texture,
                    extent: depth_extent,
                    mip_levels: 1,
                    usage,
                },
            )?;

            renderer.pool.create_image_view(&ImageViewInfo {
                view_type: vk::ImageViewType::TYPE_2D,
                mips: image.mip_levels(),
                image: image.clone(),
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
                mips.push(renderer.pool.create_image_view(&ImageViewInfo {
                    view_type: vk::ImageViewType::TYPE_2D,
                    mips: level..level + 1,
                    image: pyramid.clone(),
                })?);
            }

            Ok(mips)
        })?;

        let sampler = pool.create_sampler()?;

        let layout = pool.create_desc_layout(&[DescLayoutSlot {
            binding: 0,
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            count: rendi_render::DescCount::UnboundArray,
        }])?;

        let sampled = PerFrame::try_from_fn(|frame_index| {
            pool.create_desc_set(
                layout.clone(),
                &[DescBinding::ImageArray(
                    sampler.clone(),
                    vk::ImageLayout::GENERAL,
                    &mips[frame_index],
                )],
            )
        })?;

        let layout = pool.create_desc_layout(&[DescLayoutSlot {
            binding: 0,
            ty: vk::DescriptorType::STORAGE_IMAGE,
            count: rendi_render::DescCount::UnboundArray,
        }])?;

        let storage = PerFrame::try_from_fn(|frame_index| {
            pool.create_desc_set(
                layout.clone(),
                &[DescBinding::ImageArray(
                    sampler.clone(),
                    vk::ImageLayout::GENERAL,
                    &mips[frame_index][1..],
                )],
            )
        })?;

        let layout = pool.create_desc_layout(&[
            DescLayoutSlot {
                binding: 0,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: rendi_render::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 1,
                ty: vk::DescriptorType::STORAGE_IMAGE,
                count: rendi_render::DescCount::Single,
            },
        ])?;

        let image_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;

        let resolve_descs = PerFrame::try_from_fn(|frame_index| {
            pool.create_desc_set(
                layout.clone(),
                &[
                    DescBinding::Image(
                        sampler.clone(),
                        image_layout,
                        depth_images[frame_index].clone(),
                    ),
                    DescBinding::Image(
                        sampler.clone(),
                        vk::ImageLayout::GENERAL,
                        stagings[frame_index].clone(),
                    ),
                ],
            )
        })?;

        let resolve = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/depth_resolve.comp.spv");
            let shader = pool.create_shader_module("main", code)?;
            let prog = pool.create_compute_prog(shader)?;

            pool.create_compute_pipeline(
                prog,
                &[PushConstRange {
                    size: mem::size_of::<DepthResolveInfo>() as vk::DeviceSize,
                    stage: vk::ShaderStageFlags::COMPUTE,
                }],
            )?
        };

        let reduce = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/depth_reduce.comp.spv");
            let shader = pool.create_shader_module("main", code)?;
            let prog = pool.create_compute_prog(shader)?;

            pool.create_compute_pipeline(
                prog,
                &[PushConstRange {
                    size: mem::size_of::<DepthReduceInfo>() as vk::DeviceSize,
                    stage: vk::ShaderStageFlags::COMPUTE,
                }],
            )?
        };

        Ok(Self {
            mip_levels,
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
}

/// Descriptors for each frame in flight for buffers related to the camera.
pub struct CameraDescs {
    pub descs: PerFrame<Res<DescSet>>,
    pub view_buffers: PerFrame<Res<Buffer>>,
    pub proj_buffer: Res<Buffer>,
}

impl CameraDescs {
    fn new(renderer: &Renderer, camera: &Camera) -> Result<Self> {
        let pool = &renderer.static_pool;

        let view_buffers = PerFrame::try_from_fn(|_| {
            pool.create_buffer(
                MemoryLocation::Gpu,
                &BufferInfo {
                    usage: vk::BufferUsageFlags::UNIFORM_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    size: mem::size_of::<ViewUniform>() as vk::DeviceSize,
                },
            )
        })?;

        let proj_buffer = pool.create_buffer(
            MemoryLocation::Gpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                size: mem::size_of::<ProjUniform>() as vk::DeviceSize,
            },
        )?;

        renderer.transfer_with(|recorder| {
            recorder.update_buffer(proj_buffer.clone(), &ProjUniform::new(&camera));
        })?;

        let layout = Self::layout(renderer)?;
        let descs = PerFrame::try_from_fn(|index| {
            pool.create_desc_set(
                layout.clone(),
                &[
                    DescBinding::Buffer(proj_buffer.clone()),
                    DescBinding::Buffer(view_buffers[index].clone()),
                ],
            )
        })?;

        Ok(Self {
            descs,
            view_buffers,
            proj_buffer,
        })
    }

    pub fn handle_resize(&self, renderer: &Renderer, camera: &Camera) -> Result<()> {
        renderer.transfer_with(|recorder| {
            recorder.update_buffer(self.proj_buffer.clone(), &ProjUniform::new(&camera));
        })
    }

    pub fn layout(renderer: &Renderer) -> Result<Res<DescLayout>> {
        renderer.static_pool.create_desc_layout(&[
            DescLayoutSlot {
                binding: 0,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                count: rendi_render::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 1,
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                count: rendi_render::DescCount::Single,
            },
        ])
    }
}

pub struct DrawDesc {
    desc: Res<DescSet>,

    /// Buffer of draw commands for draw indirect calls.
    cmd_buffer: Res<Buffer>,

    /// Buffer of [`DrawCount`] used for draw indirect count.
    count_buffer: Res<Buffer>,

    /// Cpu copy of `count_buffer` for each render phase.
    count_host_buffers: [Res<Buffer>; 2],
}

impl DrawDesc {
    pub fn new(renderer: &Renderer, scene: &Scene) -> Result<Self> {
        let pool = &renderer.static_pool;

        let count_buffer = pool.create_buffer(
            MemoryLocation::Gpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDIRECT_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_SRC,
                size: mem::size_of::<DrawCount>() as vk::DeviceSize,
            },
        )?;

        let cmd_buffer = pool.create_buffer(
            MemoryLocation::Gpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::INDIRECT_BUFFER,
                size: (scene.primitives.len() * mem::size_of::<DrawCommand>()) as vk::DeviceSize,
            },
        )?;

        let count_host_buffers = array::try_from_fn(|_| {
            pool.create_buffer(
                MemoryLocation::Cpu,
                &BufferInfo {
                    usage: vk::BufferUsageFlags::STORAGE_BUFFER
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    size: mem::size_of::<DrawCount>() as vk::DeviceSize,
                },
            )
        })?;

        let flag_buffer = pool.create_buffer(
            MemoryLocation::Gpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER,
                size: (scene.primitives.len() * mem::size_of::<u32>()) as vk::DeviceSize,
            },
        )?;

        renderer.transfer_with(|recorder| {
            recorder.update_buffer(
                count_buffer.clone(),
                &DrawCount {
                    primitive_count: scene.primitive_count(),
                    command_count: 0,
                },
            );
        })?;

        let layout = Self::layout(renderer)?;
        let desc = pool.create_desc_set(
            layout,
            &[
                DescBinding::Buffer(cmd_buffer.clone()),
                DescBinding::Buffer(count_buffer.clone()),
                DescBinding::Buffer(flag_buffer.clone()),
            ],
        )?;

        Ok(Self {
            desc,
            cmd_buffer,
            count_buffer,
            count_host_buffers,
        })
    }

    pub fn layout(renderer: &Renderer) -> Result<Res<DescLayout>> {
        renderer.static_pool.create_desc_layout(&[
            DescLayoutSlot {
                binding: 0,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count: rendi_render::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 1,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count: rendi_render::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 2,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count: rendi_render::DescCount::Single,
            },
        ])
    }

    /// Get number of primitives last drawn using this descriptor.
    pub fn primitives_drawn(&self) -> Result<u32> {
        let mut count = 0;

        for buffer in &self.count_host_buffers {
            let mapped = buffer.get_mapped()?;
            let draw_count: &DrawCount = bytemuck::from_bytes(mapped.as_slice());

            count += draw_count.command_count;
        }

        Ok(count)
    }
}

/// Forward pass for rendering the geometry of a scene.
pub struct ForwardPass {
    pub lights: Lights,

    pub camera_descs: CameraDescs,
    pub draw_descs: PerFrame<DrawDesc>,

    pub depth_images: PerFrame<Res<ImageView>>,
    pub color_images: PerFrame<Res<ImageView>>,

    pub render: Res<RasterPipeline>,
    cull: Res<ComputePipeline>,

    depth_pyramid: DepthPyramid,

    render_target_info: RenderTargetInfo,
}

impl ForwardPass {
    pub fn new(renderer: &Renderer, camera: &Camera, scene: &Scene) -> Result<Self> {
        let pool = &renderer.pool;

        let extent = renderer.swapchain.extent_3d();
        let samples = renderer.device.sample_count();

        let depth_images = create_depth_images(renderer, extent, samples)?;
        let color_images = create_forward_color_images(renderer, extent, samples)?;

        let draw_descs = PerFrame::try_from_fn(|_| DrawDesc::new(renderer, scene))?;

        let depth_pyramid = DepthPyramid::new(renderer, &depth_images)?;
        let camera_descs = CameraDescs::new(renderer, &camera)?;

        let lights = Lights::new(
            &renderer,
            &camera_descs,
            &camera,
            scene.dir_light,
            &scene.point_lights,
        )?;

        let render_target_info = RenderTargetInfo {
            color_format: Some(color_images.any().image().format()),
            depth_format: depth_images.any().image().format(),
            sample_count: samples,
        };

        let render = {
            let vert_code = include_bytes_aligned_as!(u32, "../assets/shaders/pbr.vert.spv");
            let frag_code = include_bytes_aligned_as!(u32, "../assets/shaders/pbr.frag.spv");

            let vert_shader = pool.create_shader_module("main", vert_code)?;
            let frag_shader = pool.create_shader_module("main", frag_code)?;

            let prog = pool.create_raster_prog(vert_shader, frag_shader)?;

            let depth_stencil_info = &vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL)
                .depth_write_enable(true)
                .depth_test_enable(true);

            pool.create_raster_pipeline(RasterPipelineInfo {
                cull_mode: vk::CullModeFlags::BACK,
                render_target_info,
                depth_stencil_info,
                vertex_attributes: &[],
                push_consts: &[],
                prog,
            })?
        };

        let cull = {
            let code = include_bytes_aligned_as!(u32, "../assets/shaders/draw_cull.comp.spv");
            let shader = pool.create_shader_module("main", code)?;
            let prog = pool.create_compute_prog(shader)?;

            pool.create_compute_pipeline(
                prog,
                &[PushConstRange {
                    size: mem::size_of::<CullInfo>() as vk::DeviceSize,
                    stage: vk::ShaderStageFlags::COMPUTE,
                }],
            )?
        };

        Ok(Self {
            camera_descs,
            draw_descs,
            render_target_info,
            depth_images,
            color_images,
            depth_pyramid,
            lights,
            cull,
            render,
        })
    }

    pub fn render_target_info(&self) -> RenderTargetInfo {
        self.render_target_info
    }

    /// Blit depth pyramid level into swapchain image.
    pub fn pyramid_debug(
        &self,
        index: FrameIndex,
        swapchain_image: Res<Image>,
        recorder: &CommandRecorder,
        level: u32,
    ) {
        let pyramid = &self.depth_pyramid.pyramids[index];

        recorder.blit_image(&ImageBlitInfo {
            src: pyramid.clone(),
            dst: swapchain_image,
            filter: vk::Filter::NEAREST,
            src_mip: level,
            dst_mip: 0,
        });
    }

    pub fn handle_resize(&mut self, renderer: &Renderer, camera: &Camera) -> Result<()> {
        let extent = renderer.swapchain.extent_3d();
        let samples = renderer.device.sample_count();

        self.camera_descs.handle_resize(renderer, &camera)?;
        self.lights.handle_resize(renderer, &camera)?;
        self.depth_images = create_depth_images(renderer, extent, samples)?;
        self.color_images = create_forward_color_images(renderer, extent, samples)?;

        self.depth_pyramid = DepthPyramid::new(renderer, &self.depth_images)?;

        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum RenderPhase {
    /// The job of the 1st phase is to update the draw buffers using the phase 2 cull data from
    /// last frame. It still does frustrum culling.
    ///
    /// It then draws these primitives.
    Phase1 = 1,

    /// The 2nd phase runs after rendering the results from the 1st phase and generating a new
    /// depth pyramid. It updates the draw flags using both occlusion and frustrum culling.
    /// It draws primitives *if and only if* they wasn't drawn in phase 1 and is now visible.
    Phase2 = 2,
}

pub fn cull(
    pass: &ForwardPass,
    scene: &Scene,
    camera: &Camera,
    render_phase: RenderPhase,
    index: FrameIndex,
    recorder: &CommandRecorder,
) {
    let draw_desc = &pass.draw_descs[index];

    recorder.update_buffer(
        draw_desc.count_buffer.clone(),
        &DrawCount {
            primitive_count: scene.primitive_count(),
            command_count: 0,
        },
    );

    recorder.buffer_barrier(&BufferBarrierInfo {
        buffer: draw_desc.count_buffer.clone(),
        src_mask: vk::AccessFlags2::TRANSFER_WRITE,
        dst_mask: vk::AccessFlags2::SHADER_WRITE | vk::AccessFlags2::SHADER_READ,
        src_stage: vk::PipelineStageFlags2::TRANSFER,
        dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
    });

    recorder.bind_descs(&DescBindInfo {
        bind_point: vk::PipelineBindPoint::COMPUTE,
        layout: pass.cull.layout(),
        descs: &[
            pass.camera_descs.descs[index].clone(),
            draw_desc.desc.clone(),
            scene.desc.clone(),
            pass.depth_pyramid.sampled[index].clone(),
        ],
    });

    // If it's the 2nd phase, we have to wait for the pyramid to be created.
    if let RenderPhase::Phase2 = render_phase {
        let pyramid = &pass.depth_pyramid.pyramids[index];

        recorder.image_barrier(&ImageBarrierInfo {
            flags: vk::DependencyFlags::BY_REGION,
            mips: pyramid.mip_levels(),
            new_layout: vk::ImageLayout::GENERAL,
            src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            src_mask: vk::AccessFlags2::SHADER_WRITE,
            dst_mask: vk::AccessFlags2::SHADER_READ,
            image: pyramid.clone(),
        });
    }

    let cull_info = CullInfo {
        frustrum_planes: camera.frustrum_planes(),
        pyramid_width: pass.depth_pyramid.width as f32,
        pyramid_height: pass.depth_pyramid.height as f32,
        pyramid_mip_count: pass.depth_pyramid.mip_levels,
        phase: render_phase as u32,
        lod_base: 20.0,
        lod_step: 5.0,
        padding: [0x0; 2],
    };

    recorder.push_consts(
        pass.cull.layout(),
        &[PushConst {
            stage: vk::ShaderStageFlags::COMPUTE,
            bytes: bytemuck::bytes_of(&cull_info),
        }],
    );

    recorder.dispatch(
        pass.cull.clone(),
        [scene.primitive_count().div_ceil(64), 1, 1],
    );

    recorder.buffer_barrier(&BufferBarrierInfo {
        buffer: draw_desc.cmd_buffer.clone(),
        src_mask: vk::AccessFlags2::SHADER_WRITE,
        dst_mask: vk::AccessFlags2::INDIRECT_COMMAND_READ,
        src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        dst_stage: vk::PipelineStageFlags2::DRAW_INDIRECT,
    });

    let host_buffer = match render_phase {
        RenderPhase::Phase1 => draw_desc.count_host_buffers[0].clone(),
        RenderPhase::Phase2 => draw_desc.count_host_buffers[1].clone(),
    };

    recorder.copy_buffers(draw_desc.count_buffer.clone(), host_buffer);

    recorder.buffer_barrier(&BufferBarrierInfo {
        buffer: draw_desc.count_buffer.clone(),
        src_mask: vk::AccessFlags2::SHADER_WRITE,
        dst_mask: vk::AccessFlags2::TRANSFER_READ,
        src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        dst_stage: vk::PipelineStageFlags2::TRANSFER,
    });
}

fn render(
    renderer: &Renderer,
    pass: &ForwardPass,
    scene: &Scene,
    render_phase: RenderPhase,
    index: FrameIndex,
    recorder: &CommandRecorder,
) {
    let camera_desc = &pass.camera_descs.descs[index];
    let draw_desc = &pass.draw_descs[index];

    let load_op = match render_phase {
        RenderPhase::Phase1 => vk::AttachmentLoadOp::CLEAR,
        RenderPhase::Phase2 => vk::AttachmentLoadOp::LOAD,
    };

    let render_info = RenderInfo {
        color_target: Some(pass.color_images[index].clone()),
        depth_target: pass.depth_images[index].clone(),

        swapchain: renderer.swapchain.clone(),

        color_load_op: load_op,
        depth_load_op: load_op,
    };

    recorder.render(&render_info, |recorder| {
        recorder.bind_index_buffer(scene.index_buffer.clone(), vk::IndexType::UINT32);
        recorder.bind_raster_pipeline(pass.render.clone());

        recorder.bind_descs(&DescBindInfo {
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            layout: pass.render.layout(),
            descs: &[
                camera_desc.clone(),
                draw_desc.desc.clone(),
                scene.desc.clone(),
                pass.lights.descs[index].clone(),
            ],
        });

        recorder.draw_indexed_indirect_count(&IndexedIndirectDrawInfo {
            draw_command_size: mem::size_of::<DrawCommand>() as vk::DeviceSize,
            draw_buffer: draw_desc.cmd_buffer.clone(),
            count_buffer: draw_desc.count_buffer.clone(),
            max_draw_count: scene.primitive_count(),
            count_offset: 0,
            draw_offset: 0,
        });
    });
}

fn update_depth_pyramid(
    pyramid: &DepthPyramid,
    frame_index: FrameIndex,
    depth_image: &Res<ImageView>,
    recorder: &CommandRecorder,
) {
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
        image: pyramid.stagings[frame_index].image().clone(),
        mips: pyramid.stagings[frame_index].image().mip_levels(),
    });

    recorder.bind_descs(&DescBindInfo {
        bind_point: vk::PipelineBindPoint::COMPUTE,
        layout: pyramid.resolve.layout(),
        descs: &[pyramid.resolve_descs[frame_index].clone()],
    });

    let vk::Extent3D { width, height, .. } = pyramid.stagings[frame_index].image().extent(0);

    let info = DepthResolveInfo {
        image_size: UVec2::new(width, height),
        // `vk::SampleCountFlags` as raw maps to the amount of samples.
        samples: depth_image.image().sample_count().as_raw(),
    };

    recorder.push_consts(
        pyramid.resolve.layout(),
        &[PushConst {
            stage: vk::ShaderStageFlags::COMPUTE,
            bytes: bytemuck::bytes_of(&info),
        }],
    );

    recorder.dispatch(
        pyramid.resolve.clone(),
        [width.div_ceil(16), height.div_ceil(16), 1],
    );

    recorder.image_barrier(&ImageBarrierInfo {
        flags: vk::DependencyFlags::BY_REGION,
        src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        src_mask: vk::AccessFlags2::SHADER_WRITE,
        dst_mask: vk::AccessFlags2::SHADER_READ,
        new_layout: vk::ImageLayout::GENERAL,
        image: pyramid.stagings[frame_index].image().clone(),
        mips: pyramid.stagings[frame_index].image().mip_levels(),
    });

    recorder.image_barrier(&ImageBarrierInfo {
        flags: vk::DependencyFlags::BY_REGION,
        src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
        dst_stage: vk::PipelineStageFlags2::empty(),
        src_mask: vk::AccessFlags2::SHADER_READ,
        dst_mask: vk::AccessFlags2::empty(),
        new_layout: vk::ImageLayout::ATTACHMENT_OPTIMAL,
        image: depth_image.image().clone(),
        mips: depth_image.image().mip_levels(),
    });

    //
    // Reduce from each level to the next in depth pyramid.
    //

    recorder.bind_descs(&DescBindInfo {
        bind_point: vk::PipelineBindPoint::COMPUTE,
        layout: pyramid.reduce.layout(),
        descs: &[
            pyramid.sampled[frame_index].clone(),
            pyramid.storage[frame_index].clone(),
        ],
    });

    let level = &pyramid.pyramids[frame_index];

    for target in level.mip_levels() {
        let vk::Extent3D { width, height, .. } = level.extent(target);
        let info = DepthReduceInfo {
            image_size: UVec2::new(width, height),
            target,
        };

        let layout = pyramid.reduce.layout();

        recorder.push_consts(
            layout,
            &[PushConst {
                stage: vk::ShaderStageFlags::COMPUTE,
                bytes: bytemuck::bytes_of(&info),
            }],
        );

        recorder.dispatch(pyramid.reduce.clone(), [width / 32, height / 32, 1]);

        recorder.image_barrier(&ImageBarrierInfo {
            flags: vk::DependencyFlags::BY_REGION,
            src_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
            src_mask: vk::AccessFlags2::SHADER_WRITE,
            dst_mask: vk::AccessFlags2::SHADER_READ,
            new_layout: vk::ImageLayout::GENERAL,
            image: level.clone(),
            mips: target..target + 1,
        });
    }
}

pub fn draw(
    renderer: &Renderer,
    pass: &ForwardPass,
    scene: &Scene,
    camera: &Camera,
    index: FrameIndex,
    recorder: &CommandRecorder,
) {
    let view_buffer = &pass.camera_descs.view_buffers[index];

    recorder.update_buffer(view_buffer.clone(), &ViewUniform::new(&camera));
    recorder.buffer_barrier(&BufferBarrierInfo {
        buffer: view_buffer.clone(),
        src_mask: vk::AccessFlags2::TRANSFER_WRITE,
        dst_mask: vk::AccessFlags2::SHADER_READ,
        src_stage: vk::PipelineStageFlags2::TRANSFER,
        dst_stage: vk::PipelineStageFlags2::COMPUTE_SHADER,
    });

    light::prepare_lights(&pass.lights, &pass.camera_descs, index, recorder);

    cull(pass, scene, camera, RenderPhase::Phase1, index, recorder);
    render(renderer, pass, scene, RenderPhase::Phase1, index, recorder);

    update_depth_pyramid(
        &pass.depth_pyramid,
        index,
        &pass.depth_images[index],
        recorder,
    );

    cull(pass, scene, camera, RenderPhase::Phase2, index, recorder);
    render(renderer, pass, scene, RenderPhase::Phase2, index, recorder);
}

fn create_depth_images(
    renderer: &Renderer,
    extent: vk::Extent3D,
    samples: vk::SampleCountFlags,
) -> Result<PerFrame<Res<ImageView>>> {
    PerFrame::try_from_fn(|_| {
        let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_SRC
            | vk::ImageUsageFlags::SAMPLED;

        let image = renderer.pool.create_image(
            MemoryLocation::Gpu,
            &ImageInfo {
                aspect_flags: vk::ImageAspectFlags::DEPTH,
                format: DEPTH_IMAGE_FORMAT,
                kind: ImageKind::RenderTarget {
                    queue: renderer.graphics_queue(),
                    samples,
                },
                mip_levels: 1,
                extent,
                usage,
            },
        )?;

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
    PerFrame::try_from_fn(|_| {
        let usage = vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC;

        let image = renderer.pool.create_image(
            MemoryLocation::Gpu,
            &ImageInfo {
                aspect_flags: vk::ImageAspectFlags::COLOR,
                format: renderer.swapchain.format(),
                kind: ImageKind::RenderTarget {
                    queue: renderer.graphics_queue(),
                    samples,
                },
                mip_levels: 1,
                extent,
                usage,
            },
        )?;

        renderer.pool.create_image_view(&ImageViewInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            mips: image.mip_levels(),
            image,
        })
    })
}

/// Gpu resources of a scene.
pub struct Scene {
    /// All the primitives in the scene.
    primitives: Vec<Primitive>,

    /// All the point lights in the scene.
    point_lights: Vec<PointLight>,

    /// The directional light, lighting the scene.
    dir_light: DirLight,

    /// Descriptor of textures and geometry of the scene.
    desc: Res<DescSet>,

    /// Index buffer of all geomtry and lods in the scene.
    index_buffer: Res<Buffer>,
}

impl Scene {
    pub fn from_scene_asset(
        renderer: &Renderer,
        scene: &rendi_asset::Scene,
        dir_light: DirLight,
        lights: &[PointLight],
    ) -> Result<Self> {
        let pool = &renderer.static_pool;

        let instance_data: Vec<_> = scene
            .instances
            .iter()
            .map(|instance| InstanceData {
                transform: instance.transform,
            })
            .collect();

        trace!("loading {} instances", scene.instances.len());

        let primitives: Vec<Primitive> = scene
            .instances
            .iter()
            .enumerate()
            .flat_map(|(i, instance)| {
                scene.meshes[instance.mesh]
                    .primitives
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

                        let rendi_asset::Material {
                            albedo_map,
                            specular_map,
                            normal_map,
                            ..
                        } = scene.materials[prim.material];

                        let (albedo_map, specular_map, normal_map) =
                            (albedo_map as u32, specular_map as u32, normal_map as u32);

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

        trace!("loading {} primitives", primitives.len());

        let staging_pool = ResourcePool::new(renderer.device.clone());

        let primitive_data = bytemuck::cast_slice(primitives.as_slice());
        let primitive_staging = staging_pool.create_buffer(
            MemoryLocation::Cpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                size: primitive_data.len() as vk::DeviceSize,
            },
        )?;

        let instance_data = bytemuck::cast_slice(instance_data.as_slice());
        let instance_staging = staging_pool.create_buffer(
            MemoryLocation::Cpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                size: instance_data.len() as vk::DeviceSize,
            },
        )?;

        let vertex_data = bytemuck::cast_slice(scene.vertices.as_slice());
        let vertex_staging = staging_pool.create_buffer(
            MemoryLocation::Cpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                size: vertex_data.len() as vk::DeviceSize,
            },
        )?;

        let index_data = bytemuck::cast_slice(scene.indices.as_slice());
        let index_staging = staging_pool.create_buffer(
            MemoryLocation::Cpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                size: index_data.len() as vk::DeviceSize,
            },
        )?;

        primitive_staging.get_mapped()?.fill(primitive_data);
        instance_staging.get_mapped()?.fill(instance_data);
        vertex_staging.get_mapped()?.fill(vertex_data);
        index_staging.get_mapped()?.fill(index_data);

        let instance_buffer = pool.create_buffer(
            MemoryLocation::Gpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                size: instance_data.len() as vk::DeviceSize,
            },
        )?;

        let primitive_buffer = pool.create_buffer(
            MemoryLocation::Gpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                size: primitive_data.len() as vk::DeviceSize,
            },
        )?;

        let vertex_buffer = pool.create_buffer(
            MemoryLocation::Gpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                size: vertex_data.len() as vk::DeviceSize,
            },
        )?;

        let index_buffer = pool.create_buffer(
            MemoryLocation::Gpu,
            &BufferInfo {
                usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                size: index_data.len() as vk::DeviceSize,
            },
        )?;

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

        let staging: Result<Vec<Vec<_>>> = scene
            .textures
            .iter()
            .map(|texture| {
                texture
                    .mips
                    .iter()
                    .map(|data| {
                        let buffer = staging_pool.create_buffer(
                            MemoryLocation::Cpu,
                            &BufferInfo {
                                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                                size: data.len() as vk::DeviceSize,
                            },
                        )?;

                        buffer.get_mapped()?.fill(data);

                        Ok(buffer)
                    })
                    .collect()
            })
            .collect();

        let staging = staging?;

        let images: Result<Vec<_>> = scene
            .textures
            .iter()
            .map(|texture| {
                pool.create_image(
                    MemoryLocation::Gpu,
                    &ImageInfo {
                        mip_levels: texture.mip_levels(),
                        format: texture.format.into(),
                        kind: ImageKind::Texture,
                        aspect_flags: vk::ImageAspectFlags::COLOR,
                        usage: vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED,
                        extent: vk::Extent3D {
                            width: texture.width,
                            height: texture.height,
                            depth: 1,
                        },
                    },
                )
            })
            .collect();

        let images = images?;

        renderer.transfer_with(|recorder| {
            for image in images.iter() {
                recorder.image_barrier(&ImageBarrierInfo {
                    flags: vk::DependencyFlags::BY_REGION,
                    mips: image.mip_levels(),
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
                    mips: image.mip_levels(),
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

        let desc_layout = pool.create_desc_layout(&[
            DescLayoutSlot {
                binding: 0,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count: rendi_render::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 1,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count: rendi_render::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 2,
                ty: vk::DescriptorType::STORAGE_BUFFER,
                count: rendi_render::DescCount::Single,
            },
            DescLayoutSlot {
                binding: 3,
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                count: rendi_render::DescCount::UnboundArray,
            },
        ])?;

        let sampler = pool.create_sampler()?;

        let desc = pool.create_desc_set(
            desc_layout,
            &[
                DescBinding::Buffer(instance_buffer.clone()),
                DescBinding::Buffer(primitive_buffer.clone()),
                DescBinding::Buffer(vertex_buffer.clone()),
                DescBinding::ImageArray(
                    sampler.clone(),
                    vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    &textures,
                ),
            ],
        )?;

        let point_lights = Vec::from(lights);

        Ok(Self {
            desc,
            primitives,
            index_buffer,
            point_lights,
            dir_light,
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
