use anyhow::Result;
use ash::vk;

use std::mem;

use crate::frame::{PerFrame, FrameIndex};
use crate::command::*;
use crate::core::*;
use crate::resource::*;

use rendi_math::prelude::*;
use rendi_res::Res;

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::NoUninit)]
struct Vertex {
    pos: Vec3,
    texcoord: Vec2,
}

pub struct TextPass {
    pub pipeline: Res<RasterPipeline>,
    pub desc: Res<DescSet>,

    vertex_buffers: PerFrame<Res<Buffer>>,
    index_buffers: PerFrame<Res<Buffer>>,

    /// The projection matrix used for rendering text.
    ///
    /// Orthographic for now, but could a perspective.
    proj: Mat4,

    text_objects: TextObjects,
}

impl TextPass {
    pub fn new(
        renderer: &Renderer,
        render_target_info: RenderTargetInfo,
        atlas: rendi_sdf::Atlas,
    ) -> Result<Self> {
        let pool = &renderer.static_pool;

        let vertex_buffers = PerFrame::try_from_fn(|_| {
            pool.create_buffer(MemoryLocation::Cpu, &BufferInfo {
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                size : mem::size_of::<[Vertex; MAX_VERTEX_COUNT]>() as vk::DeviceSize,
            })
        })?;

        let index_buffers = PerFrame::try_from_fn(|_| {
            pool.create_buffer(MemoryLocation::Cpu, &BufferInfo {
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                size : mem::size_of::<[u16; MAX_INDEX_COUNT]>() as vk::DeviceSize,
            })
        })?;

        let staging_pool = ResourcePool::with_block_size(renderer.device.clone(), 128, 1024);

        let atlas_staging = staging_pool.create_buffer(MemoryLocation::Cpu, &BufferInfo {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: atlas.image().base_image_data().len() as vk::DeviceSize,
        })?;

        atlas_staging
            .get_mapped()?
            .fill(atlas.image().base_image_data());

        let extent = vk::Extent3D {
            width: atlas.image().width,
            height: atlas.image().height,
            depth: 1,
        };

        let sampler = pool.create_sampler()?;
   
        let glyph_atlas = pool.create_image(MemoryLocation::Gpu, &ImageInfo {
            usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
            aspect_flags: vk::ImageAspectFlags::COLOR,
            kind: ImageKind::Texture, extent,
            format: atlas.image().format.into(),
            mip_levels: 1,
        })?;

        let view = pool.create_image_view(&ImageViewInfo {
            view_type: vk::ImageViewType::TYPE_2D,
            mips: glyph_atlas.mip_levels(),
            image: glyph_atlas.clone(),
        })?;

        renderer.transfer_with(|recorder| {
            recorder.image_barrier(&ImageBarrierInfo {
                flags: vk::DependencyFlags::BY_REGION,
                mips: glyph_atlas.mip_levels(),
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_stage: vk::PipelineStageFlags2::empty(),
                dst_stage: vk::PipelineStageFlags2::TRANSFER,
                src_mask: vk::AccessFlags2::empty(),
                dst_mask: vk::AccessFlags2::TRANSFER_WRITE,
                image: glyph_atlas.clone(),
            });

            recorder.copy_buffer_to_image(atlas_staging.clone(), glyph_atlas.clone(), 0);

            recorder.image_barrier(&ImageBarrierInfo {
                flags: vk::DependencyFlags::BY_REGION,
                mips: glyph_atlas.mip_levels(),
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_stage: vk::PipelineStageFlags2::TRANSFER,
                dst_stage: vk::PipelineStageFlags2::empty(),
                src_mask: vk::AccessFlags2::TRANSFER_WRITE,
                dst_mask: vk::AccessFlags2::empty(),
                image: glyph_atlas.clone(),
            });
        })?;

        let layout = pool.create_desc_layout(&[DescLayoutSlot {
            binding: 0,
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            count: rendi_shader::DescCount::Single,
        }])?;

        let desc = pool.create_desc_set(layout.clone(), &[
            DescBinding::Image(
                sampler.clone(),
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                view.clone()
            ),
        ])?;

        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
           
        let vert_code = include_bytes_aligned_as!(u32, "../assets/shaders/sdf.vert.spv");
        let frag_code = include_bytes_aligned_as!(u32, "../assets/shaders/sdf.frag.spv");

        let frag_shader = pool.create_shader_module("main", frag_code)?;
        let vert_shader = pool.create_shader_module("main", vert_code)?;

        let push_consts = &[PushConstRange {
            size: mem::size_of::<Mat4>() as vk::DeviceSize,
            stage: vk::ShaderStageFlags::VERTEX,
        }];

        let prog = pool.create_raster_prog(vert_shader, frag_shader)?;
        let pipeline = pool.create_raster_pipeline(RasterPipelineInfo {
            depth_stencil_info: &depth_stencil_info,
            cull_mode: vk::CullModeFlags::NONE,
            vertex_attributes: &[
                VertexAttribute {
                    format: vk::Format::R32G32B32_SFLOAT,
                    size: mem::size_of::<Vec3>() as vk::DeviceSize,
                },
                VertexAttribute {
                    format: vk::Format::R32G32_SFLOAT,
                    size: mem::size_of::<Vec2>() as vk::DeviceSize,
                },
            ],
            render_target_info,
            push_consts,
            prog,
        })?;

        let width = renderer.swapchain.size().x;
        let height = renderer.swapchain.size().y;

        let proj = Mat4::orthographic_lh(0.0, width, 0.0, height, 0.0, 1.0);
        let text_objects = TextObjects::new(atlas);

        Ok(Self { text_objects, proj, pipeline, desc, index_buffers, vertex_buffers })
    }

    pub fn handle_resize(&mut self, renderer: &Renderer) {
        let width = renderer.swapchain.size().x;
        let height = renderer.swapchain.size().y;
        self.proj = Mat4::orthographic_lh(0.0, width, 0.0, height, 0.0, 1.0);
    }

    pub fn draw_text<F>(
        &mut self,
        recorder: &DrawRecorder,
        frame_index: FrameIndex,
        mut func: F,
    ) -> Result<()>
    where
        F: FnMut(&mut TextObjects),
    {
        self.text_objects.clear();

        func(&mut self.text_objects);

        let vertex_data = bytemuck::cast_slice(self.text_objects.vertices.as_slice());
        let index_data = bytemuck::cast_slice(self.text_objects.indices.as_slice());

        let vertex_size = vertex_data.len() as vk::DeviceSize;
        let index_size = index_data.len() as vk::DeviceSize;

        self.vertex_buffers[frame_index]
            .get_mapped()?
            .fill_range(0..vertex_size, vertex_data);

        self.index_buffers[frame_index]
            .get_mapped()?
            .fill_range(0..index_size, index_data);

        recorder.bind_raster_pipeline(self.pipeline.clone());
        recorder.bind_descs(&DescBindInfo {
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            layout: self.pipeline.layout(),
            descs: &[self.desc.clone()],
        });

        recorder.bind_index_buffer(self.index_buffers[frame_index].clone(), vk::IndexType::UINT16);
        recorder.bind_vertex_buffer(self.vertex_buffers[frame_index].clone());

        for label in &self.text_objects.labels {
            let proj_transform = self.proj * Mat4::from_scale_rotation_translation(
                Vec3::splat(label.scale),
                Quat::from_xyzw(0.0, 0.0, 0.0, -1.0),
                label.pos,
            );

            recorder.push_consts(self.pipeline.layout(), &[PushConst {
                stage: vk::ShaderStageFlags::VERTEX,
                bytes: bytemuck::bytes_of(&proj_transform),
            }]);

            recorder.draw_indexed(IndexedDrawInfo {
                index_count: label.index_count,
                index_start: label.index_offset,
                vertex_offset: 0,
                instance: 0,
            });
        }

        Ok(())
    }
}

struct TextLabel {
    scale: f32,
    pos: Vec3,
    index_offset: u32,
    index_count: u32,
}

pub struct TextObjects {
    atlas: rendi_sdf::Atlas,
    labels: Vec<TextLabel>,
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
}

impl TextObjects {
    fn new(atlas: rendi_sdf::Atlas) -> Self {
        let vertices = Vec::with_capacity(MAX_VERTEX_COUNT);
        let indices = Vec::with_capacity(MAX_INDEX_COUNT);
        Self { atlas, labels: Vec::new(), vertices, indices }
    }

    fn clear(&mut self) {
        self.labels.clear();
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn add_label(&mut self, scale: f32, pos: Vec3, text: &str) {
        let index_offset = self.indices.len() as u32;

        let mut index  = self.vertices.len() as u16;
        let mut screen_pos = Vec2::new(0.0, 0.0);

        let mut prev_ch = None;

        for ch in text.chars() {
            let glyph = &self.atlas.glyph(ch).unwrap_or_else(|| {
                panic!("no char '{ch}' in atlas");
            });

            screen_pos.y = glyph.offset().y;

            if let Some(prev_ch) = prev_ch {
                screen_pos.x += self.atlas.kerning(prev_ch, ch);
            }

            self.vertices.extend_from_slice(&[
                Vertex {
                    texcoord: Vec2::new(
                        glyph.atlas_rect().max.x,
                        glyph.atlas_rect().max.y,
                    ),
                    pos: Vec3::new(
                        screen_pos.x + glyph.dim().x + glyph.offset().x,
                        screen_pos.y - glyph.dim().y,
                        0.0,
                    ),
                },
                Vertex {
                    texcoord: Vec2::new(
                        glyph.atlas_rect().min.x,
                        glyph.atlas_rect().max.y,
                    ),
                    pos: Vec3::new(
                        screen_pos.x + glyph.offset().x,
                        screen_pos.y - glyph.dim().y,
                        0.0,
                    ),
                },
                Vertex {
                    texcoord: Vec2::new(
                        glyph.atlas_rect().min.x,
                        glyph.atlas_rect().min.y,
                    ),
                    pos: Vec3::new(
                        screen_pos.x + glyph.offset().x,
                        screen_pos.y,
                        0.0,
                    ),
                },
                Vertex {
                    texcoord: Vec2::new(
                        glyph.atlas_rect().max.x,
                        glyph.atlas_rect().min.y,
                    ),
                    pos: Vec3::new(
                        screen_pos.x + glyph.dim().x + glyph.offset().x,
                        screen_pos.y,
                        0.0,
                    ),
                },
            ]);

            self.indices.extend_from_slice(&[
                index,
                1 + index,
                2 + index,
                2 + index,
                3 + index,
                index,
            ]);

            index += 4;
            screen_pos.x += glyph.advance();
            prev_ch = Some(ch);
        }

        let index_count = self.indices.len() as u32 - index_offset;
     
        self.labels.push(TextLabel {
            index_offset,
            index_count,
            scale,
            pos,
        });
    }
}

const MAX_VERTEX_COUNT: usize = 1028;
const MAX_INDEX_COUNT: usize = (MAX_VERTEX_COUNT / 4) * 6;
