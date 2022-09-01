use anyhow::Result;
use ash::vk;
use glam::{Vec3, Quat, Vec2, Mat4};
use nohash_hasher::NoHashHasher;

use std::{array, mem, hash};
use std::collections::HashMap;

use crate::core::*;
use crate::resource::*;
use asset::{Font, Glyph};

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::NoUninit)]
struct Vertex {
    pos: Vec3,
    texcoord: Vec2,
}

pub struct TextPass {
    pub pipeline: GraphicsPipeline,
    pub descriptor: DescriptorSet,

    vertex_buffers: [Res<Buffer>; FRAMES_IN_FLIGHT],
    index_buffers: [Res<Buffer>; FRAMES_IN_FLIGHT],

    /// The projection matrix used for rendering text.
    ///
    /// Orthographic for now, but could a perspective.
    proj: Mat4,

    text_objects: TextObjects,
}

impl TextPass {
    pub fn new(renderer: &Renderer, pool: &ResourcePool, font: &Font) -> Result<Self> {
        let text_objects = TextObjects::new(FontAtlas::new(font));

        let memory_flags =
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;

        let vertex_buffers = array::try_from_fn(|_| {
            Buffer::new(renderer, pool, memory_flags, &BufferReq {
                usage: vk::BufferUsageFlags::VERTEX_BUFFER,
                size : mem::size_of::<[Vertex; MAX_VERTEX_COUNT]>() as vk::DeviceSize,
            })
        })?;

        let index_buffers = array::try_from_fn(|_| {
            Buffer::new(renderer, pool, memory_flags, &BufferReq {
                usage: vk::BufferUsageFlags::INDEX_BUFFER,
                size : mem::size_of::<[u16; MAX_INDEX_COUNT]>() as vk::DeviceSize,
            })
        })?;

        let staging_pool = ResourcePool::with_block_size(128, 1024);

        let atlas_staging = Buffer::new(&renderer, &staging_pool, memory_flags, &BufferReq {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            size: font.atlas.data.len() as u64,
        })?;

        atlas_staging.get_mapped()?.fill(font.atlas.data.as_slice());

        let extent = vk::Extent3D { width: font.atlas.width, height: font.atlas.height, depth: 1 };
        let sampler = pool.alloc(TextureSampler::new(&renderer)?);
   
        let memory_flags = vk::MemoryPropertyFlags::DEVICE_LOCAL;

        let glyph_atlas = Image::new(&renderer, pool, memory_flags, &ImageReq {
            format: font.atlas.format.into(),
            kind: ImageKind::Texture, extent,
        })?;

        renderer.transfer_with(|recorder| {
            recorder.transition_image_layout(&glyph_atlas, vk::ImageLayout::TRANSFER_DST_OPTIMAL);
            recorder.copy_buffer_to_image(&atlas_staging, &glyph_atlas);
            recorder.transition_image_layout(&glyph_atlas, vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        })?;

        let layout = pool.alloc(DescriptorSetLayout::new(&renderer, &[LayoutBinding {
            ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            stage: vk::ShaderStageFlags::FRAGMENT,
            array_count: None,
        }])?);

        let descriptor = DescriptorSet::new_single(&renderer, layout, &[
            DescriptorBinding::Image(sampler.clone(), [glyph_atlas.clone()]),
        ])?;

        let depth_stencil_info = vk::PipelineDepthStencilStateCreateInfo::builder()
            .depth_test_enable(false)
            .depth_write_enable(false)
            .depth_compare_op(vk::CompareOp::LESS_OR_EQUAL);
           
        let vertex_code = include_bytes_aligned_as!(u32, "../assets/shaders/sdf.vert.spv");
        let fragment_code = include_bytes_aligned_as!(u32, "../assets/shaders/sdf.frag.spv");

        let vertex_module = ShaderModule::new(&renderer, "main", vertex_code)?;
        let fragment_module = ShaderModule::new(&renderer, "main", fragment_code)?;

        let push_consts = [vk::PushConstantRange::builder()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(mem::size_of::<Mat4>() as u32)
            .offset(0)
            .build()];

        let layout = pool.alloc(
            PipelineLayout::new(&renderer, &push_consts, &[descriptor.layout.clone()])?
        );

        let pipeline = GraphicsPipeline::new(&renderer, GraphicsPipelineReq {
            layout,
            vertex_attributes: &[
                vk::VertexInputAttributeDescription {
                    format: vk::Format::R32G32B32_SFLOAT,
                    binding: 0,
                    location: 0,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    format: vk::Format::R32G32_SFLOAT,
                    binding: 0,
                    location: 1,
                    offset: mem::size_of::<Vec3>() as u32,
                },
            ],
            vertex_bindings: &[vk::VertexInputBindingDescription {
                binding: 0,
                stride: mem::size_of::<Vertex>() as u32,
                input_rate: vk::VertexInputRate::VERTEX,
            }],
            depth_stencil_info: &depth_stencil_info,
            vertex_shader: &vertex_module,
            fragment_shader: &fragment_module,
            cull_mode: vk::CullModeFlags::BACK,
        })?;

        let width = renderer.swapchain.extent.width as f32;
        let height = renderer.swapchain.extent.height as f32;
        let proj = Mat4::orthographic_lh(0.0, width, 0.0, height, 0.0, 1.0);

        Ok(Self { text_objects, proj, pipeline, descriptor, index_buffers, vertex_buffers })
    }

    pub fn handle_resize(&mut self, renderer: &Renderer) {
        let width = renderer.swapchain.extent.width as f32;
        let height = renderer.swapchain.extent.height as f32;
        self.proj = Mat4::orthographic_lh(0.0, width, 0.0, height, 0.0, 1.0);
    }

    pub fn draw_text<F>(
        &mut self,
        recorder: &CommandRecorder,
        frame_index: usize,
        mut func: F,
    ) -> Result<()>
    where
        F: FnMut(&mut TextObjects),
    {
        self.text_objects.clear();

        func(&mut self.text_objects);

        let index_data = bytemuck::cast_slice(self.text_objects.indices.as_slice());
        let vertex_data = bytemuck::cast_slice(self.text_objects.vertices.as_slice());

        self.vertex_buffers[frame_index].get_mapped()?.fill_from_start(vertex_data);
        self.index_buffers[frame_index].get_mapped()?.fill_from_start(index_data);

        recorder.bind_graphics_pipeline(&self.pipeline);
        recorder.bind_descriptor_sets(
            frame_index,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline.layout(),
            &[&self.descriptor],
        );

        recorder.bind_index_buffer(&self.index_buffers[frame_index], vk::IndexType::UINT16);
        recorder.bind_vertex_buffer(&self.vertex_buffers[frame_index]);

        for label in &self.text_objects.labels {
            let proj_transform = self.proj * Mat4::from_scale_rotation_translation(
                Vec3::splat(label.scale),
                Quat::IDENTITY,
                label.pos,
            );

            recorder.push_constants(
                self.pipeline.layout(),
                vk::ShaderStageFlags::VERTEX,
                0,
                &proj_transform,
            );

            recorder.draw_indexed(IndexedDrawCall {
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
    font_atlas: FontAtlas,
    labels: Vec<TextLabel>,
    vertices: Vec<Vertex>,
    indices: Vec<u16>,
}

impl TextObjects {
    fn new(font_atlas: FontAtlas) -> Self {
        let vertices = Vec::with_capacity(MAX_VERTEX_COUNT);
        let indices = Vec::with_capacity(MAX_INDEX_COUNT);
        Self { font_atlas, labels: Vec::new(), vertices, indices }
    }

    fn clear(&mut self) {
        self.labels.clear();
        self.vertices.clear();
        self.indices.clear();
    }

    pub fn add_label(&mut self, scale: f32, pos: Vec3, text: &str) {
        let index_offset = self.indices.len() as u32;

        let mut index: u16 = self.vertices.len() as u16;
        let mut screen_pos = Vec2::new(0.0, 0.0);

        for c in text.chars() {
            let glyph = &self.font_atlas.glyph_map[&(c as u32)];
           
            screen_pos.y = glyph.scaled_offset.y;

            self.vertices.extend_from_slice(&[
                Vertex {
                    texcoord: Vec2::new(
                        glyph.texcoord_max.x,
                        glyph.texcoord_max.y,
                    ),
                    pos: Vec3::new(
                        screen_pos.x + glyph.scaled_dim.x + glyph.scaled_offset.x,
                        screen_pos.y + glyph.scaled_dim.y,
                        0.0,
                    ),
                },
                Vertex {
                    texcoord: Vec2::new(
                        glyph.texcoord_min.x,
                        glyph.texcoord_max.y,
                    ),
                    pos: Vec3::new(
                        screen_pos.x + glyph.scaled_offset.x,
                        screen_pos.y + glyph.scaled_dim.y,
                        0.0,
                    ),
                },
                Vertex {
                    texcoord: Vec2::new(
                        glyph.texcoord_min.x,
                        glyph.texcoord_min.y,
                    ),
                    pos: Vec3::new(
                        screen_pos.x + glyph.scaled_offset.x,
                        screen_pos.y,
                        0.0,
                    ),
                },
                Vertex {
                    texcoord: Vec2::new(
                        glyph.texcoord_max.x,
                        glyph.texcoord_min.y,
                    ),
                    pos: Vec3::new(
                        screen_pos.x + glyph.scaled_dim.x + glyph.scaled_offset.x,
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
            screen_pos.x += glyph.advance;
        }

        let index_count = self.indices.len() as u32 - index_offset;
     
        self.labels.push(TextLabel { index_offset, index_count, pos, scale });
    }
}

type GlyphMap = HashMap<u32, Glyph, hash::BuildHasherDefault<NoHashHasher<u32>>>;

struct FontAtlas {
    glyph_map: GlyphMap,
}

impl FontAtlas {
    fn new(font: &Font) -> Self {
        let hasher = hash::BuildHasherDefault::default();
        let mut glyph_map = GlyphMap::with_capacity_and_hasher(font.glyphs.len(), hasher);

        for glyph in font.glyphs.iter() {
            let codepoint = u32::from(glyph.codepoint.clone());
            glyph_map.insert(codepoint, glyph.clone());
        }

        Self { glyph_map }
    }
}

const MAX_VERTEX_COUNT: usize = 1028;
const MAX_INDEX_COUNT: usize = (MAX_VERTEX_COUNT / 4) * 6;
