use std::{mem, slice};

use ash::vk::{self, MemoryPropertyFlags};
use eyre::Result;
use glam::Vec2;

use crate::constants::Constants;
use crate::{hal, Descriptors, Update};

pub struct GuiRequest {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

pub(super) struct Texture {
    pub id: epaint::TextureId,
    pub image: hal::Image,
    pub memory: hal::Memory,
}

pub(super) struct Gui {
    pub vertices: hal::Buffer,
    pub indices: hal::Buffer,
    memory: hal::Memory,
    pipeline: hal::Pipeline,
    descriptor_layout: hal::DescriptorLayout,
    sampler: hal::Sampler,
    pub textures: Vec<Texture>,
}

impl Gui {
    pub fn new(
        device: &hal::Device,
        swapchain: &hal::Swapchain,
    ) -> Result<Self> {
        let (indices, vertices, memory) = create_buffers(
            device,
            INITIAL_VERTEX_BUFFER_SIZE,
            INITIAL_INDEX_BUFFER_SIZE,
        )?;
        let textures = Vec::default();
        let descriptor_layout = hal::DescriptorLayoutBuilder::default()
            .binding(vk::DescriptorType::UNIFORM_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .array_binding(vk::DescriptorType::COMBINED_IMAGE_SAMPLER, 1024)
            .build(device)?;
        let pipeline = create_pipeline(device, swapchain, &descriptor_layout)?;
        let sampler = hal::Sampler::new(
            device,
            &hal::SamplerRequest {
                filter: vk::Filter::LINEAR,
                address_mode: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ..Default::default()
            },
        )?;
        Ok(Self {
            descriptor_layout,
            indices,
            vertices,
            memory,
            textures,
            pipeline,
            sampler,
        })
    }

    pub fn update(
        &mut self,
        device: &hal::Device,
        swapchain: &hal::Swapchain,
        request: &GuiRequest,
    ) -> Result<GuiUpdate> {
        let mut update = GuiUpdate::default();

        update.screen_size_in_points = Vec2 {
            x: swapchain.extent.width as f32,
            y: swapchain.extent.height as f32,
        } / request.pixels_per_point;

        // Update textures.
        for id in &request.textures_delta.free {
            self.textures
                .swap_remove(self.texture_index(*id))
                .image
                .destroy(device);
        }

        for (id, delta) in &request.textures_delta.set {
            let (bytes, width, height) = match &delta.image {
                egui::ImageData::Color(image) => {
                    let bytes =
                        bytemuck::cast_slice::<_, u8>(&image.pixels).to_vec();
                    (bytes, image.width() as u32, image.height() as u32)
                }
                egui::ImageData::Font(image) => {
                    let bytes = image
                        .srgba_pixels(None)
                        .flat_map(|pixel| pixel.to_array())
                        .collect();
                    (bytes, image.width() as u32, image.height() as u32)
                }
            };
            let offset = if let Some([x, y]) = delta.pos {
                vk::Offset3D { x: x as i32, y: y as i32, z: 0 }
            } else {
                let extent = vk::Extent3D { width, height, depth: 1 };
                self.textures.push(create_texture(device, *id, extent)?);
                vk::Offset3D::default()
            };
            let extent = vk::Extent3D {
                width: width as u32,
                height: height as u32,
                depth: 1,
            };
            update.texture_updates.push(TextureUpdate {
                bytes: bytes.into_boxed_slice(),
                index: self.texture_index(*id),
                extent,
                offset,
            });
        }

        // Update buffers.
        for primitive in &request.primitives {
            let scissor_rect = scissor_rect(
                &primitive.clip_rect,
                request.pixels_per_point,
                swapchain.extent,
            );

            match &primitive.primitive {
                epaint::Primitive::Callback(_) => todo!(),
                epaint::Primitive::Mesh(mesh) => {
                    let vertex_offset = (update.vertices.len()
                        / mem::size_of::<epaint::Vertex>())
                        as i32;
                    let index_start =
                        (update.indices.len() / mem::size_of::<u32>()) as u32;
                    update.vertices.extend_from_slice(bytemuck::cast_slice(
                        &mesh.vertices,
                    ));
                    update
                        .indices
                        .extend_from_slice(bytemuck::cast_slice(&mesh.indices));
                    update.draws.push(Draw {
                        scissor_rect,
                        texture_index: self.texture_index(mesh.texture_id)
                            as u32,
                        index_count: mesh.indices.len() as u32,
                        vertex_offset,
                        index_start,
                    });
                }
            }
        }

        let index_buffer_size = update.indices.len() as vk::DeviceSize;
        let vertex_buffer_size = update.vertices.len() as vk::DeviceSize;

        let recreated = self.indices.size < index_buffer_size
            || self.vertices.size < vertex_buffer_size;
        if recreated {
            let index_buffer_size =
                index_buffer_size.next_multiple_of(INITIAL_INDEX_BUFFER_SIZE);
            let vertex_buffer_size =
                vertex_buffer_size.next_multiple_of(INITIAL_VERTEX_BUFFER_SIZE);
            self.indices.destroy(device);
            self.vertices.destroy(device);
            self.memory.free(device);
            (self.indices, self.vertices, self.memory) =
                create_buffers(device, index_buffer_size, vertex_buffer_size)?;
        }

        update.update.set(Update::RECREATE_DESCRIPTORS, recreated);

        Ok(update)
    }

    fn texture_index(&self, id: epaint::TextureId) -> usize {
        self.textures
            .iter()
            .position(|texture| texture.id == id)
            .expect("invalid texture id")
    }

    pub fn destroy(&self, device: &hal::Device) {
        self.descriptor_layout.destroy(device);
        self.pipeline.destroy(device);
        self.indices.destroy(device);
        self.vertices.destroy(device);
        self.sampler.destroy(device);
        self.memory.free(device);
        for texture in &self.textures {
            texture.image.destroy(device);
            texture.memory.free(device);
        }
    }
}

fn create_texture(
    device: &hal::Device,
    id: epaint::TextureId,
    extent: vk::Extent3D,
) -> Result<Texture> {
    let mut image = hal::Image::new(
        device,
        &hal::ImageRequest {
            format: vk::Format::R8G8B8A8_SRGB,
            mip_level_count: 1,
            usage: vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::TRANSFER_SRC,
            extent,
        },
    )?;
    let memory =
        hal::image_memory(device, &image, MemoryPropertyFlags::DEVICE_LOCAL)?;
    let view_request =
        hal::ImageViewRequest { mip_level_count: 1, base_mip_level: 0 };
    image.add_view(device, view_request)?;
    Ok(Texture { image, memory, id })
}

pub(super) fn create_shaders(device: &hal::Device) -> Result<[hal::Shader; 2]> {
    let vertex_shader = hal::Shader::new(
        device,
        &hal::ShaderRequest {
            stage: vk::ShaderStageFlags::VERTEX,
            source: vk_shader_macros::include_glsl!(
                "src/shaders/gui/vert.glsl",
                kind: vert,
            ),
        },
    )?;
    let fragment_shader = hal::Shader::new(
        device,
        &hal::ShaderRequest {
            stage: vk::ShaderStageFlags::FRAGMENT,
            source: vk_shader_macros::include_glsl!(
                "src/shaders/gui/frag.glsl",
                kind: frag,
            ),
        },
    )?;
    Ok([vertex_shader, fragment_shader])
}

pub(super) fn create_pipeline(
    device: &hal::Device,
    swapchain: &hal::Swapchain,
    descriptor_layout: &hal::DescriptorLayout,
) -> Result<hal::Pipeline> {
    let shaders = create_shaders(device)?;
    let pipeline_layout = hal::PipelineLayout {
        descriptors: &[&descriptor_layout],
        push_constant: Some(vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX
                | vk::ShaderStageFlags::FRAGMENT,
            size: mem::size_of::<DrawConstants>() as u32,
            offset: 0,
        }),
    };
    let specializations = hal::Specializations::default();
    let pipeline = hal::Pipeline::graphics(
        device,
        &pipeline_layout,
        &hal::GraphicsPipelineRequest {
            color_attachments: &[hal::ColorAttachment {
                format: swapchain.format,
                blend: Some(hal::Blend {
                    src: vk::BlendFactor::ONE,
                    dst: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                }),
            }],
            depth_format: None,
            cull_mode: vk::CullModeFlags::NONE,
            shaders: &[
                hal::ShaderStage {
                    shader: &shaders[0],
                    specializations: &specializations,
                },
                hal::ShaderStage {
                    shader: &shaders[1],
                    specializations: &specializations,
                },
            ],
        },
    )?;
    for shader in shaders {
        shader.destroy(device);
    }
    Ok(pipeline)
}

fn create_buffers(
    device: &hal::Device,
    index_buffer_size: vk::DeviceSize,
    vertex_buffer_size: vk::DeviceSize,
) -> Result<(hal::Buffer, hal::Buffer, hal::Memory)> {
    let indices = hal::Buffer::new(
        device,
        &hal::BufferRequest {
            size: index_buffer_size,
            kind: hal::BufferKind::Index,
        },
    )?;
    let vertices = hal::Buffer::new(
        device,
        &hal::BufferRequest {
            size: vertex_buffer_size,
            kind: hal::BufferKind::Storage,
        },
    )?;
    let mut allocator = hal::Allocator::new(device);
    allocator.alloc_buffer(&indices);
    allocator.alloc_buffer(&vertices);
    let memory = allocator.finish(vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
    Ok((indices, vertices, memory))
}

fn scissor_rect(
    clip_rect: &epaint::Rect,
    pixels_per_point: f32,
    screen_size: vk::Extent2D,
) -> vk::Rect2D {
    let clip_min_x = (pixels_per_point * clip_rect.min.x).round() as u32;
    let clip_min_y = (pixels_per_point * clip_rect.min.y).round() as u32;
    let clip_max_x = (pixels_per_point * clip_rect.max.x).round() as u32;
    let clip_max_y = (pixels_per_point * clip_rect.max.y).round() as u32;

    let clip_min_x = clip_min_x.clamp(0, screen_size.width);
    let clip_min_y = clip_min_y.clamp(0, screen_size.height);
    let clip_max_x = clip_max_x.clamp(clip_min_x, screen_size.width);
    let clip_max_y = clip_max_y.clamp(clip_min_y, screen_size.height);

    vk::Rect2D {
        offset: vk::Offset2D { x: clip_min_x as i32, y: clip_min_y as i32 },
        extent: vk::Extent2D {
            width: clip_max_x - clip_min_x,
            height: clip_max_y - clip_min_y,
        },
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::NoUninit)]
struct DrawConstants {
    screen_size_in_points: Vec2,
    texture_index: u32,
}

#[derive(Debug)]
struct TextureUpdate {
    index: usize,
    bytes: Box<[u8]>,
    offset: vk::Offset3D,
    extent: vk::Extent3D,
}

#[derive(Default, Debug)]
pub(super) struct GuiUpdate {
    vertices: Vec<u8>,
    indices: Vec<u8>,
    draws: Vec<Draw>,
    texture_updates: Vec<TextureUpdate>,
    screen_size_in_points: Vec2,
    pub update: Update,
}

#[derive(Debug)]
pub(super) struct Draw {
    scissor_rect: vk::Rect2D,
    vertex_offset: i32,
    index_start: u32,
    index_count: u32,
    texture_index: u32,
}

pub(super) fn image_writes<'a>(
    gui: &'a Gui,
    update: &'a GuiUpdate,
) -> impl Iterator<Item = hal::ImageWrite<'a>> {
    update.texture_updates.iter().map(|update| {
        let TextureUpdate { bytes, extent, offset, index } = update;
        let mips = slice::from_ref(bytes);
        let image = &gui.textures[*index].image;
        hal::ImageWrite { mips, image, extent: *extent, offset: *offset }
    })
}

pub(super) fn buffer_writes<'a>(
    gui: &'a Gui,
    update: &'a GuiUpdate,
) -> impl Iterator<Item = hal::BufferWrite<'a>> {
    let writes = [
        hal::BufferWrite { buffer: &gui.indices, data: &update.indices },
        hal::BufferWrite { buffer: &gui.vertices, data: &update.vertices },
    ];
    writes.into_iter()
}

pub(super) fn create_descriptor(
    device: &hal::Device,
    gui: &Gui,
    constants: &Constants,
    data: &mut hal::DescriptorData,
) -> hal::Descriptor {
    let textures = gui
        .textures
        .iter()
        .map(|texture| texture.image.view(&hal::ImageViewRequest::BASE));
    data.builder(device, &gui.descriptor_layout)
        .uniform_buffer(&constants.buffer)
        .storage_buffer(&gui.vertices)
        .combined_image_samplers(&gui.sampler, textures)
        .build()
}

pub(super) fn render(
    device: &hal::Device,
    command_buffer: &mut hal::CommandBuffer,
    swapchain_image: &hal::Image,
    descriptors: &Descriptors,
    update: &GuiUpdate,
    gui: &Gui,
) {
    let extent = vk::Extent2D {
        width: swapchain_image.extent.width,
        height: swapchain_image.extent.height,
    };

    command_buffer
        .bind_pipeline(device, &gui.pipeline)
        .set_viewport(
            device,
            &[vk::Viewport {
                width: extent.width as f32,
                height: extent.height as f32,
                max_depth: 1.0,
                ..Default::default()
            }],
        )
        .bind_index_buffer(device, &gui.indices)
        .bind_descriptor(device, &gui.pipeline, &descriptors.gui)
        .begin_rendering(
            device,
            &hal::BeginRendering {
                depth_attachment: None,
                color_attachments: &[hal::Attachment {
                    view: swapchain_image.view(&hal::ImageViewRequest::BASE),
                    load: hal::Load::Load,
                }],
                extent,
            },
        );
    for draw in &update.draws {
        let draw_constants = DrawConstants {
            screen_size_in_points: update.screen_size_in_points,
            texture_index: draw.texture_index,
        };

        command_buffer
            .set_scissor(device, &[draw.scissor_rect])
            .push_constants(
                device,
                &gui.pipeline,
                bytemuck::bytes_of(&draw_constants),
            )
            .draw_indexed(
                device,
                &hal::DrawIndexed {
                    index_count: draw.index_count,
                    first_index: draw.index_start,
                    vertex_offset: draw.vertex_offset,
                    instance_count: 1,
                    first_instance: 0,
                },
            );
    }
    command_buffer.end_rendering(device);
}

const INITIAL_VERTEX_BUFFER_SIZE: vk::DeviceSize =
    10 * 1024 * mem::size_of::<epaint::Vertex>() as vk::DeviceSize;
const INITIAL_INDEX_BUFFER_SIZE: vk::DeviceSize =
    30 * 1024 * mem::size_of::<u32>() as vk::DeviceSize;
