use std::mem;

use ash::vk;
use eyre::Result;

use crate::command;
use crate::device::Device;
use crate::resources::{
    self, Allocator, Buffer, BufferKind, BufferRequest, BufferWrite, Image,
    ImageRequest, ImageViewRequest, ImageWrite, Memory,
};

pub(super) struct Scene {
    indices: Buffer,
    vertices: Buffer,
    meshlets: Buffer,
    meshlet_data: Buffer,
    materials: Buffer,
    meshes: Buffer,
    textures: Vec<Image>,
    memory: Memory,
}

impl Scene {
    pub fn new(device: &Device, scene: &asset::Scene) -> Result<Self> {
        let indices = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.indices) as vk::DeviceSize,
                kind: BufferKind::Index,
            },
        )?;
        let vertices = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.vertices) as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?;
        let meshlets = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.meshlets) as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?;
        let meshlet_data = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.meshlet_data) as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?;
        let meshes = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.meshes) as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?;
        let materials = Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.materials) as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?;
        let mut textures: Vec<_> = scene
            .textures
            .iter()
            .map(|texture| {
                let mip_level_count = texture.mips.len() as u32;
                let usage = vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::SAMPLED;
                let request = ImageRequest {
                    format: texture_kind_format(texture.kind),
                    extent: vk::Extent3D {
                        width: texture.width,
                        height: texture.height,
                        depth: 1,
                    },
                    mip_level_count,
                    usage,
                };
                Image::new(device, &request)
            })
            .collect::<Result<_>>()?;

        let mut allocator = Allocator::new(device);
        for texture in &textures {
            allocator.alloc_image(texture);
        }

        let memory = allocator
            .alloc_buffer(&indices)
            .alloc_buffer(&vertices)
            .alloc_buffer(&meshlets)
            .alloc_buffer(&meshlet_data)
            .alloc_buffer(&meshes)
            .alloc_buffer(&materials)
            .finish(vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        for (image, texture) in textures.iter_mut().zip(scene.textures.iter()) {
            image.add_view(
                device,
                ImageViewRequest {
                    mip_level_count: texture.mips.len() as u32,
                    base_mip_level: 0,
                },
            )?;
        }

        let texture_writes: Vec<_> = textures
            .iter()
            .zip(scene.textures.iter())
            .map(|(image, texture)| ImageWrite {
                offset: vk::Offset3D::default(),
                mips: &texture.mips,
                image,
            })
            .collect();

        let buffer_writes = [
            BufferWrite {
                buffer: &indices,
                data: bytemuck::cast_slice(&scene.indices),
            },
            BufferWrite {
                buffer: &vertices,
                data: bytemuck::cast_slice(&scene.vertices),
            },
            BufferWrite {
                buffer: &meshlets,
                data: bytemuck::cast_slice(&scene.meshlets),
            },
            BufferWrite {
                buffer: &meshlet_data,
                data: bytemuck::cast_slice(&scene.meshlet_data),
            },
            BufferWrite {
                buffer: &meshes,
                data: bytemuck::cast_slice(&scene.meshes),
            },
            BufferWrite {
                buffer: &materials,
                data: bytemuck::cast_slice(&scene.materials),
            },
        ];

        let scratch = command::quickie(device, |command_buffer| {
            let buffer_scratch = resources::upload_buffer_data(
                device,
                &command_buffer,
                &buffer_writes,
            )?;
            let image_scratch = resources::upload_image_data(
                device,
                &command_buffer,
                &texture_writes,
            )?;
            Ok([buffer_scratch, image_scratch])
        })?;

        for scratch in scratch {
            scratch.destroy(device);
        }

        Ok(Self {
            indices,
            vertices,
            meshlets,
            meshlet_data,
            materials,
            meshes,
            textures,
            memory,
        })
    }

    pub fn destroy(&self, device: &Device) {
        self.indices.destroy(device);
        self.vertices.destroy(device);
        self.meshlets.destroy(device);
        self.meshlet_data.destroy(device);
        self.materials.destroy(device);
        self.meshes.destroy(device);
        for texture in &self.textures {
            texture.destroy(device);
        }
        self.memory.free(device);
    }
}

fn texture_kind_format(kind: asset::TextureKind) -> vk::Format {
    match kind {
        asset::TextureKind::Albedo => vk::Format::BC1_RGBA_SRGB_BLOCK,
        asset::TextureKind::Normal => vk::Format::BC5_UNORM_BLOCK,
        asset::TextureKind::Specular => vk::Format::BC5_UNORM_BLOCK,
        asset::TextureKind::Emissive => vk::Format::BC1_RGB_SRGB_BLOCK,
    }
}
