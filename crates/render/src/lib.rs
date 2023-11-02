use std::collections::HashMap;
use std::{mem, slice};

use ash::vk::{self};
use command::{CommandBuffer, ImageBarrier};
use descriptor::{DescriptorBuffer, LayoutBinding};
use device::Device;
use eyre::Result;
use instance::Instance;
use raw_window_handle::RawWindowHandle;
use resources::{
    Allocator, Buffer, BufferKind, BufferRequest, BufferWrite, Image,
    ImageRequest, ImageViewRequest, ImageWrite, Memory,
};
use shader::{Pipeline, PipelineLayout, Shader, ShaderRequest};
use swapchain::Swapchain;
use sync::Sync;

mod command;
mod descriptor;
mod device;
mod instance;
mod resources;
mod shader;
mod swapchain;
mod sync;

pub struct RendererRequest<'a> {
    pub window: RawWindowHandle,
    pub width: u32,
    pub height: u32,
    pub validate: bool,
    pub scene: &'a asset::Scene,
}

pub struct Renderer {
    instance: Instance,
    device: Device,
    swapchain: Swapchain,
    shaders: HashMap<ShaderId, Shader>,
    buffers: HashMap<BufferId, Buffer>,
    images: HashMap<ImageId, Image>,
    pipelines: HashMap<PipelineId, Pipeline>,
    passes: HashMap<PassId, Pass>,
    descriptor_buffer: DescriptorBuffer,
    sync: Sync,
    memory: Memory,
}

impl Renderer {
    pub fn new(request: RendererRequest) -> Result<Self> {
        let instance = Instance::new(request.validate)?;
        let device = Device::new(&instance)?;
        let extent =
            vk::Extent2D { width: request.width, height: request.height };
        let (swapchain, swapchain_images) =
            Swapchain::new(&instance, &device, request.window, extent)?;
        let scene = request.scene;
        let shaders = create_shaders(&device)?;
        let pipelines = create_pipelines(&device, scene, &shaders)?;
        let buffers = create_buffers(&device, scene)?;
        let mut images = create_images(&device, scene)?;
        let mut allocator = Allocator::new(&device);
        buffers.values().for_each(|buffer| allocator.alloc_buffer(buffer));
        images.values().for_each(|image| allocator.alloc_image(image));
        let memory = allocator.finish(vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        let swapchain_images =
            swapchain_images.into_iter().enumerate().map(|(index, image)| {
                (ImageId::Swapchain { index: index as u32 }, image)
            });
        images.extend(swapchain_images);
        let (passes, descriptor_data) =
            create_passes(&device, &swapchain, &buffers, &images, &pipelines)?;
        let descriptor_buffer =
            DescriptorBuffer::new(&device, &descriptor_data.data)?;
        upload_data(&device, scene, &buffers, &images)?;
        let sync = Sync::new(&device)?;
        Ok(Self {
            descriptor_buffer,
            passes,
            instance,
            device,
            swapchain,
            shaders,
            pipelines,
            buffers,
            images,
            memory,
            sync,
        })
    }

    fn bind_descriptor_buffer(&self, command_buffer: &CommandBuffer) {
        let binding_info = vk::DescriptorBufferBindingInfoEXT::builder()
            .usage(vk::BufferUsageFlags::RESOURCE_DESCRIPTOR_BUFFER_EXT)
            .address(self.descriptor_buffer.address);
        unsafe {
            self.device.descriptor_buffer_loader.cmd_bind_descriptor_buffers(
                **command_buffer,
                slice::from_ref(&binding_info),
            );
        }
    }

    fn start_pass(&self, command_buffer: &CommandBuffer, pass: PassId) {
        let pass = &self.passes[&pass];
        let pipeline = &self.pipelines[&pass.pipeline];

        unsafe {
            let descriptor_offset = 0;
            self.device
                .descriptor_buffer_loader
                .cmd_set_descriptor_buffer_offsets(
                    **command_buffer,
                    pipeline.bind_point,
                    pipeline.pipeline_layout,
                    descriptor_offset,
                    slice::from_ref(&0),
                    slice::from_ref(&pass.descriptor_buffer_offset),
                );
            self.device.cmd_bind_pipeline(
                **command_buffer,
                pipeline.bind_point,
                **pipeline,
            );
        }
    }

    pub fn render_frame(&self) -> Result<()> {
        let swapchain_index = self.swapchain.image_index(&self.sync)?;
        let buffer =
            command::frame(&self.device, &self.sync, |command_buffer| {
                self.bind_descriptor_buffer(command_buffer);
                self.start_pass(
                    command_buffer,
                    PassId::Test { index: swapchain_index },
                );

                let swapchain_image = &self.images
                    [&ImageId::Swapchain { index: swapchain_index }];
                command_buffer.pipeline_barriers(
                    &self.device,
                    &[ImageBarrier {
                        image: swapchain_image,
                        new_layout: vk::ImageLayout::GENERAL,
                        src_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                        dst_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                        src_access: vk::AccessFlags2::NONE,
                        dst_access: vk::AccessFlags2::TRANSFER_WRITE,
                    }],
                );

                let width = swapchain_image.extent.width.div_ceil(32);
                let height = swapchain_image.extent.height.div_ceil(32);
                command_buffer.dispatch(&self.device, width, height, 1);

                command_buffer.pipeline_barriers(
                    &self.device,
                    &[ImageBarrier {
                        image: swapchain_image,
                        new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                        src_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                        dst_stage: vk::PipelineStageFlags2::ALL_COMMANDS,
                        src_access: vk::AccessFlags2::TRANSFER_WRITE,
                        dst_access: vk::AccessFlags2::NONE,
                    }],
                );

                Ok(())
            })?;
        self.swapchain.present(&self.device, &self.sync, swapchain_index)?;
        self.device.wait_until_idle()?;
        buffer.destroy(&self.device);
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if self.device.wait_until_idle().is_err() {
            return;
        }
        self.sync.destroy(&self.device);
        self.descriptor_buffer.destroy(&self.device);
        for pipeline in self.pipelines.values() {
            pipeline.destroy(&self.device);
        }
        for shader in self.shaders.values() {
            shader.destroy(&self.device);
        }
        for buffer in self.buffers.values() {
            buffer.destroy(&self.device);
        }
        for image in self.images.values() {
            image.destroy(&self.device);
        }
        self.memory.free(&self.device);
        self.swapchain.destroy(&self.device);
        self.device.destroy();
        self.instance.destroy();
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum ShaderId {
    Test,
}

fn create_shaders(device: &Device) -> Result<HashMap<ShaderId, Shader>> {
    let mut shaders = HashMap::default();
    shaders.insert(
        ShaderId::Test,
        Shader::new(
            device,
            &ShaderRequest {
                stage: vk::ShaderStageFlags::COMPUTE,
                source: vk_shader_macros::include_glsl!(
                    "src/shaders/test.glsl",
                    kind: comp,
                ),
            },
        )?,
    );
    Ok(shaders)
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum BufferId {
    Indices,
    Vertices,
    Meshlets,
    MeshletData,
    Materials,
    Meshes,
}

fn create_buffers(
    device: &Device,
    scene: &asset::Scene,
) -> Result<HashMap<BufferId, Buffer>> {
    let mut buffers = HashMap::default();
    buffers.insert(
        BufferId::Indices,
        Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.indices) as vk::DeviceSize,
                kind: BufferKind::Index,
            },
        )?,
    );
    buffers.insert(
        BufferId::Vertices,
        Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.vertices) as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?,
    );
    buffers.insert(
        BufferId::Meshlets,
        Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.meshlets) as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?,
    );
    buffers.insert(
        BufferId::MeshletData,
        Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.meshlet_data) as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?,
    );
    buffers.insert(
        BufferId::Meshes,
        Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.meshes) as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?,
    );
    buffers.insert(
        BufferId::Materials,
        Buffer::new(
            device,
            &BufferRequest {
                size: mem::size_of_val(&scene.materials) as vk::DeviceSize,
                kind: BufferKind::Storage,
            },
        )?,
    );
    Ok(buffers)
}

fn buffer_uploads<'a>(
    scene: &'a asset::Scene,
    buffers: &'a HashMap<BufferId, Buffer>,
) -> Vec<BufferWrite<'a>> {
    vec![
        BufferWrite {
            buffer: &buffers[&BufferId::Indices],
            data: bytemuck::cast_slice(&scene.indices),
        },
        BufferWrite {
            buffer: &buffers[&BufferId::Vertices],
            data: bytemuck::cast_slice(&scene.vertices),
        },
        BufferWrite {
            buffer: &buffers[&BufferId::Meshlets],
            data: bytemuck::cast_slice(&scene.meshlets),
        },
        BufferWrite {
            buffer: &buffers[&BufferId::MeshletData],
            data: bytemuck::cast_slice(&scene.meshlet_data),
        },
        BufferWrite {
            buffer: &buffers[&BufferId::Meshes],
            data: bytemuck::cast_slice(&scene.meshes),
        },
        BufferWrite {
            buffer: &buffers[&BufferId::Materials],
            data: bytemuck::cast_slice(&scene.materials),
        },
    ]
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum ImageId {
    Texture { index: usize },
    Swapchain { index: u32 },
}

fn create_images(
    device: &Device,
    scene: &asset::Scene,
) -> Result<HashMap<ImageId, Image>> {
    let images: HashMap<_, _> = scene
        .textures
        .iter()
        .enumerate()
        .map(|(index, texture)| {
            let mip_level_count = texture.mips.len() as u32;
            let usage = vk::ImageUsageFlags::TRANSFER_DST
                | vk::ImageUsageFlags::SAMPLED;
            let view_request =
                ImageViewRequest { mip_level_count, base_mip_level: 0 };
            let request = ImageRequest {
                format: texture_kind_format(texture.kind),
                views: slice::from_ref(&view_request),
                extent: vk::Extent3D {
                    width: texture.width,
                    height: texture.height,
                    depth: 1,
                },
                mip_level_count,
                usage,
            };
            let image = Image::new(device, &request)?;
            Ok((ImageId::Texture { index }, image))
        })
        .collect::<Result<_>>()?;
    Ok(images)
}

fn image_uploads<'a>(
    scene: &'a asset::Scene,
    images: &'a HashMap<ImageId, Image>,
) -> Vec<ImageWrite<'a>> {
    scene
        .textures
        .iter()
        .enumerate()
        .map(|(index, texture)| {
            let id = ImageId::Texture { index };
            let image = &images[&id];
            ImageWrite { extent: image.extent, mips: &texture.mips, image }
        })
        .collect()
}

fn upload_data(
    device: &Device,
    scene: &asset::Scene,
    buffers: &HashMap<BufferId, Buffer>,
    images: &HashMap<ImageId, Image>,
) -> Result<()> {
    let buffer_uploads = buffer_uploads(scene, buffers);
    let image_uploads = image_uploads(scene, images);
    let scratch = command::quickie(device, |command_buffer| {
        let buffer_scratch = resources::upload_buffer_data(
            device,
            &command_buffer,
            &buffer_uploads,
        )?;
        let image_scratch = resources::upload_image_data(
            device,
            &command_buffer,
            &image_uploads,
        )?;
        Ok([buffer_scratch, image_scratch])
    })?;
    for scratch in scratch {
        scratch.destroy(device);
    }
    Ok(())
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum PipelineId {
    Test,
}

fn create_pipelines(
    device: &Device,
    _scene: &asset::Scene,
    shaders: &HashMap<ShaderId, Shader>,
) -> Result<HashMap<PipelineId, Pipeline>> {
    let mut pipelines = HashMap::default();
    pipelines.insert(
        PipelineId::Test,
        Pipeline::compute(
            device,
            &shaders[&ShaderId::Test],
            &PipelineLayout {
                push_constant: None,
                bindings: &[
                    LayoutBinding {
                        ty: vk::DescriptorType::STORAGE_IMAGE,
                        count: 1,
                    },
                    LayoutBinding {
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        count: 1,
                    },
                ],
            },
        )?,
    );
    Ok(pipelines)
}

#[derive(Debug)]
struct DescriptorData {
    alignment: usize,
    data: Vec<u8>,
}

impl DescriptorData {
    fn new(device: &Device) -> Self {
        let alignment = device
            .descriptor_buffer_properties
            .descriptor_buffer_offset_alignment;
        Self { data: Vec::default(), alignment: alignment as usize }
    }

    fn insert(&mut self, mut data: Vec<u8>) -> vk::DeviceSize {
        while self.data.len() % self.alignment != 0 {
            self.data.push(0);
        }
        let offset = self.data.len() as vk::DeviceSize;
        self.data.append(&mut data);
        offset
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum PassId {
    Test { index: u32 },
}

struct Pass {
    descriptor_buffer_offset: vk::DeviceSize,
    pipeline: PipelineId,
}

fn create_passes(
    device: &Device,
    swapchain: &Swapchain,
    buffers: &HashMap<BufferId, Buffer>,
    images: &HashMap<ImageId, Image>,
    pipelines: &HashMap<PipelineId, Pipeline>,
) -> Result<(HashMap<PassId, Pass>, DescriptorData)> {
    let mut passes = HashMap::default();
    let mut descriptor_data = DescriptorData::new(device);
    let test_passes = (0..swapchain.image_count).map(|index| {
        let pass = Pass {
            descriptor_buffer_offset: {
                let layout = &pipelines[&PipelineId::Test].descriptor_layout;
                let data = descriptor::Builder::new(device, layout)
                    .storage_image(
                        &images[&ImageId::Swapchain { index }]
                            .view(&ImageViewRequest::BASE),
                    )
                    .storage_buffer(&buffers[&BufferId::MeshletData])
                    .finish();
                descriptor_data.insert(data)
            },
            pipeline: PipelineId::Test,
        };
        (PassId::Test { index }, pass)
    });
    passes.extend(test_passes);
    Ok((passes, descriptor_data))
}

fn texture_kind_format(kind: asset::TextureKind) -> vk::Format {
    match kind {
        asset::TextureKind::Albedo => vk::Format::BC1_RGBA_SRGB_BLOCK,
        asset::TextureKind::Normal => vk::Format::BC5_UNORM_BLOCK,
        asset::TextureKind::Specular => vk::Format::BC5_UNORM_BLOCK,
        asset::TextureKind::Emissive => vk::Format::BC1_RGB_SRGB_BLOCK,
    }
}
