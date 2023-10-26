use std::collections::HashMap;
use std::{mem, slice};

use ash::vk::{self};
use command::CommandBuffer;
use descriptor::{Binding, DescriptorBuffer, LayoutBinding};
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

mod command;
mod descriptor;
mod device;
mod instance;
mod resources;
mod shader;
mod swapchain;

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
    memory: Memory,
}

impl Renderer {
    pub fn new(request: RendererRequest) -> Result<Self> {
        let instance = Instance::new(request.validate)?;
        let device = Device::new(&instance)?;

        let extent =
            vk::Extent2D { width: request.width, height: request.height };
        let swapchain =
            Swapchain::new(&instance, &device, request.window, extent)?;

        let scene = request.scene;

        let shaders = create_shaders(&device)?;
        let pipelines = create_pipelines(&device, scene, &shaders)?;

        let buffers = create_buffers(&device, scene)?;
        let images = create_images(&device, scene)?;

        let mut allocator = Allocator::new(&device);
        buffers.values().for_each(|buffer| allocator.alloc_buffer(buffer));
        images.values().for_each(|image| allocator.alloc_image(image));

        let memory = allocator.finish(vk::MemoryPropertyFlags::DEVICE_LOCAL)?;

        let (passes, descriptor_data) =
            create_passes(&device, &buffers, &images, &pipelines)?;
        let descriptor_buffer =
            DescriptorBuffer::new(&device, &descriptor_data.data)?;

        upload_data(&device, scene, &buffers, &images)?;

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

        let buffer_offset = pass.descriptor_buffer_offset;
        let descriptor_offset = 0;

        unsafe {
            self.device
                .descriptor_buffer_loader
                .cmd_set_descriptor_buffer_offsets(
                    **command_buffer,
                    pipeline.bind_point,
                    pipeline.pipeline_layout,
                    descriptor_offset,
                    slice::from_ref(&0),
                    slice::from_ref(&buffer_offset),
                );
            self.device.cmd_bind_pipeline(
                **command_buffer,
                pipeline.bind_point,
                **pipeline,
            );
        }
    }

    fn dispatch(&self, command_buffer: &CommandBuffer, x: u32, y: u32, z: u32) {
        unsafe { self.device.cmd_dispatch(**command_buffer, x, y, z) }
    }

    pub fn render_frame(&self) -> Result<()> {
        command::quickie(&self.device, |command_buffer| {
            self.bind_descriptor_buffer(command_buffer);
            self.start_pass(command_buffer, PassId::Test);
            self.dispatch(command_buffer, 1, 1, 1);

            Ok(())
        })
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        self.device.wait_until_idle();

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

        self.swapchain.destroy();
        self.device.destroy();
        self.instance.destroy();
    }
}

#[derive(Clone, Copy, Hash, PartialEq, Eq)]
enum ShaderId {
    Test,
    Vertex,
    Fragment,
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
}

fn create_images(
    device: &Device,
    scene: &asset::Scene,
) -> Result<HashMap<ImageId, Image>> {
    scene
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
                extent: vk::Extent3D {
                    width: texture.width,
                    height: texture.height,
                    depth: 1,
                },
                format: texture_kind_format(texture.kind),
                views: slice::from_ref(&view_request),
                mip_level_count,
                usage,
            };

            let image = Image::new(device, &request)?;
            Ok((ImageId::Texture { index }, image))
        })
        .collect()
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
    Draw,
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
                bindings: &[LayoutBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    count: 1,
                }],
            },
        )?,
    );

    pipelines.insert(
        PipelineId::Draw,
        Pipeline::compute(
            device,
            &shaders[&ShaderId::Test],
            &PipelineLayout {
                push_constant: None,
                bindings: &[LayoutBinding {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    count: 1,
                }],
            },
        )?,
    );

    Ok(pipelines)
}

struct DescriptorData {
    alignment: usize,
    data: Vec<u8>,
}

impl DescriptorData {
    fn new(device: &Device) -> Self {
        Self {
            data: Vec::default(),
            alignment: device
                .descriptor_buffer_properties
                .descriptor_buffer_offset_alignment
                as usize,
        }
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
    Test,
}

struct Pass {
    descriptor_buffer_offset: vk::DeviceSize,
    pipeline: PipelineId,
}

fn create_passes(
    device: &Device,
    buffers: &HashMap<BufferId, Buffer>,
    _images: &HashMap<ImageId, Image>,
    pipelines: &HashMap<PipelineId, Pipeline>,
) -> Result<(HashMap<PassId, Pass>, DescriptorData)> {
    let mut passes = HashMap::default();
    let mut descriptor_data = DescriptorData::new(device);

    let offset = descriptor_data.insert(descriptor::descriptor_data(
        device,
        &pipelines[&PipelineId::Test].descriptor_layout,
        [Binding::StorageBuffer(&[&buffers[&BufferId::Vertices]])],
    ));

    passes.insert(
        PassId::Test,
        Pass { descriptor_buffer_offset: offset, pipeline: PipelineId::Test },
    );

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
