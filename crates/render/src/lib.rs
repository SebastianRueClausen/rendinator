use ash::vk::{self, ImageBlit};
use command::ImageBarrier;
use constants::Constants;
use descriptor::{
    Descriptor, DescriptorBuffer, DescriptorBuilder, DescriptorData,
    DescriptorLayout, DescriptorLayoutBuilder,
};
use device::Device;
use eyre::Result;
#[cfg(feature = "gui")]
use gui::Gui;
#[cfg(feature = "gui")]
pub use gui::GuiRequest;
use instance::Instance;
use raw_window_handle::RawWindowHandle;
use resources::{Image, ImageViewRequest};
use scene::Scene;
use shader::{Pipeline, PipelineLayout, Shader, ShaderRequest};
use swapchain::Swapchain;
use sync::Sync;

use crate::command::Access;

mod command;
mod constants;
mod debug;
mod descriptor;
mod device;
mod instance;
mod resources;
mod scene;
mod shader;
mod swapchain;
mod sync;

#[cfg(feature = "gui")]
mod gui;

pub struct RendererRequest<'a> {
    pub window: RawWindowHandle,
    pub width: u32,
    pub height: u32,
    pub validate: bool,
    pub scene: &'a asset::Scene,
}

#[cfg(feature = "gui")]
pub struct FrameRequest<'a> {
    pub gui: gui::GuiRequest<'a>,
}

#[cfg(not(feature = "gui"))]
pub struct FrameRequest {}

pub struct Renderer {
    instance: Instance,
    device: Device,
    swapchain: Swapchain,
    sync: Sync,
    swapchain_images: Vec<Image>,
    test_descriptor_layout: DescriptorLayout,
    test_pipeline: Pipeline,
    scene: Scene,
    constants: Constants,
    #[cfg(feature = "gui")]
    gui: Gui,
}

impl Renderer {
    pub fn new(request: RendererRequest) -> Result<Self> {
        let instance = Instance::new(request.validate)?;
        let device = Device::new(&instance)?;
        let sync = Sync::new(&device)?;
        let extent =
            vk::Extent2D { width: request.width, height: request.height };
        let (swapchain, swapchain_images) =
            Swapchain::new(&instance, &device, request.window, extent)?;
        let scene = request.scene;
        let (test_descriptor_layout, test_pipeline) =
            create_test_pipeline(&device)?;
        let scene = Scene::new(&device, scene)?;
        #[cfg(feature = "gui")]
        let gui = Gui::new(&device, &swapchain)?;
        let constants = Constants::new(&device, &swapchain)?;
        Ok(Self {
            instance,
            device,
            swapchain,
            sync,
            swapchain_images,
            constants,
            test_pipeline,
            test_descriptor_layout,
            scene,
            gui,
        })
    }

    fn create_descriptors(&self) -> Result<(Descriptors, DescriptorBuffer)> {
        let mut descriptor_data = DescriptorData::new(&self.device);
        let passes = Descriptors {
            test: create_test_passes(
                &self.device,
                &self.swapchain_images,
                &self.test_descriptor_layout,
                &mut descriptor_data,
            )?,
            #[cfg(feature = "gui")]
            gui: gui::create_descriptor(
                &self.device,
                &self.gui,
                &self.constants,
                &mut descriptor_data,
            )?,
        };
        let descriptor_buffer =
            DescriptorBuffer::new(&self.device, &descriptor_data)?;
        Ok((passes, descriptor_buffer))
    }

    pub fn render_frame(&mut self, request: &FrameRequest) -> Result<()> {
        let swapchain_index = self.swapchain.image_index(&self.sync)?;

        let mut update = Update::empty();
        #[cfg(feature = "gui")]
        let gui_update =
            self.gui.update(&self.device, &self.swapchain, &request.gui)?;

        #[cfg(feature = "gui")]
        {
            update |= gui_update.update;
        }
        self.update(update)?;

        let (mut image_writes, mut buffer_writes) =
            (Vec::default(), Vec::default());

        #[cfg(feature = "gui")]
        {
            image_writes.extend(gui::image_writes(&self.gui, &gui_update));
            buffer_writes.extend(gui::buffer_writes(&self.gui, &gui_update));
        }

        buffer_writes.push(self.constants.buffer_write());
        let (descriptors, descriptor_buffer) = self.create_descriptors()?;

        let (buffer, scratchs) =
            command::frame(&self.device, &self.sync, |command_buffer| {
                let swapchain_image =
                    &self.swapchain_images[swapchain_index as usize];

                let buffer_scratch = resources::upload_buffer_data(
                    &self.device,
                    command_buffer,
                    &buffer_writes,
                )?;

                #[cfg(feature = "gui")]
                {
                    let images =
                        self.gui.textures.iter().map(|texture| &texture.image);
                    command_buffer.ensure_image_layouts(
                        &self.device,
                        command::ImageLayouts {
                            layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            src: Access::NONE,
                            dst: Access::TRANSFER_DST,
                        },
                        images,
                    );
                }

                let image_scratch = resources::upload_image_data(
                    &self.device,
                    command_buffer,
                    &image_writes,
                )?;

                let swapchain_access = Access {
                    stage: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
                    access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
                };

                command_buffer
                    .pipeline_barriers(
                        &self.device,
                        &[ImageBarrier {
                            image: swapchain_image,
                            new_layout:
                                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                            src: Access::NONE,
                            dst: swapchain_access,
                        }],
                    )
                    .bind_descriptor_buffer(&self.device, &descriptor_buffer);

                #[cfg(feature = "gui")]
                {
                    let images =
                        self.gui.textures.iter().map(|texture| &texture.image);
                    command_buffer.ensure_image_layouts(
                        &self.device,
                        command::ImageLayouts {
                            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            src: Access::TRANSFER_DST,
                            dst: Access::NONE,
                        },
                        images,
                    );
                    gui::render(
                        &self.device,
                        command_buffer,
                        swapchain_image,
                        &descriptors,
                        &gui_update,
                        &self.gui,
                    );
                }

                /*
                debug::display_texture(
                    &self.device,
                    command_buffer,
                    swapchain_image,
                    &self.gui.textures[0].image,
                );
                */

                command_buffer.pipeline_barriers(
                    &self.device,
                    &[ImageBarrier {
                        image: swapchain_image,
                        new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                        src: Access::ALL,
                        dst: Access::NONE,
                    }],
                );

                Ok([buffer_scratch, image_scratch])
            })?;

        self.swapchain.present(&self.device, &self.sync, swapchain_index)?;
        self.device.wait_until_idle()?;

        buffer.destroy(&self.device);
        descriptor_buffer.destroy(&self.device);

        for scratch in scratchs {
            scratch.destroy(&self.device);
        }

        Ok(())
    }

    fn update(&mut self, update: Update) -> Result<()> {
        self.constants.update(&self.swapchain);
        if update.contains(Update::RECREATE_DESCRIPTORS) {}
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if self.device.wait_until_idle().is_err() {
            return;
        }
        self.test_descriptor_layout.destroy(&self.device);
        self.test_pipeline.destroy(&self.device);
        for image in &self.swapchain_images {
            image.destroy(&self.device);
        }
        #[cfg(feature = "gui")]
        {
            self.gui.destroy(&self.device);
        }
        self.constants.destroy(&self.device);
        self.scene.destroy(&self.device);
        self.sync.destroy(&self.device);
        self.swapchain.destroy(&self.device);
        self.device.destroy();
        self.instance.destroy();
    }
}

bitflags::bitflags! {
    #[derive(Debug, Default, Clone, Copy)]
    pub struct Update: u32 {
        const RECREATE_DESCRIPTORS = 1;
    }
}

fn create_test_pipeline(
    device: &Device,
) -> Result<(DescriptorLayout, Pipeline)> {
    let shader = Shader::new(
        device,
        &ShaderRequest {
            stage: vk::ShaderStageFlags::COMPUTE,
            source: vk_shader_macros::include_glsl!(
                "src/shaders/test.glsl",
                kind: comp,
            ),
        },
    )?;
    let descriptor_layout = DescriptorLayoutBuilder::default()
        .binding(vk::DescriptorType::STORAGE_IMAGE)
        .build(device)?;
    let pipeline_layout = PipelineLayout {
        descriptors: &[&descriptor_layout],
        push_constant: None,
    };
    let pipeline = Pipeline::compute(device, &shader, &pipeline_layout)?;
    shader.destroy(device);
    Ok((descriptor_layout, pipeline))
}

fn create_test_passes(
    device: &Device,
    swapchain_images: &[Image],
    layout: &DescriptorLayout,
    data: &mut DescriptorData,
) -> Result<Vec<Descriptor>> {
    swapchain_images
        .iter()
        .map(|image| {
            let set = DescriptorBuilder::new(device, layout, data)
                .storage_image(image.view(&ImageViewRequest::BASE))
                .set();
            Ok(set)
        })
        .collect()
}

struct Descriptors {
    test: Vec<Descriptor>,
    #[cfg(feature = "gui")]
    gui: Descriptor,
}
