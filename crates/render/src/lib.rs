use ash::vk::{self};
pub use camera::{Camera, CameraMove};
use constants::Constants;
use eyre::Result;
use glam::Vec2;
#[cfg(feature = "gui")]
use gui::Gui;
#[cfg(feature = "gui")]
pub use gui::GuiRequest;
use mesh::MeshPhase;
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};
use render_targets::RenderTargets;
use scene::{NodeTree, Scene};

mod camera;
mod constants;
mod debug;
mod hal;
mod mesh;
mod render_targets;
pub mod scene;

#[cfg(feature = "gui")]
mod gui;

pub struct RendererRequest<'a> {
    pub window: RawWindowHandle,
    pub display: RawDisplayHandle,
    pub width: u32,
    pub height: u32,
    pub validate: bool,
    pub scene: &'a asset::Scene,
}

pub struct FrameRequest {
    #[cfg(feature = "gui")]
    pub gui: gui::GuiRequest,
    pub camera_move: CameraMove,
}

pub struct Renderer {
    instance: hal::Instance,
    device: hal::Device,
    swapchain: hal::Swapchain,
    sync: hal::Sync,
    scene: Scene,
    constants: Constants,
    mesh_phase: MeshPhase,
    render_targets: RenderTargets,
    #[cfg(feature = "gui")]
    gui: Gui,
    camera: Camera,
}

impl Renderer {
    pub fn new(request: RendererRequest) -> Result<Self> {
        let instance = hal::Instance::new(request.validate)?;
        let device = hal::Device::new(&instance)?;
        let sync = hal::Sync::new(&device)?;
        let extent =
            vk::Extent2D { width: request.width, height: request.height };
        let (swapchain, swapchain_images) = hal::Swapchain::new(
            &instance,
            &device,
            request.window,
            request.display,
            extent,
        )?;
        let scene = request.scene;
        let scene = Scene::new(&device, scene)?;
        #[cfg(feature = "gui")]
        let gui = Gui::new(&device, &swapchain)?;
        let mesh_phase = MeshPhase::new(&device, &swapchain)?;
        let render_targets =
            RenderTargets::new(&device, swapchain_images, &swapchain)?;
        let camera = Camera::new(Vec2 {
            x: swapchain.extent.width as f32,
            y: swapchain.extent.height as f32,
        });
        let constants = Constants::new(&device, &swapchain, &camera)?;
        Ok(Self {
            instance,
            device,
            swapchain,
            sync,
            constants,
            mesh_phase,
            render_targets,
            scene,
            #[cfg(feature = "gui")]
            gui,
            camera,
        })
    }

    fn create_descriptors(
        &self,
    ) -> Result<(Descriptors, hal::DescriptorBuffer)> {
        let mut descriptor_data = hal::DescriptorData::new(&self.device);
        let passes = Descriptors {
            #[cfg(feature = "gui")]
            gui: gui::create_descriptor(
                &self.device,
                &self.gui,
                &self.constants,
                &mut descriptor_data,
            ),
            mesh_phase: mesh::create_descriptor(
                &self.device,
                &self.mesh_phase,
                &self.constants,
                &self.scene,
                &mut descriptor_data,
            ),
            depth_reduce: mesh::create_depth_reduce_descriptors(
                &self.device,
                &self.mesh_phase,
                &self.render_targets,
                &mut descriptor_data,
            ),
            cull: mesh::create_cull_descriptor(
                &self.device,
                &self.mesh_phase,
                &self.constants,
                &self.scene,
                &mut descriptor_data,
            ),
        };
        let descriptor_buffer =
            hal::DescriptorBuffer::new(&self.device, &descriptor_data)?;
        Ok((passes, descriptor_buffer))
    }

    pub fn render_frame(&mut self, request: &FrameRequest) -> Result<()> {
        self.camera.move_by(request.camera_move);
        let swapchain_index = self.swapchain.image_index(&self.sync)?;

        let mut update = Update::empty();
        #[cfg(feature = "gui")]
        let gui_update =
            self.gui.update(&self.device, &self.swapchain, &request.gui)?;
        let scene_update = self.scene.update();

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

        buffer_writes.extend(scene::buffer_writes(&self.scene, &scene_update));
        buffer_writes.push(self.constants.buffer_write());

        self.scene.update_tlas(&self.device)?;

        let (descriptors, descriptor_buffer) = self.create_descriptors()?;

        let (buffer, scratchs) =
            hal::command::frame(&self.device, &self.sync, |command_buffer| {
                let swapchain_image =
                    &self.render_targets.swapchain[swapchain_index as usize];

                let buffer_scratch = hal::upload_buffer_data(
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
                        hal::ImageLayouts {
                            layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            src: hal::Access::NONE,
                            dst: hal::Access::TRANSFER_DST,
                        },
                        images,
                    );
                }

                let image_scratch = hal::upload_image_data(
                    &self.device,
                    command_buffer,
                    &image_writes,
                )?;

                command_buffer
                    .bind_descriptor_buffer(&self.device, &descriptor_buffer);

                mesh::render(
                    &self.device,
                    command_buffer,
                    swapchain_image,
                    &descriptors,
                    &self.mesh_phase,
                    &self.render_targets,
                    &self.scene,
                );

                #[cfg(feature = "gui")]
                {
                    let images =
                        self.gui.textures.iter().map(|texture| &texture.image);
                    command_buffer.ensure_image_layouts(
                        &self.device,
                        hal::ImageLayouts {
                            layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                            src: hal::Access::TRANSFER_DST,
                            dst: hal::Access::NONE,
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

                command_buffer.pipeline_barriers(
                    &self.device,
                    &[hal::ImageBarrier {
                        image: swapchain_image,
                        new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                        mip_levels: hal::MipLevels::All,
                        src: hal::Access::ALL,
                        dst: hal::Access::NONE,
                    }],
                    &[],
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

    pub fn node_tree_mut(&mut self) -> &mut NodeTree {
        &mut self.scene.node_tree
    }

    pub fn change_scene(&mut self, scene: &asset::Scene) -> Result<()> {
        self.device.wait_until_idle()?;
        self.scene.destroy(&self.device);
        self.scene = Scene::new(&self.device, &scene)?;
        Ok(())
    }

    pub fn move_camera(&mut self, camera_move: CameraMove) {
        self.camera.move_by(camera_move);
    }

    pub fn camera_mut(&mut self) -> &mut Camera {
        &mut self.camera
    }

    fn update(&mut self, update: Update) -> Result<()> {
        self.constants.update(&self.swapchain, &self.camera);
        if update.contains(Update::RECREATE_DESCRIPTORS) {}
        Ok(())
    }
}

impl Drop for Renderer {
    fn drop(&mut self) {
        if self.device.wait_until_idle().is_err() {
            return;
        }
        self.render_targets.destroy(&self.device);
        self.mesh_phase.destroy(&self.device);
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

struct Descriptors {
    #[cfg(feature = "gui")]
    gui: hal::Descriptor,
    mesh_phase: hal::Descriptor,
    depth_reduce: Vec<hal::Descriptor>,
    cull: hal::Descriptor,
}
