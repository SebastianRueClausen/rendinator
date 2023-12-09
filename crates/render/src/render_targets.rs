use ash::vk;
use eyre::Result;

use crate::hal;

pub(crate) struct RenderTargets {
    pub depth: hal::Image,
    pub swapchain: Vec<hal::Image>,
    memory: hal::Memory,
}

impl RenderTargets {
    pub fn new(
        device: &hal::Device,
        swapchain_images: Vec<hal::Image>,
        swapchain: &hal::Swapchain,
    ) -> Result<Self> {
        let extent = vk::Extent3D {
            width: swapchain.extent.width,
            height: swapchain.extent.height,
            depth: 1,
        };
        let mut depth = hal::Image::new(
            device,
            &hal::ImageRequest {
                format: DEPTH_FORMAT,
                mip_level_count: 1,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED,
                extent,
            },
        )?;
        let memory = hal::Allocator::new(device)
            .alloc_image(&depth)
            .finish(vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        depth.add_view(device, hal::ImageViewRequest::BASE)?;
        Ok(Self { depth, memory, swapchain: swapchain_images })
    }

    pub fn destroy(&self, device: &hal::Device) {
        for image in &self.swapchain {
            image.destroy(device);
        }
        self.depth.destroy(device);
        self.memory.free(device);
    }
}

pub(crate) const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
