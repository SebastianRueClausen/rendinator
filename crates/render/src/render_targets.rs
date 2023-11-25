use ash::vk;
use eyre::Result;

use crate::device::Device;
use crate::resources::{
    Allocator, Image, ImageRequest, ImageViewRequest, Memory,
};
use crate::swapchain::Swapchain;

pub(crate) struct RenderTargets {
    pub depth: Image,
    pub swapchain: Vec<Image>,
    memory: Memory,
}

impl RenderTargets {
    pub fn new(
        device: &Device,
        swapchain_images: Vec<Image>,
        swapchain: &Swapchain,
    ) -> Result<Self> {
        let extent = vk::Extent3D {
            width: swapchain.extent.width,
            height: swapchain.extent.height,
            depth: 1,
        };
        let mut depth = Image::new(
            device,
            &ImageRequest {
                format: DEPTH_FORMAT,
                mip_level_count: 1,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED,
                extent,
            },
        )?;
        let memory = Allocator::new(device)
            .alloc_image(&depth)
            .finish(vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        depth.add_view(device, ImageViewRequest::BASE)?;
        Ok(Self { depth, memory, swapchain: swapchain_images })
    }

    pub fn destroy(&self, device: &Device) {
        for image in &self.swapchain {
            image.destroy(device);
        }
        self.depth.destroy(device);
        self.memory.free(device);
    }
}

pub(crate) const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
