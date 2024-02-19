use ash::vk;
use eyre::Result;

use crate::hal;

pub(crate) struct RenderTargets {
    pub depth: hal::Image,
    pub color_buffer: hal::Image,
    pub swapchain: Vec<hal::Image>,
    pub visibility: hal::Image,
    // r11g11 is normal vector and b10 unused.
    pub gbuffer0: hal::Image,
    // r8g8b8 is albedo color and a8 is metallic.
    pub gbuffer1: hal::Image,
    // r8g8b8 is emissive color and a8 is roughness.
    pub gbuffer2: hal::Image,
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
        let mut visibility = hal::Image::new(
            device,
            &hal::ImageRequest {
                // Note: This can further compressed to R32.
                format: vk::Format::R32G32_UINT,
                mip_level_count: 1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED,
                extent,
            },
        )?;
        let mut gbuffer0 = hal::Image::new(
            device,
            &hal::ImageRequest {
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                mip_level_count: 1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED,
                extent,
            },
        )?;
        let mut gbuffer1 = hal::Image::new(
            device,
            &hal::ImageRequest {
                format: vk::Format::R8G8B8A8_UNORM,
                mip_level_count: 1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED,
                extent,
            },
        )?;
        let mut gbuffer2 = hal::Image::new(
            device,
            &hal::ImageRequest {
                format: vk::Format::R8G8B8A8_UNORM,
                mip_level_count: 1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED,
                extent,
            },
        )?;
        let mut color_buffer = hal::Image::new(
            device,
            &hal::ImageRequest {
                format: vk::Format::B10G11R11_UFLOAT_PACK32,
                mip_level_count: 1,
                usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED,
                extent,
            },
        )?;
        let memory = hal::Allocator::new(device)
            .alloc_image(&depth)
            .alloc_image(&visibility)
            .alloc_image(&gbuffer0)
            .alloc_image(&gbuffer1)
            .alloc_image(&gbuffer2)
            .alloc_image(&color_buffer)
            .finish(vk::MemoryPropertyFlags::DEVICE_LOCAL)?;
        depth.add_view(device, hal::ImageViewRequest::BASE)?;
        visibility.add_view(device, hal::ImageViewRequest::BASE)?;
        for gbuffer in [&mut gbuffer0, &mut gbuffer1, &mut gbuffer2] {
            gbuffer.add_view(device, hal::ImageViewRequest::BASE)?;
        }
        color_buffer.add_view(device, hal::ImageViewRequest::BASE)?;
        Ok(Self {
            depth,
            memory,
            swapchain: swapchain_images,
            visibility,
            gbuffer0,
            gbuffer1,
            gbuffer2,
            color_buffer,
        })
    }

    pub fn destroy(&self, device: &hal::Device) {
        for image in &self.swapchain {
            image.destroy(device);
        }
        for gbuffer in [&self.gbuffer0, &self.gbuffer1, &self.gbuffer2] {
            gbuffer.destroy(device);
        }
        self.depth.destroy(device);
        self.visibility.destroy(device);
        self.color_buffer.destroy(device);
        self.memory.free(device);
    }
}

pub(crate) const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;
