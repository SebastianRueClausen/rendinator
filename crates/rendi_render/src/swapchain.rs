use crate::{surface, Device, PresentMode, Queue, RenderError, Surface, SurfaceExtent};
use ash::{extensions::khr, vk};
use rendi_res::Res;
use smallvec::SmallVec;

pub struct SwapchainInfo<'a> {
    pub device: Res<Device>,
    pub surface: Res<Surface>,
    pub queues: &'a [Queue],
    pub extent: SurfaceExtent,
    pub preferred_present_mode: PresentMode,
}

pub struct Swapchain {
    pub(crate) handle: vk::SwapchainKHR,
    loader: khr::Swapchain,
    extent: SurfaceExtent,
    present_mode: PresentMode,
}

impl Swapchain {
    pub fn new(info: SwapchainInfo) -> Result<Self, RenderError> {
        let device = info.device;
        let surface = info.surface;
        let surface_formats = surface::surface_formats(device.physical(), &surface)?;
        let present_modes = surface::present_modes(device.physical(), &surface)?;
        let surface_capabilities = surface::surface_capabilities(device.physical(), &surface)?;
        let queue_families: SmallVec<[_; 4]> = info
            .queues
            .iter()
            .map(|queue| queue.family_index())
            .collect();
        let min_image_count = 2_u32.min(surface_capabilities.min_image_count);
        let surface_format = surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| surface_formats.first())
            .cloned()
            .ok_or(RenderError::SurfaceUnsupported)?;
        let extent = (surface_capabilities.current_extent.width != u32::MAX)
            .then_some(surface_capabilities.current_extent)
            .unwrap_or_else(|| vk::Extent2D {
                width: info.extent.width.clamp(
                    surface_capabilities.min_image_extent.width,
                    surface_capabilities.max_image_extent.width,
                ),
                height: info.extent.height.clamp(
                    surface_capabilities.min_image_extent.height,
                    surface_capabilities.max_image_extent.height,
                ),
            });
        let present_mode = present_modes
            .into_iter()
            .any(|mode| mode == info.preferred_present_mode.into())
            .then_some(info.preferred_present_mode)
            .unwrap_or(PresentMode::Fifo);
        let image_usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
        let sharing_mode = if queue_families.len() > 1 {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };
        let create_info = vk::SwapchainCreateInfoKHR::builder()
            .pre_transform(surface_capabilities.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode.into())
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_usage(image_usage)
            .image_sharing_mode(sharing_mode)
            .surface(surface.handle)
            .image_extent(extent)
            .image_array_layers(1);
        let loader = khr::Swapchain::new(&device.instance().handle(), &device.handle());
        let handle = unsafe { loader.create_swapchain(&create_info, None)? };
        Ok(Self {
            present_mode,
            extent: extent.into(),
            loader,
            handle,
        })
    }

    pub fn present_mode(&self) -> PresentMode {
        self.present_mode
    }

    pub fn extent(&self) -> SurfaceExtent {
        self.extent
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.handle, None);
        }
    }
}
