use ash::{extensions::khr, vk};
use std::cell::{Cell, RefCell};

use crate::{Device, Surface};

pub struct Swapchain {
    loader: khr::Swapchain,

    present_mode: vk::PresentModeKHR,
    surface_format: vk::SurfaceFormatKHR,

    handle: Cell<vk::SwapchainKHR>,
    extent: Cell<vk::Extent2D>,

    images: RefCell<Vec<Res<ImageView>>>,

    device: Rc<Device>,
    surface: Rc<Surface>,

    graphics_queue: Res<Queue>,
}

enum NextSwapchainImage {
    UpToDate { image_index: u32 },
    OutOfDate,
}

impl Swapchain {
    /// Create a new swapchain. `extent` is used to determine the size of the swapchain images only
    /// if it aren't able to determine it from `surface`.
    pub fn new(
        device: Rc<Device>,
        pool: &ResourcePool,
        surface: Rc<Surface>,
        graphics_queue: Res<Queue>,
        extent: vk::Extent2D,
    ) -> Result<Self> {
        let (surface_formats, present_modes, surface_caps) = unsafe {
            let format = surface
                .loader
                .get_physical_device_surface_formats(device.physical.handle, surface.handle)?;
            let modes = surface.loader.get_physical_device_surface_present_modes(
                device.physical.handle,
                surface.handle,
            )?;
            let caps = surface
                .loader
                .get_physical_device_surface_capabilities(device.physical.handle, surface.handle)?;
            (format, modes, caps)
        };

        let queue_families = [graphics_queue.family_index];
        let min_image_count = 2.max(surface_caps.min_image_count);

        let surface_format = surface_formats
            .iter()
            .find(|format| {
                format.format == vk::Format::B8G8R8A8_SRGB
                    && format.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            })
            .or_else(|| surface_formats.first())
            .ok_or_else(|| anyhow!("can't find valid surface format"))?
            .clone();

        let extent = if surface_caps.current_extent.width != u32::MAX {
            surface_caps.current_extent
        } else {
            vk::Extent2D {
                width: extent.width.clamp(
                    surface_caps.min_image_extent.width,
                    surface_caps.max_image_extent.width,
                ),
                height: extent.height.clamp(
                    surface_caps.min_image_extent.height,
                    surface_caps.max_image_extent.height,
                ),
            }
        };

        let preferred_present_mode = vk::PresentModeKHR::FIFO;

        let present_mode = present_modes
            .iter()
            .any(|mode| *mode == preferred_present_mode)
            .then_some(preferred_present_mode)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface.handle)
            .min_image_count(min_image_count)
            .image_format(surface_format.format)
            .image_color_space(surface_format.color_space)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(present_mode)
            .image_extent(extent)
            .image_array_layers(1);
        let loader = khr::Swapchain::new(&device.instance.handle, &device.handle);

        let handle = unsafe { loader.create_swapchain(&swapchain_info, None)? };
        let images = unsafe { loader.get_swapchain_images(handle)? };

        trace!("using {} swap chain images", images.len());

        let images: Result<Vec<_>> = images
            .into_iter()
            .map(|handle| {
                let image = pool.create_image(
                    MemoryLocation::Gpu,
                    &ImageInfo {
                        usage: vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        aspect_flags: vk::ImageAspectFlags::COLOR,
                        kind: ImageKind::Swapchain { handle },
                        format: surface_format.format,
                        mip_levels: 1,
                        extent: vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        },
                    },
                )?;

                pool.create_image_view(&ImageViewInfo {
                    view_type: vk::ImageViewType::TYPE_2D,
                    image: image.clone(),
                    mips: 0..1,
                })
            })
            .collect();

        Ok(Self {
            handle: Cell::new(handle),
            extent: Cell::new(extent),
            images: RefCell::new(images?),
            device: device.clone(),
            graphics_queue,
            surface_format,
            present_mode,
            surface,
            loader,
        })
    }

    /// Recreate swapchain from `self` to a new `extent`.
    ///
    /// `extent` must be valid here unlike in `Self::new`, otherwise it could end in and endless
    /// cycle if recreating the swapchain, if for some reason the surface continues to give us and
    /// invalid extent.
    pub fn recreate(&self, pool: &ResourcePool, extent: vk::Extent2D) -> Result<()> {
        if extent.width == u32::MAX {
            return Err(anyhow!("`extent` must be valid when recreating swapchain"));
        }

        let surface_caps = unsafe {
            self.surface
                .loader
                .get_physical_device_surface_capabilities(
                    self.device.physical.handle,
                    self.surface.handle,
                )?
        };

        let queue_families = [self.graphics_queue.family_index];
        let min_image_count = (FRAMES_IN_FLIGHT as u32).max(surface_caps.min_image_count);

        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .old_swapchain(self.handle.get())
            .surface(self.surface.handle)
            .min_image_count(min_image_count)
            .image_format(self.surface_format.format)
            .image_color_space(self.surface_format.color_space)
            .image_usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::COLOR_ATTACHMENT)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .queue_family_indices(&queue_families)
            .pre_transform(surface_caps.current_transform)
            .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(self.present_mode)
            .image_extent(extent)
            .image_array_layers(1);

        let new = unsafe { self.loader.create_swapchain(&swapchain_info, None)? };

        unsafe {
            self.loader.destroy_swapchain(self.handle.get(), None);
            self.images.borrow_mut().clear();
        }

        self.handle.set(new);
        self.extent.set(extent);

        let images = unsafe { self.loader.get_swapchain_images(self.handle.get())? };
        let images: Result<Vec<_>> = images
            .into_iter()
            .map(|handle| {
                let image = pool.create_image(
                    MemoryLocation::Gpu,
                    &ImageInfo {
                        usage: vk::ImageUsageFlags::TRANSFER_DST
                            | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                        aspect_flags: vk::ImageAspectFlags::COLOR,
                        kind: ImageKind::Swapchain { handle },
                        format: self.surface_format.format,
                        mip_levels: 1,
                        extent: vk::Extent3D {
                            width: extent.width,
                            height: extent.height,
                            depth: 1,
                        },
                    },
                )?;

                pool.create_image_view(&ImageViewInfo {
                    view_type: vk::ImageViewType::TYPE_2D,
                    image: image.clone(),
                    mips: 0..1,
                })
            })
            .collect();

        *self.images.borrow_mut() = images?;

        Ok(())
    }

    pub fn image(&self, index: u32) -> Res<ImageView> {
        self.images.borrow()[index as usize].clone()
    }

    fn get_next_image(&self, frame: &Frame) -> Result<NextSwapchainImage> {
        let next_image = unsafe {
            self.loader.acquire_next_image(
                self.handle.get(),
                u64::MAX,
                frame.presented,
                vk::Fence::null(),
            )
        };

        match next_image {
            // The image is up to date.
            Ok((image_index, false)) => Ok(NextSwapchainImage::UpToDate { image_index }),

            // The image is suboptimal or unavailable.
            Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                Ok(NextSwapchainImage::OutOfDate)
            }

            Err(result) => Err(result.into()),
        }
    }

    pub fn viewports(&self) -> [vk::Viewport; 1] {
        [vk::Viewport {
            width: self.extent().width as f32,
            height: self.extent().height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
            x: 0.0,
            y: 0.0,
        }]
    }

    pub fn scissors(&self) -> [vk::Rect2D; 1] {
        [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent(),
        }]
    }

    pub fn size(&self) -> Vec2 {
        Vec2 {
            x: self.extent().width as f32,
            y: self.extent().height as f32,
        }
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent.get()
    }

    pub fn extent_3d(&self) -> vk::Extent3D {
        vk::Extent3D {
            width: self.extent().width,
            height: self.extent().height,
            depth: 1,
        }
    }

    pub fn format(&self) -> vk::Format {
        self.surface_format.format
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            self.loader.destroy_swapchain(self.handle.get(), None);
        }
    }
}
