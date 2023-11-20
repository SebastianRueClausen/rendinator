use std::slice;

use ash::extensions::khr;
use ash::vk;
use eyre::{Context, Result};
use raw_window_handle::RawWindowHandle;

use crate::device::Device;
use crate::instance::Instance;
use crate::resources::{Image, ImageView, ImageViewRequest};
use crate::sync::Sync;

pub(crate) struct Swapchain {
    surface_loader: khr::Surface,
    swapchain_loader: khr::Swapchain,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    pub(super) fn new(
        instance: &Instance,
        device: &Device,
        window: RawWindowHandle,
        extent: vk::Extent2D,
    ) -> Result<(Self, Vec<Image>)> {
        let (surface_loader, surface) = create_surface(instance, window)?;
        let surface_capabilities = unsafe {
            surface_loader.get_physical_device_surface_capabilities(
                device.physical_device,
                surface,
            )?
        };
        let format =
            swapchain_format(&surface_loader, surface, device.physical_device)?;
        let usages = vk::ImageUsageFlags::COLOR_ATTACHMENT
            | vk::ImageUsageFlags::TRANSFER_DST
            | vk::ImageUsageFlags::STORAGE;
        let swapchain_info = vk::SwapchainCreateInfoKHR::builder()
            .surface(surface)
            .min_image_count(surface_capabilities.min_image_count.max(2))
            .image_color_space(vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .image_array_layers(1)
            .image_usage(usages)
            .image_format(format)
            .queue_family_indices(slice::from_ref(&device.queue_family_index))
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .image_extent(extent)
            .composite_alpha({
                let composite_modes = [
                    vk::CompositeAlphaFlagsKHR::OPAQUE,
                    vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED,
                    vk::CompositeAlphaFlagsKHR::POST_MULTIPLIED,
                ];
                composite_modes
                    .into_iter()
                    .find(|mode| {
                        surface_capabilities
                            .supported_composite_alpha
                            .contains(*mode)
                    })
                    .unwrap_or(vk::CompositeAlphaFlagsKHR::INHERIT)
            })
            .present_mode(vk::PresentModeKHR::FIFO);
        let swapchain_loader = khr::Swapchain::new(instance, device);
        let swapchain = unsafe {
            swapchain_loader.create_swapchain(&swapchain_info, None)?
        };
        let images = unsafe {
            swapchain_loader
                .get_swapchain_images(swapchain)
                .wrap_err("failed to get swapchain images")?
        };
        let images: Vec<_> = images
            .into_iter()
            .map(|image| {
                let request = ImageViewRequest::BASE;
                let view = ImageView::new(device, image, format, request)?;
                Ok(Image {
                    layout: vk::ImageLayout::UNDEFINED.into(),
                    aspect: vk::ImageAspectFlags::COLOR,
                    views: vec![view],
                    extent: extent.into(),
                    swapchain: true,
                    format,
                    image,
                })
            })
            .collect::<Result<_>>()?;
        let swapchain = Self {
            surface_loader,
            surface,
            swapchain_loader,
            swapchain,
            format,
            extent,
        };
        Ok((swapchain, images))
    }

    pub(super) fn destroy(&self, _device: &Device) {
        unsafe {
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }

    pub(super) fn image_index(&self, sync: &Sync) -> Result<u32> {
        let (index, outdated) = unsafe {
            self.swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    sync.acquire,
                    vk::Fence::null(),
                )
                .wrap_err("failed to acquire swapchain image index")?
        };
        if outdated {
            println!("outdated swapchain image");
        }
        Ok(index)
    }

    pub(super) fn present(
        &self,
        device: &Device,
        sync: &Sync,
        index: u32,
    ) -> Result<()> {
        let present_info = vk::PresentInfoKHR::builder()
            .image_indices(slice::from_ref(&index))
            .swapchains(slice::from_ref(&self.swapchain))
            .wait_semaphores(slice::from_ref(&sync.release))
            .build();
        let result = unsafe {
            self.swapchain_loader.queue_present(device.queue, &present_info)
        };
        match result {
            Result::Ok(true) => {
                println!("suboptimal swapchain image");
            }
            Result::Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                println!("out of date swapchain");
            }
            _ => (),
        }
        Ok(())
    }
}

pub fn create_surface(
    instance: &Instance,
    window: RawWindowHandle,
) -> Result<(khr::Surface, vk::SurfaceKHR)> {
    let loader = khr::Surface::new(&instance.entry, instance);
    let surface = match window {
        #[cfg(target_os = "windows")]
        RawWindowHandle::Win32(handle) => {
            let info = vk::Win32SurfaceCreateInfoKHR::default()
                .hinstance(handle.hinstance)
                .hwnd(handle.hwnd);
            let loader =
                khr::Win32Surface::new(&instance.entry, &instance.handle);
            unsafe { loader.create_win32_surface(&info, None) }
        }
        #[cfg(target_os = "linux")]
        RawWindowHandle::Wayland(handle) => {
            let info = vk::WaylandSurfaceCreateInfoKHR::builder()
                .display(handle.display)
                .surface(handle.surface);
            let loader = khr::WaylandSurface::new(&instance.entry, instance);
            unsafe { loader.create_wayland_surface(&info, None) }
        }
        #[cfg(target_os = "linux")]
        RawWindowHandle::Xlib(handle) => {
            let info = vk::XlibSurfaceCreateInfoKHR::builder()
                .dpy(handle.display as *mut _)
                .window(handle.window);
            let loader = khr::XlibSurface::new(&instance.entry, instance);
            unsafe { loader.create_xlib_surface(&info, None) }
        }
        #[cfg(target_os = "linux")]
        RawWindowHandle::Xcb(handle) => {
            let info = vk::XcbSurfaceCreateInfoKHR::builder()
                .connection(handle.connection)
                .window(handle.window);
            let loader = khr::XcbSurface::new(&instance.entry, instance);
            unsafe { loader.create_xcb_surface(&info, None) }
        }
        #[cfg(target_os = "macos")]
        RawWindowHandle::AppKit(handle) => unsafe {
            use raw_window_metal::{appkit, Layer};
            let layer = appkit::metal_layer_from_handle(handle);
            let layer = match layer {
                Layer::Existing(layer) | Layer::Allocated(layer) => {
                    layer as *mut _
                }
                Layer::None => {
                    return Err(anyhow!("failed to load metal layer"));
                }
            };
            let info = vk::MetalSurfaceCreateInfoEXT::builder().layer(&*layer);
            let loader =
                ext::MetalSurface::new(&instance.entry, &instance.handle);
            loader.create_metal_surface(&info, None)
        },
        _ => {
            return Err(eyre::eyre!("unsupported platform"));
        }
    };
    let surface = surface.wrap_err("failed to create surface")?;
    Ok((loader, surface))
}

fn swapchain_format(
    loader: &khr::Surface,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
) -> Result<vk::Format> {
    let formats = unsafe {
        loader.get_physical_device_surface_formats(physical_device, surface)?
    };
    if formats.is_empty() {
        return Err(eyre::eyre!("no supported swawpchain formats"));
    }
    if formats.len() == 1
        && formats
            .first()
            .is_some_and(|format| format.format == vk::Format::UNDEFINED)
    {
        return Ok(vk::Format::R8G8B8A8_UNORM);
    }
    let Some(format) = formats.into_iter().find(|format| {
        format.format == vk::Format::R8G8B8A8_UNORM
            || format.format == vk::Format::B8G8R8A8_UNORM
    }) else {
        return Err(eyre::eyre!("no supported swawpchain formats"));
    };
    Ok(format.format)
}
