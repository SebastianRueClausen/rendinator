use std::slice;

use ash::extensions::khr;
use ash::vk;
use eyre::{Context, Result};
use raw_window_handle::{RawDisplayHandle, RawWindowHandle};

use super::{Device, Image, ImageView, ImageViewRequest, Instance, Sync};

pub struct Swapchain {
    surface_loader: khr::Surface,
    swapchain_loader: khr::Swapchain,
    surface: vk::SurfaceKHR,
    swapchain: vk::SwapchainKHR,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(
        instance: &Instance,
        device: &Device,
        window: RawWindowHandle,
        display: RawDisplayHandle,
        extent: vk::Extent2D,
    ) -> Result<(Self, Vec<Image>)> {
        let (surface_loader, surface) =
            create_surface(instance, window, display)?;
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
                    mip_level_count: 1,
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

    pub fn destroy(&self, _device: &Device) {
        unsafe {
            self.swapchain_loader.destroy_swapchain(self.swapchain, None);
            self.surface_loader.destroy_surface(self.surface, None);
        }
    }

    pub fn image_index(&self, sync: &Sync) -> Result<u32> {
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

    pub fn present(
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
    display: RawDisplayHandle,
) -> Result<(khr::Surface, vk::SurfaceKHR)> {
    let loader = khr::Surface::new(&instance.entry, instance);
    let surface = match (display, window) {
        (RawDisplayHandle::Windows(_), RawWindowHandle::Win32(handle)) => {
            let info = vk::Win32SurfaceCreateInfoKHR::builder()
                .hinstance(handle.hinstance)
                .hwnd(handle.hwnd);
            let loader =
                khr::Win32Surface::new(&instance.entry, &instance.instance);
            unsafe { loader.create_win32_surface(&info, None) }
        }
        (
            RawDisplayHandle::Wayland(display),
            RawWindowHandle::Wayland(window),
        ) => {
            let info = vk::WaylandSurfaceCreateInfoKHR::builder()
                .display(display.display)
                .surface(window.surface);
            let loader = khr::WaylandSurface::new(&instance.entry, instance);
            unsafe { loader.create_wayland_surface(&info, None) }
        }
        (RawDisplayHandle::Xlib(display), RawWindowHandle::Xlib(window)) => {
            let info = vk::XlibSurfaceCreateInfoKHR::builder()
                .dpy(display.display.cast())
                .window(window.window);
            let loader = khr::XlibSurface::new(&instance.entry, instance);
            unsafe { loader.create_xlib_surface(&info, None) }
        }
        (RawDisplayHandle::Xcb(display), RawWindowHandle::Xcb(window)) => {
            let info = vk::XcbSurfaceCreateInfoKHR::builder()
                .connection(display.connection)
                .window(window.window);
            let loader = khr::XcbSurface::new(&instance.entry, instance);
            unsafe { loader.create_xcb_surface(&info, None) }
        }
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
