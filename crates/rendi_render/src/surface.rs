use ash::{extensions::khr, vk};

use crate::{Instance, PhysicalDevice, RenderError};
use rendi_res::Res;

/// Interface for
pub struct Surface {
    pub(crate) handle: vk::SurfaceKHR,
    pub(crate) loader: khr::Surface,

    #[allow(dead_code)]
    instance: Res<Instance>,
}

impl Surface {
    pub fn new(
        instance: Res<Instance>,
        window: &winit::window::Window,
    ) -> Result<Self, RenderError> {
        let loader = khr::Surface::new(instance.entry(), instance.handle());

        use raw_window_handle::{HasRawWindowHandle, RawWindowHandle};
        let handle = match window.raw_window_handle() {
            #[cfg(target_os = "windows")]
            RawWindowHandle::Win32(handle) => {
                let info = vk::Win32SurfaceCreateInfoKHR::default()
                    .hinstance(handle.hinstance)
                    .hwnd(handle.hwnd);
                let loader = khr::Win32Surface::new(instance.entry(), instance.handle());
                unsafe { loader.create_win32_surface(&info, None) }
            }

            #[cfg(target_os = "linux")]
            RawWindowHandle::Wayland(handle) => {
                let info = vk::WaylandSurfaceCreateInfoKHR::builder()
                    .display(handle.display)
                    .surface(handle.surface);
                let loader = khr::WaylandSurface::new(instance.entry(), instance.handle());
                unsafe { loader.create_wayland_surface(&info, None) }
            }

            #[cfg(target_os = "linux")]
            RawWindowHandle::Xlib(handle) => {
                let info = vk::XlibSurfaceCreateInfoKHR::builder()
                    .dpy(handle.display as *mut _)
                    .window(handle.window);
                let loader = khr::XlibSurface::new(instance.entry(), instance.handle());
                unsafe { loader.create_xlib_surface(&info, None) }
            }

            #[cfg(target_os = "linux")]
            RawWindowHandle::Xcb(handle) => {
                let info = vk::XcbSurfaceCreateInfoKHR::builder()
                    .connection(handle.connection)
                    .window(handle.window);
                let loader = khr::XcbSurface::new(instance.entry(), instance.handle());
                unsafe { loader.create_xcb_surface(&info, None) }
            }

            #[cfg(target_os = "macos")]
            RawWindowHandle::AppKit(handle) => unsafe {
                use ash::extensions::ext;
                use raw_window_metal::{appkit, Layer};

                let layer = appkit::metal_layer_from_handle(handle);
                let layer = match layer {
                    Layer::Existing(layer) | Layer::Allocated(layer) => layer as *mut _,
                    Layer::None => {
                        return Err(anyhow!("failed to load metal layer"));
                    }
                };

                let info = vk::MetalSurfaceCreateInfoEXT::builder().layer(&*layer);
                let loader = ext::MetalSurface::new(instance.entry(), instance.handle());

                loader.create_metal_surface(&info, None)
            },
            _ => return Err(RenderError::UnsupportedPlatform),
        };

        Ok(Self {
            handle: handle?,
            loader,
            instance,
        })
    }
}

impl Drop for Surface {
    fn drop(&mut self) {
        unsafe { self.loader.destroy_surface(self.handle, None) }
    }
}

pub(crate) fn surface_formats(
    physical: &PhysicalDevice,
    surface: &Surface,
) -> Result<Vec<vk::SurfaceFormatKHR>, RenderError> {
    let formats = unsafe {
        surface
            .loader
            .get_physical_device_surface_formats(physical.handle, surface.handle)?
    };
    Ok(formats)
}

pub(crate) fn present_modes(
    physical: &PhysicalDevice,
    surface: &Surface,
) -> Result<Vec<vk::PresentModeKHR>, RenderError> {
    let modes = unsafe {
        surface
            .loader
            .get_physical_device_surface_present_modes(physical.handle, surface.handle)?
    };
    Ok(modes)
}

pub(crate) fn surface_capabilities(
    physical: &PhysicalDevice,
    surface: &Surface,
) -> Result<vk::SurfaceCapabilitiesKHR, RenderError> {
    let capabilities = unsafe {
        surface
            .loader
            .get_physical_device_surface_capabilities(physical.handle, surface.handle)?
    };
    Ok(capabilities)
}
