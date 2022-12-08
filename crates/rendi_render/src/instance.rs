use ash::{
    extensions::{ext, khr},
    vk,
};
use std::ffi::{self, CStr, CString};

use crate::{PhysicalDevice, RenderError};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationLayers {
    /// Have validation layers enabled.
    ///
    /// Gives additional information if something goes wrong but also has some CPU overhead, and
    /// requires that vulkan validation layers are installed.
    Enabled,
    /// Have validtion layers disabled.
    Disabled,
}

/// A vulkan instance and debug messenger.
///
/// # Examples
///
/// ```
/// use rendi_render::{Instance, ValidationLayers};
///
/// let instance = Instance::new(ValidationLayers::Disabled);
/// assert!(instance.is_ok());
/// ```
pub struct Instance {
    handle: ash::Instance,
    layers: Vec<CString>,

    #[allow(unused)]
    entry: ash::Entry,
    #[allow(unused)]
    messenger: DebugMessenger,
}

impl Instance {
    pub fn new(validation: ValidationLayers) -> Result<Self, RenderError> {
        let entry = unsafe { ash::Entry::load()? };

        let required_exts = [
            ext::DebugUtils::name().as_ptr(),
            khr::Surface::name().as_ptr(),
            #[cfg(target_os = "windows")]
            khr::Win32Surface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            khr::WaylandSurface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            khr::XlibSurface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            khr::XcbSurface::name().as_ptr(),
            #[cfg(target_os = "macos")]
            ext::MetalSurface::name().as_ptr(),
            #[cfg(target_os = "macos")]
            vk::KhrPortabilityEnumerationFn::name().as_ptr(),
            #[cfg(target_os = "macos")]
            vk::KhrGetPhysicalDeviceProperties2Fn::name().as_ptr(),
        ];

        trace!("creating instance");

        // Check that all the required extensions are available.
        let available_exts = entry.enumerate_instance_extension_properties(None)?;
        for req in required_exts {
            let req_name = unsafe { CStr::from_ptr(req) };
            let available = available_exts.iter().any(|extension| unsafe {
                CStr::from_ptr(extension.extension_name.as_ptr()) == req_name
            });
            if !available {
                return Err(RenderError::MissingExt(
                    req_name
                        .to_str()
                        .expect("extension name should be valid ascii")
                        .into(),
                ));
            }
        }

        let mut debug_info = {
            use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;
            use vk::DebugUtilsMessageTypeFlagsEXT as Type;

            vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    Severity::ERROR | Severity::WARNING | Severity::INFO | Severity::VERBOSE,
                )
                .message_type(Type::GENERAL | Type::PERFORMANCE | Type::VALIDATION)
                .pfn_user_callback(Some(debug_callback))
        };

        let layers = match validation {
            ValidationLayers::Enabled => vec![CString::new(VALIDATE_LAYER_NAME).unwrap()],
            ValidationLayers::Disabled => Vec::default(),
        };

        let layer_names: Vec<_> = layers.iter().map(|layer| layer.as_ptr()).collect();
        let version = vk::make_api_version(0, 1, 3, 0);

        let handle = unsafe {
            let engine_name = CString::new("rendinator").unwrap();
            let app_name = CString::new("rendinator").unwrap();
            let app_info = vk::ApplicationInfo::builder()
                .application_name(&app_name)
                .application_version(vk::make_api_version(0, 0, 0, 1))
                .engine_name(&engine_name)
                .engine_version(vk::make_api_version(0, 0, 0, 1))
                .api_version(version);

            let flags = if cfg!(target_os = "macos") {
                vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
            } else {
                vk::InstanceCreateFlags::default()
            };

            let info = vk::InstanceCreateInfo::builder()
                .flags(flags)
                .push_next(&mut debug_info)
                .application_info(&app_info)
                .enabled_layer_names(&layer_names)
                .enabled_extension_names(&required_exts);

            match entry.create_instance(&info, None) {
                Ok(handle) => handle,
                Err(error) => {
                    return Err(if error == vk::Result::ERROR_LAYER_NOT_PRESENT {
                        RenderError::MissingLayer(VALIDATE_LAYER_NAME.into())
                    } else {
                        error.into()
                    })
                }
            }
        };

        let messenger = DebugMessenger::new(&entry, &handle, &debug_info)?;

        Ok(Self {
            entry,
            handle,
            layers,
            messenger,
        })
    }

    /// Returns a vector of the available physical devices.
    pub fn physical_devices(&self) -> Result<Vec<PhysicalDevice>, RenderError> {
        unsafe {
            self.handle
                .enumerate_physical_devices()?
                .into_iter()
                .map(|dev| PhysicalDevice::new(self, dev))
                .collect()
        }
    }

    pub(crate) fn layers(&self) -> &[CString] {
        &self.layers
    }

    pub(crate) fn handle(&self) -> &ash::Instance {
        &self.handle
    }

    pub(crate) fn entry(&self) -> &ash::Entry {
        &self.entry
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe {
            self.messenger
                .loader
                .destroy_debug_utils_messenger(self.messenger.handle, None);
            self.handle.destroy_instance(None);
        }
    }
}

struct DebugMessenger {
    loader: ext::DebugUtils,
    handle: vk::DebugUtilsMessengerEXT,
}

impl DebugMessenger {
    fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
        info: &vk::DebugUtilsMessengerCreateInfoEXT,
    ) -> Result<Self, RenderError> {
        trace!("creating debug messenger");

        let loader = ext::DebugUtils::new(entry, instance);
        let handle = unsafe { loader.create_debug_utils_messenger(info, None)? };

        Ok(Self { loader, handle })
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut ffi::c_void,
) -> vk::Bool32 {
    let message = CStr::from_ptr((*cb_data).p_message);

    use vk::DebugUtilsMessageSeverityFlagsEXT as Severity;

    if severity.contains(Severity::ERROR) {
        error!("vulkan({ty:?}): {message:?}");
    } else if severity.contains(Severity::WARNING) {
        warn!("vulkan({ty:?}): {message:?}");
    } else if severity.contains(Severity::INFO) {
        info!("vulkan({ty:?}): {message:?}");
    } else if severity.contains(Severity::VERBOSE) {
        trace!("vulkan({ty:?}): {message:?}");
    }

    vk::FALSE
}

const VALIDATE_LAYER_NAME: &str = "VK_LAYER_KHRONOS_validation";
