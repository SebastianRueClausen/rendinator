use std::ffi::{self, CStr, CString};
use std::ops::Deref;

use ash::extensions::{ext, khr};
use ash::vk;
use eyre::{Result, WrapErr};

pub struct Instance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub debug_utils: ext::DebugUtils,
    pub debug_messenger: vk::DebugUtilsMessengerEXT,
}

impl Instance {
    pub fn new(validate: bool) -> Result<Self> {
        let entry = unsafe { ash::Entry::load().wrap_err("loading entry")? };
        let mut debug_info = {
            use vk::{
                DebugUtilsMessageSeverityFlagsEXT as Severity,
                DebugUtilsMessageTypeFlagsEXT as Type,
            };
            vk::DebugUtilsMessengerCreateInfoEXT::builder()
                .message_severity(
                    Severity::ERROR
                        | Severity::WARNING
                        | Severity::INFO
                        | Severity::VERBOSE,
                )
                .message_type(
                    Type::GENERAL
                        | Type::PERFORMANCE
                        | Type::VALIDATION
                        | Type::DEVICE_ADDRESS_BINDING,
                )
                .pfn_user_callback(Some(debug_callback))
        };
        let layers = if validate {
            vec![CString::new("VK_LAYER_KHRONOS_validation").unwrap()]
        } else {
            vec![]
        };
        let layer_names: Vec<_> =
            layers.iter().map(|layer| layer.as_ptr()).collect();
        let extension_names = [
            ext::DebugUtils::name().as_ptr(),
            khr::Surface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            khr::WaylandSurface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            khr::XlibSurface::name().as_ptr(),
            #[cfg(target_os = "linux")]
            khr::XcbSurface::name().as_ptr(),
        ];
        let application_info = vk::ApplicationInfo::builder()
            .api_version(vk::make_api_version(0, 1, 3, 0));
        let instance_info = vk::InstanceCreateInfo::builder()
            .push_next(&mut debug_info)
            .application_info(&application_info)
            .enabled_layer_names(&layer_names)
            .enabled_extension_names(&extension_names);
        let instance = unsafe { entry.create_instance(&instance_info, None)? };
        let debug_utils = ext::DebugUtils::new(&entry, &instance);
        let debug_messenger = unsafe {
            debug_utils.create_debug_utils_messenger(&debug_info, None)?
        };
        Ok(Self { entry, instance, debug_utils, debug_messenger })
    }

    pub fn destroy(&self) {
        unsafe {
            self.debug_utils
                .destroy_debug_utils_messenger(self.debug_messenger, None);
            self.instance.destroy_instance(None);
        }
    }
}

impl Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

unsafe extern "system" fn debug_callback(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    ty: vk::DebugUtilsMessageTypeFlagsEXT,
    cb_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _: *mut ffi::c_void,
) -> vk::Bool32 {
    let types = vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
        | vk::DebugUtilsMessageTypeFlagsEXT::DEVICE_ADDRESS_BINDING;
    let severities = vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING;
    if types.contains(ty) && severities.contains(severity) {
        let message = CStr::from_ptr((*cb_data).p_message);
        println!("vulkan({ty:?}): {message:?}\n");
    }
    vk::FALSE
}