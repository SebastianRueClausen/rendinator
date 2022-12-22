use crate::{mem, surface, Instance, RenderError, Surface};
use ash::{extensions::khr, vk};
use rendi_res::Res;
use std::cmp::Ordering;
use std::ffi::CStr;
use std::fmt;
use std::hash::{Hash, Hasher};

/// A representation of a graphics card.
pub struct PhysicalDevice {
    pub(crate) handle: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
    queue_properties: Vec<vk::QueueFamilyProperties>,
    name: String,
}

fn get_device_name(properties: &vk::PhysicalDeviceProperties) -> String {
    unsafe {
        CStr::from_ptr(properties.device_name.as_ptr())
            .to_str()
            .unwrap_or("invalid")
            .to_string()
    }
}

impl PhysicalDevice {
    pub(crate) fn new(
        instance: &Instance,
        handle: vk::PhysicalDevice,
    ) -> Result<Self, RenderError> {
        let memory_properties = unsafe {
            instance
                .handle()
                .get_physical_device_memory_properties(handle)
        };
        let properties = unsafe { instance.handle().get_physical_device_properties(handle) };
        let queue_properties = unsafe {
            instance
                .handle()
                .get_physical_device_queue_family_properties(handle)
        };
        let name = get_device_name(&properties);
        Ok(Self {
            handle,
            name,
            memory_properties,
            properties,
            queue_properties,
        })
    }

    /// Select the "best" physical device.
    ///
    /// This will favour discrete GPUs and use various heuristics to choose the most capable
    /// physical device.
    pub fn select_best(
        instance: &Instance,
        surface: Option<&Surface>,
    ) -> Result<Self, RenderError> {
        let handle = unsafe {
            instance
                .handle()
                .enumerate_physical_devices()?
                .into_iter()
                .max_by_key(|dev| {
                    let properties = instance.handle().get_physical_device_properties(*dev);

                    if log::log_enabled!(log::Level::Trace) {
                        trace!("device candidate: {}", get_device_name(&properties));
                    }

                    let mut score = properties.limits.max_image_dimension2_d
                        + properties.limits.max_framebuffer_width
                        + properties.limits.max_framebuffer_height;

                    if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                        score += 7000;
                    }
                    if properties.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
                        score += 3000;
                    }
                    if properties.device_type == vk::PhysicalDeviceType::CPU {
                        score = 0;
                    }

                    score
                })
                .ok_or(RenderError::NoPhysicalDeviceFound)?
        };

        let physical = Self::new(instance, handle)?;
        if log::log_enabled!(log::Level::Trace) {
            trace!("chose device: {}", physical.name());
        }

        Ok(physical)
    }

    /// Returns a queue request if it's available from the physical device.
    pub fn get_queue_req(&self, kind: QueueRequestKind) -> Option<QueueRequest> {
        let index = match &kind {
            QueueRequestKind::Graphics(surface) => {
                let flags = vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE;

                self.queue_properties.iter().enumerate().position(|(i, p)| {
                    let i = i as u32;
                    p.queue_flags.contains(flags)
                        && unsafe {
                            surface
                                .loader
                                .get_physical_device_surface_support(self.handle, i, surface.handle)
                                .unwrap_or(false)
                        }
                })
            }
            QueueRequestKind::Transfer => self
                .queue_properties
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::TRANSFER)),
            QueueRequestKind::Compute => self
                .queue_properties
                .iter()
                .position(|p| p.queue_flags.contains(vk::QueueFlags::COMPUTE)),
        };

        index.map(|family_index| {
            let flags = self.queue_properties[family_index].queue_flags;
            QueueRequest {
                flags,
                family_index: family_index as u32,
            }
        })
    }

    pub fn get_memory_type(
        &self,
        type_bits: u32,
        location: mem::MemoryLocation,
    ) -> Result<mem::MemoryType, RenderError> {
        let flags: vk::MemoryPropertyFlags = location.into();
        let count = self.memory_properties.memory_type_count as usize;

        self.memory_properties.memory_types[0..count]
            .iter()
            .enumerate()
            .position(|(i, memory_type)| {
                let masked = memory_type.property_flags & flags == flags;
                let bit = type_bits & (1 << i) != 0;

                bit & masked
            })
            .map(|i| mem::MemoryType::new(i as u32, location))
            .ok_or(RenderError::InvalidMemoryType {
                location,
                type_bits,
            })
    }

    /// Get the name of the physical device.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl fmt::Debug for PhysicalDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PhysicalDevice")
            .field("name", &self.name)
            .finish()
    }
}

impl PartialEq for PhysicalDevice {
    fn eq(&self, other: &Self) -> bool {
        self.handle.eq(&other.handle)
    }
}

impl Eq for PhysicalDevice {}

impl Hash for PhysicalDevice {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.hash(state);
    }
}

pub struct Device {
    handle: ash::Device,
    physical: PhysicalDevice,

    #[allow(unused)]
    instance: Res<Instance>,
}

impl Device {
    /// Creates a new device from the physical device `physical`. `queue_reqs` requests the queues
    /// that can be created from this device.
    pub fn new(
        instance: Res<Instance>,
        physical: PhysicalDevice,
        queue_reqs: &[&QueueRequest],
    ) -> Result<Self, RenderError> {
        trace!("creating device from physical device: {}", physical.name());

        let priorities = [1.0_f32];
        let mut queue_infos: Vec<_> = queue_reqs
            .iter()
            .map(|req| {
                vk::DeviceQueueCreateInfo::builder()
                    .queue_family_index(req.family_index)
                    .queue_priorities(&priorities)
                    .build()
            })
            .collect();

        queue_infos.dedup_by_key(|info| info.queue_family_index);

        let extensions = [
            khr::Swapchain::name().as_ptr(),
            khr::PushDescriptor::name().as_ptr(),
            #[cfg(target_os = "macos")]
            vk::KhrPortabilitySubsetFn::name().as_ptr(),
        ];

        let enabled_features = vk::PhysicalDeviceFeatures::builder()
            .shader_storage_image_multisample(true)
            .sampler_anisotropy(true)
            .multi_draw_indirect(true)
            .build();
        let mut vk11_features = vk::PhysicalDeviceVulkan11Features::builder()
            .storage_buffer16_bit_access(true)
            .shader_draw_parameters(true)
            .build();
        let mut vk12_features = vk::PhysicalDeviceVulkan12Features::builder()
            .shader_sampled_image_array_non_uniform_indexing(true)
            .shader_input_attachment_array_dynamic_indexing(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .scalar_block_layout(true)
            .draw_indirect_count(true)
            .descriptor_indexing(true)
            .shader_float16(true)
            .build();
        let mut vk13_features = vk::PhysicalDeviceVulkan13Features::builder()
            .dynamic_rendering(true)
            .synchronization2(true)
            .build();
        let layer_names: Vec<_> = instance
            .layers()
            .iter()
            .map(|layer| layer.as_ptr())
            .collect();
        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&extensions)
            .enabled_layer_names(&layer_names)
            .enabled_features(&enabled_features)
            .push_next(&mut vk11_features)
            .push_next(&mut vk12_features)
            .push_next(&mut vk13_features);
        let handle = unsafe {
            instance
                .handle()
                .create_device(physical.handle, &device_info, None)?
        };
        Ok(Self {
            instance,
            physical,
            handle,
        })
    }

    /// Wait until the device is idle.
    pub fn wait_until_idle(&self) {
        unsafe {
            self.handle()
                .device_wait_idle()
                .expect("failed waiting for idle device");
        }
    }

    /// Get the physical device used to create the device.
    pub fn physical(&self) -> &PhysicalDevice {
        &self.physical
    }

    /// Get the raw vulkan handle.
    pub fn handle(&self) -> &ash::Device {
        &self.handle
    }

    pub fn instance(&self) -> &Res<Instance> {
        &self.instance
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.handle.destroy_device(None);
        }
    }
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Device")
            .field("physical", &self.physical)
            .finish()
    }
}

impl PartialEq for Device {
    fn eq(&self, other: &Self) -> bool {
        self.handle.handle().eq(&other.handle.handle())
    }
}

impl Eq for Device {}

impl Hash for Device {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.handle.handle().hash(state);
    }
}

#[derive(Clone)]
pub enum QueueRequestKind {
    /// Graphics queues are guaranteed to be able to execute both render commands and compute
    /// commands.
    Graphics(Res<Surface>),
    /// Transfer queues are guaranteed to be able execute transfer commands.
    Transfer,
    /// Compute queues are guaranteed to be able execute compute commands.
    Compute,
}

pub type QueueFamilyIndex = u32;

#[derive(Clone, Copy)]
pub struct QueueRequest {
    flags: vk::QueueFlags,
    family_index: QueueFamilyIndex,
}

pub struct Queue {
    pub(crate) handle: vk::Queue,

    #[allow(unused)]
    pub(crate) pool: vk::CommandPool,

    #[allow(unused)]
    flags: vk::QueueFlags,
    family_index: QueueFamilyIndex,

    device: Res<Device>,
}

impl Queue {
    /// Create a new queue from the request `req`.
    pub fn new(device: Res<Device>, req: &QueueRequest) -> Result<Self, RenderError> {
        let (pool, handle) = unsafe {
            let info = vk::CommandPoolCreateInfo::builder()
                .queue_family_index(req.family_index)
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
            let pool = device.handle.create_command_pool(&info, None)?;
            let handle = device.handle.get_device_queue(req.family_index, 0);

            (pool, handle)
        };

        Ok(Self {
            handle,
            flags: req.flags,
            family_index: req.family_index,
            device,
            pool,
        })
    }

    /// Get the family index of the queue.
    pub fn family_index(&self) -> QueueFamilyIndex {
        self.family_index
    }

    /// Wait until the queue is idle.
    pub fn wait_until_idle(&self) -> Result<(), RenderError> {
        unsafe { self.device.handle.queue_wait_idle(self.handle)? }
        Ok(())
    }
}

impl fmt::Debug for Queue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Queue").field("flags", &self.flags).finish()
    }
}

impl PartialEq for Queue {
    fn eq(&self, other: &Self) -> bool {
        self.handle == other.handle
    }
}

impl Eq for Queue {}

impl PartialOrd for Queue {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.handle.partial_cmp(&other.handle)
    }
}

impl Ord for Queue {
    fn cmp(&self, other: &Self) -> Ordering {
        self.handle.cmp(&other.handle)
    }
}
