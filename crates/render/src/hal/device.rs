use std::ops::Deref;
use std::slice;

use ash::extensions::{ext, khr};
use ash::vk;
use eyre::{Context, Result};

use super::Instance;

pub struct Device {
    pub device: ash::Device,
    pub physical_device: vk::PhysicalDevice,
    pub queue_family_index: u32,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub queue: vk::Queue,
    pub command_pool: vk::CommandPool,
    pub descriptor_buffer_loader: ext::DescriptorBuffer,
    pub acc_struct_loader: khr::AccelerationStructure,
    pub descriptor_buffer_properties:
        vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
    pub limits: vk::PhysicalDeviceLimits,
}

impl Device {
    pub fn new(instance: &Instance) -> Result<Self> {
        let (physical_device, queue_family_index) =
            select_physical_device(instance)?;
        let memory_properties = unsafe {
            instance.get_physical_device_memory_properties(physical_device)
        };
        let properties =
            unsafe { instance.get_physical_device_properties(physical_device) };
        let limits = properties.limits;
        let queue_info = vk::DeviceQueueCreateInfo::builder()
            .queue_family_index(queue_family_index)
            .queue_priorities(&[1.0]);
        let extensions = [
            khr::Swapchain::name().as_ptr(),
            ext::MeshShader::name().as_ptr(),
            ext::DescriptorBuffer::name().as_ptr(),
            khr::DeferredHostOperations::name().as_ptr(),
            khr::AccelerationStructure::name().as_ptr(),
            khr::RayTracingPipeline::name().as_ptr(),
            vk::KhrRayQueryFn::name().as_ptr(),
        ];
        let mut features = vk::PhysicalDeviceFeatures2::builder()
            .features({
                vk::PhysicalDeviceFeatures::builder()
                    .independent_blend(true)
                    .multi_draw_indirect(true)
                    .pipeline_statistics_query(true)
                    .sampler_anisotropy(true)
                    // For access to the primitive index in shaders.
                    .geometry_shader(true)
                    .shader_int16(true)
                    .shader_int64(true)
                    .build()
            })
            .build();
        let mut features_1_1 = vk::PhysicalDeviceVulkan11Features::builder()
            .storage_buffer16_bit_access(true)
            .uniform_and_storage_buffer16_bit_access(true)
            .shader_draw_parameters(true)
            .build();
        let mut features_1_2 = vk::PhysicalDeviceVulkan12Features::builder()
            .buffer_device_address(true)
            .descriptor_binding_variable_descriptor_count(true)
            .runtime_descriptor_array(true)
            .draw_indirect_count(true)
            .storage_buffer8_bit_access(true)
            .shader_float16(true)
            .shader_int8(true)
            .sampler_filter_minmax(true)
            .scalar_block_layout(true)
            .build();
        let mut features_1_3 = vk::PhysicalDeviceVulkan13Features::builder()
            .dynamic_rendering(true)
            .synchronization2(true)
            .maintenance4(true)
            .build();
        let mut features_mesh =
            vk::PhysicalDeviceMeshShaderFeaturesEXT::builder()
                .task_shader(true)
                .mesh_shader(true)
                .build();
        let mut features_descriptor_buffer =
            vk::PhysicalDeviceDescriptorBufferFeaturesEXT::builder()
                .descriptor_buffer(true)
                .descriptor_buffer_image_layout_ignored(true)
                .build();
        let mut features_acc_struct =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true)
                .build();
        let mut features_ray_query =
            vk::PhysicalDeviceRayQueryFeaturesKHR::builder().ray_query(true);
        let device_info = vk::DeviceCreateInfo::builder()
            .queue_create_infos(slice::from_ref(&queue_info))
            .enabled_extension_names(&extensions)
            .push_next(&mut features)
            .push_next(&mut features_1_1)
            .push_next(&mut features_1_2)
            .push_next(&mut features_1_3)
            .push_next(&mut features_mesh)
            .push_next(&mut features_descriptor_buffer)
            .push_next(&mut features_acc_struct)
            .push_next(&mut features_ray_query);
        let device = unsafe {
            instance.create_device(physical_device, &device_info, None)?
        };
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };
        let command_pool = unsafe {
            let info = vk::CommandPoolCreateInfo::builder()
                .flags(vk::CommandPoolCreateFlags::empty())
                .queue_family_index(queue_family_index);
            device.create_command_pool(&info, None)?
        };
        let descriptor_buffer_loader =
            ext::DescriptorBuffer::new(instance, &device);
        let acc_struct_loader =
            khr::AccelerationStructure::new(instance, &device);
        let descriptor_buffer_properties =
            get_descriptor_buffer_properties(instance, physical_device);
        Ok(Self {
            device,
            physical_device,
            queue_family_index,
            memory_properties,
            command_pool,
            queue,
            descriptor_buffer_loader,
            descriptor_buffer_properties,
            acc_struct_loader,
            limits,
        })
    }

    pub fn wait_until_idle(&self) -> Result<()> {
        unsafe { self.device_wait_idle().wrap_err("failed to wait idle") }
    }

    pub fn destroy(&self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

fn get_graphics_queue_family_index(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> Option<u32> {
    unsafe {
        instance.get_physical_device_queue_family_properties(physical_device)
    }
    .into_iter()
    .enumerate()
    .find_map(|(queue_index, queue)| {
        queue
            .queue_flags
            .contains(vk::QueueFlags::GRAPHICS)
            .then_some(queue_index as u32)
    })
}

fn get_descriptor_buffer_properties(
    instance: &Instance,
    physical_device: vk::PhysicalDevice,
) -> vk::PhysicalDeviceDescriptorBufferPropertiesEXT {
    unsafe {
        let mut descriptor_buffer_properties =
            vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut props = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut descriptor_buffer_properties);
        instance.get_physical_device_properties2(physical_device, &mut props);
        descriptor_buffer_properties
    }
}

fn select_physical_device(
    instance: &Instance,
) -> Result<(vk::PhysicalDevice, u32)> {
    let mut fallback = None;
    let mut preferred = None;
    unsafe {
        for physical_device in instance.enumerate_physical_devices()? {
            let properties =
                instance.get_physical_device_properties(physical_device);
            let Some(queue_index) =
                get_graphics_queue_family_index(instance, physical_device)
            else {
                continue;
            };
            if properties.api_version < vk::make_api_version(0, 1, 3, 0) {
                continue;
            }
            if properties.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
                preferred.get_or_insert((physical_device, queue_index));
            }
            fallback.get_or_insert((physical_device, queue_index));
        }
    }
    preferred
        .or(fallback)
        .ok_or_else(|| eyre::eyre!("no suitable physical device found"))
}
