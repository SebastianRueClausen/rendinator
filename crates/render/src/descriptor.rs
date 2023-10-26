use std::ops::Deref;

use ash::vk;
use eyre::{Context, Result};

use crate::device::Device;
use crate::resources::{
    buffer_memory, Buffer, BufferKind, BufferRequest, BufferWrite, ImageView,
    Memory,
};
use crate::{command, resources};

#[derive(Clone, Copy)]
pub(crate) struct LayoutBinding {
    pub ty: vk::DescriptorType,
    pub count: u32,
}

pub(crate) struct Layout {
    layout: vk::DescriptorSetLayout,
}

impl Deref for Layout {
    type Target = vk::DescriptorSetLayout;

    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl Layout {
    pub fn new(device: &Device, bindings: &[LayoutBinding]) -> Result<Self> {
        let flags: Vec<_> = bindings
            .iter()
            .map(|binding| {
                (binding.count != 1)
                    .then_some(
                        vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
                    )
                    .unwrap_or_default()
            })
            .collect();
        let bindings: Vec<_> = bindings
            .iter()
            .enumerate()
            .map(|(location, binding)| {
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(location as u32)
                    .descriptor_type(binding.ty)
                    .stage_flags(vk::ShaderStageFlags::ALL)
                    .descriptor_count(binding.count)
                    .build()
            })
            .collect();
        let mut binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                .binding_flags(&flags);
        let create_flags =
            vk::DescriptorSetLayoutCreateFlags::DESCRIPTOR_BUFFER_EXT;
        let layout_info = vk::DescriptorSetLayoutCreateInfo::builder()
            .bindings(&bindings)
            .flags(create_flags)
            .push_next(&mut binding_flags);
        let layout = unsafe {
            device
                .create_descriptor_set_layout(&layout_info, None)
                .wrap_err("failed to create descriptor set layout")?
        };

        Ok(Self { layout })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe { device.destroy_descriptor_set_layout(self.layout, None) }
    }

    pub fn size(&self, device: &Device) -> vk::DeviceSize {
        unsafe {
            device
                .descriptor_buffer_loader
                .get_descriptor_set_layout_size(self.layout)
        }
    }
}

pub(crate) struct DescriptorBuffer {
    pub buffer: Buffer,
    pub memory: Memory,
    pub address: vk::DeviceAddress,
}

impl DescriptorBuffer {
    pub fn new(device: &Device, data: &[u8]) -> Result<Self> {
        let buffer = Buffer::new(
            device,
            &BufferRequest {
                kind: BufferKind::Descriptor { sampler: false },
                size: data.len() as vk::DeviceSize,
            },
        )?;

        let memory = buffer_memory(
            device,
            &buffer,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;

        let scratch = command::quickie(device, |command_buffer| {
            let write = [BufferWrite { buffer: &buffer, data }];
            resources::upload_buffer_data(device, command_buffer, &write)
        })?;

        scratch.destroy(device);

        let address = buffer.device_address(device);

        Ok(Self { buffer, memory, address })
    }

    pub fn destroy(&self, device: &Device) {
        self.buffer.destroy(device);
        self.memory.free(device);
    }
}

pub(crate) enum Binding<'a> {
    StorageBuffer(&'a [&'a Buffer]),
    UniformBuffer(&'a [&'a Buffer]),
    StorageImage(&'a [&'a ImageView]),
    SampledImage(&'a [&'a ImageView]),
}

pub(crate) fn descriptor_data<'a>(
    device: &Device,
    layout: &Layout,
    bindings: impl IntoIterator<Item = Binding<'a>>,
) -> Vec<u8> {
    let mut descriptor_data = vec![0x0; layout.size(device) as usize];

    for (location, binding) in bindings.into_iter().enumerate() {
        let offset = unsafe {
            let offset = device
                .descriptor_buffer_loader
                .get_descriptor_set_layout_binding_offset(
                    *layout.deref(),
                    location as u32,
                );
            offset as usize
        };

        match binding {
            Binding::StorageBuffer(buffers) => {
                let size = device
                    .descriptor_buffer_properties
                    .storage_buffer_descriptor_size;
                for (index, buffer) in buffers.iter().enumerate() {
                    let descriptor_address_info =
                        vk::DescriptorAddressInfoEXT::builder()
                            .address(buffer.device_address(device))
                            .range(buffer.size)
                            .build();
                    let descriptor_info = vk::DescriptorGetInfoEXT::builder()
                        .ty(vk::DescriptorType::STORAGE_BUFFER)
                        .data(vk::DescriptorDataEXT {
                            p_storage_buffer: &descriptor_address_info
                                as *const _,
                        });
                    let start = offset + size * index;
                    unsafe {
                        device.descriptor_buffer_loader.get_descriptor(
                            &descriptor_info,
                            &mut descriptor_data[start..start + size],
                        );
                    }
                }
            }
            Binding::UniformBuffer(buffers) => {
                let size = device
                    .descriptor_buffer_properties
                    .uniform_buffer_descriptor_size;
                for (index, buffer) in buffers.iter().enumerate() {
                    let descriptor_address_info =
                        vk::DescriptorAddressInfoEXT::builder()
                            .address(buffer.device_address(device))
                            .range(buffer.size)
                            .build();
                    let descriptor_info = vk::DescriptorGetInfoEXT::builder()
                        .ty(vk::DescriptorType::UNIFORM_BUFFER)
                        .data(vk::DescriptorDataEXT {
                            p_uniform_buffer: &descriptor_address_info
                                as *const _,
                        });
                    let start = offset + size * index;
                    unsafe {
                        device.descriptor_buffer_loader.get_descriptor(
                            &descriptor_info,
                            &mut descriptor_data[start..start + size],
                        );
                    }
                }
            }
            Binding::StorageImage(views) => {
                let size = device
                    .descriptor_buffer_properties
                    .storage_image_descriptor_size;
                for (index, view) in views.iter().copied().enumerate() {
                    let descriptor_image_info =
                        vk::DescriptorImageInfo::builder()
                            .image_view(**view)
                            .build();
                    let descriptor_info = vk::DescriptorGetInfoEXT::builder()
                        .ty(vk::DescriptorType::STORAGE_IMAGE)
                        .data(vk::DescriptorDataEXT {
                            p_storage_image: &descriptor_image_info as *const _,
                        });
                    let start = offset + size * index;
                    unsafe {
                        device.descriptor_buffer_loader.get_descriptor(
                            &descriptor_info,
                            &mut descriptor_data[start..start + size],
                        );
                    }
                }
            }
            Binding::SampledImage(views) => {
                let size = device
                    .descriptor_buffer_properties
                    .sampled_image_descriptor_size;
                for (index, view) in views.iter().copied().enumerate() {
                    let descriptor_image_info =
                        vk::DescriptorImageInfo::builder()
                            .image_view(**view)
                            .build();
                    let descriptor_info = vk::DescriptorGetInfoEXT::builder()
                        .ty(vk::DescriptorType::SAMPLED_IMAGE)
                        .data(vk::DescriptorDataEXT {
                            p_sampled_image: &descriptor_image_info as *const _,
                        });
                    let start = offset + size * index;
                    unsafe {
                        device.descriptor_buffer_loader.get_descriptor(
                            &descriptor_info,
                            &mut descriptor_data[start..start + size],
                        );
                    }
                }
            }
        }
    }

    descriptor_data
}
