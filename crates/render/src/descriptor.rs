use std::mem;
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

pub(crate) struct Builder<'a> {
    layout: &'a Layout,
    device: &'a Device,
    bound: usize,
    data: Vec<u8>,
}

impl<'a> Builder<'a> {
    pub fn new(device: &'a Device, layout: &'a Layout) -> Self {
        let size = layout.size(device) as usize;
        Self { data: vec![0; size], bound: 0, layout, device }
    }

    fn offset(&self, location: usize) -> usize {
        unsafe {
            let offset = self
                .device
                .descriptor_buffer_loader
                .get_descriptor_set_layout_binding_offset(
                    **self.layout,
                    location as u32,
                );
            offset as usize
        }
    }

    fn next_offset(&mut self) -> usize {
        let offset = self.offset(self.bound);
        self.bound += 1;
        offset
    }

    fn write_descriptor(
        &mut self,
        start: usize,
        size: usize,
        descriptor_info: &vk::DescriptorGetInfoEXT,
    ) {
        unsafe {
            self.device.descriptor_buffer_loader.get_descriptor(
                descriptor_info,
                &mut self.data[start..start + size],
            );
        }
        debug_assert!(!self.data[start..start + size]
            .iter()
            .all(|byte| *byte == 0));
    }

    pub fn storage_buffers(
        &mut self,
        buffers: impl IntoIterator<Item = &'a Buffer>,
    ) -> &mut Self {
        let size = self
            .device
            .descriptor_buffer_properties
            .storage_buffer_descriptor_size;
        let offset = self.next_offset();
        for (index, buffer) in buffers.into_iter().enumerate() {
            let descriptor_address_info =
                vk::DescriptorAddressInfoEXT::builder()
                    .address(buffer.device_address(self.device))
                    .range(buffer.size)
                    .build();
            let descriptor_info = vk::DescriptorGetInfoEXT::builder()
                .ty(vk::DescriptorType::STORAGE_BUFFER)
                .data(vk::DescriptorDataEXT {
                    p_storage_buffer: &descriptor_address_info as *const _,
                });
            let start = offset + size * index;
            self.write_descriptor(start, size, &descriptor_info);
        }
        self
    }

    pub fn uniform_buffers(
        &mut self,
        buffers: impl IntoIterator<Item = &'a Buffer>,
    ) -> &mut Self {
        let size = self
            .device
            .descriptor_buffer_properties
            .uniform_buffer_descriptor_size;
        let offset = self.next_offset();
        for (index, buffer) in buffers.into_iter().enumerate() {
            let descriptor_address_info =
                vk::DescriptorAddressInfoEXT::builder()
                    .address(buffer.device_address(self.device))
                    .range(buffer.size)
                    .build();
            let descriptor_info = vk::DescriptorGetInfoEXT::builder()
                .ty(vk::DescriptorType::UNIFORM_BUFFER)
                .data(vk::DescriptorDataEXT {
                    p_uniform_buffer: &descriptor_address_info as *const _,
                });
            let start = offset + size * index;
            self.write_descriptor(start, size, &descriptor_info);
        }
        self
    }

    pub fn storage_images(
        &mut self,
        views: impl IntoIterator<Item = &'a ImageView>,
    ) -> &mut Self {
        let size = self
            .device
            .descriptor_buffer_properties
            .storage_image_descriptor_size;
        let offset = self.next_offset();
        for (index, view) in views.into_iter().enumerate() {
            let descriptor_image_info =
                vk::DescriptorImageInfo::builder().image_view(**view).build();
            let descriptor_info = vk::DescriptorGetInfoEXT::builder()
                .ty(vk::DescriptorType::STORAGE_IMAGE)
                .data(vk::DescriptorDataEXT {
                    p_storage_image: &descriptor_image_info as *const _,
                });
            let start = offset + size * index;
            self.write_descriptor(start, size, &descriptor_info);
        }
        self
    }

    pub fn sampled_images(
        &mut self,
        views: impl IntoIterator<Item = &'a ImageView>,
    ) -> &mut Self {
        let size = self
            .device
            .descriptor_buffer_properties
            .sampled_image_descriptor_size;
        let offset = self.next_offset();
        for (index, view) in views.into_iter().enumerate() {
            let descriptor_image_info =
                vk::DescriptorImageInfo::builder().image_view(**view).build();
            let descriptor_info = vk::DescriptorGetInfoEXT::builder()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .data(vk::DescriptorDataEXT {
                    p_sampled_image: &descriptor_image_info as *const _,
                });
            let start = offset + size * index;
            self.write_descriptor(start, size, &descriptor_info);
        }
        self
    }

    pub fn storage_buffer(&mut self, buffer: &'a Buffer) -> &mut Self {
        self.storage_buffers([buffer])
    }

    pub fn uniform_buffer(&mut self, buffer: &'a Buffer) -> &mut Self {
        self.uniform_buffers([buffer])
    }

    pub fn storage_image(&mut self, view: &'a ImageView) -> &mut Self {
        self.storage_images([view])
    }

    pub fn sampled_image(&mut self, view: &'a ImageView) -> &mut Self {
        self.sampled_images([view])
    }

    pub fn finish(&mut self) -> Vec<u8> {
        self.bound = 0;
        mem::take(&mut self.data)
    }
}
