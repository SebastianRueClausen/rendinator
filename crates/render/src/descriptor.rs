use std::iter;
use std::ops::Deref;

use ash::vk;
use eyre::{Context, Result};

use crate::device::Device;
use crate::resources::{
    buffer_memory, Buffer, BufferKind, BufferRequest, BufferWrite, ImageView,
    Memory, Sampler, Tlas,
};
use crate::{command, resources};

#[derive(Clone, Copy)]
pub(crate) struct LayoutBinding {
    pub ty: vk::DescriptorType,
    pub count: u32,
}

#[derive(Default)]
pub(crate) struct DescriptorLayoutBuilder {
    bindings: Vec<LayoutBinding>,
}

impl DescriptorLayoutBuilder {
    pub fn binding(&mut self, ty: vk::DescriptorType) -> &mut Self {
        self.bindings.push(LayoutBinding { count: 1, ty });
        self
    }

    pub fn array_binding(
        &mut self,
        ty: vk::DescriptorType,
        count: u32,
    ) -> &mut Self {
        self.bindings.push(LayoutBinding { count, ty });
        self
    }

    pub fn build(&mut self, device: &Device) -> Result<DescriptorLayout> {
        DescriptorLayout::new(device, &self.bindings)
    }
}

pub(crate) struct DescriptorLayout {
    layout: vk::DescriptorSetLayout,
}

impl Deref for DescriptorLayout {
    type Target = vk::DescriptorSetLayout;

    fn deref(&self) -> &Self::Target {
        &self.layout
    }
}

impl DescriptorLayout {
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

pub(super) struct DescriptorBuffer {
    pub buffer: Buffer,
    pub memory: Memory,
}

impl DescriptorBuffer {
    pub fn new(device: &Device, data: &DescriptorData) -> Result<Self> {
        let buffer = Buffer::new(
            device,
            &BufferRequest {
                kind: BufferKind::Descriptor,
                size: data.data.len() as vk::DeviceSize,
            },
        )?;
        let memory = buffer_memory(
            device,
            &buffer,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )?;
        let scratch = command::quickie(device, |command_buffer| {
            let write = [BufferWrite { buffer: &buffer, data: &data.data }];
            resources::upload_buffer_data(device, command_buffer, &write)
        })?;
        scratch.destroy(device);
        Ok(Self { buffer, memory })
    }

    pub fn destroy(&self, device: &Device) {
        self.buffer.destroy(device);
        self.memory.free(device);
    }
}

pub(crate) struct DescriptorBuilder<'a> {
    layout: &'a DescriptorLayout,
    device: &'a Device,
    data: &'a mut DescriptorData,
    /// The base offset into `data``.
    data_start: usize,
    /// The number of bindings bound.
    bound: usize,
}

impl<'a> DescriptorBuilder<'a> {
    pub fn new(
        device: &'a Device,
        layout: &'a DescriptorLayout,
        data: &'a mut DescriptorData,
    ) -> Self {
        let data_start = data.reserve_for_layout(device, layout);
        Self { bound: 0, layout, device, data, data_start }
    }

    /// Returns the byte offset into the set of the binding at `location`.
    fn set_offset(&self, location: usize) -> usize {
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

    fn next_binding(&mut self) -> usize {
        let set_offset = self.set_offset(self.bound);
        self.bound += 1;
        self.data_start + set_offset
    }

    fn write_descriptor(
        &mut self,
        start: usize,
        size: usize,
        descriptor_info: &vk::DescriptorGetInfoEXT,
    ) {
        let data = &mut self.data.data[start..start + size];
        unsafe {
            self.device
                .descriptor_buffer_loader
                .get_descriptor(descriptor_info, data);
        }
        debug_assert!(!self.data.data[start..start + size]
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
        let offset = self.next_binding();
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
        let offset = self.next_binding();
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
        let offset = self.next_binding();
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

    pub fn combined_image_samplers(
        &mut self,
        sampler: &Sampler,
        views: impl IntoIterator<Item = &'a ImageView>,
    ) -> &mut Self {
        let size = self
            .device
            .descriptor_buffer_properties
            .combined_image_sampler_descriptor_size;
        let offset = self.next_binding();
        for (index, view) in views.into_iter().enumerate() {
            let descriptor_image_info = vk::DescriptorImageInfo::builder()
                .sampler(**sampler)
                .image_view(**view)
                .build();
            let descriptor_info = vk::DescriptorGetInfoEXT::builder()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .data(vk::DescriptorDataEXT {
                    p_combined_image_sampler: &descriptor_image_info
                        as *const _,
                });
            let start = offset + size * index;
            self.write_descriptor(start, size, &descriptor_info);
        }
        self
    }

    pub fn tlas(&mut self, tlas: &Tlas) -> &mut Self {
        let descriptor_info = vk::DescriptorGetInfoEXT::builder()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .data(vk::DescriptorDataEXT {
                acceleration_structure: tlas.device_address(self.device),
            });
        let size = self
            .device
            .descriptor_buffer_properties
            .acceleration_structure_descriptor_size;
        let offset = self.next_binding();
        self.write_descriptor(offset, size, &descriptor_info);
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

    pub fn combined_image_sampler(
        &mut self,
        sampler: &Sampler,
        view: &'a ImageView,
    ) -> &mut Self {
        self.combined_image_samplers(sampler, [view])
    }

    pub fn build(&mut self) -> Descriptor {
        Descriptor { buffer_offset: self.data_start as u64 }
    }
}

#[derive(Debug)]
pub(crate) struct DescriptorData {
    alignment: usize,
    data: Vec<u8>,
}

impl DescriptorData {
    pub fn new(device: &Device) -> Self {
        let alignment = device
            .descriptor_buffer_properties
            .descriptor_buffer_offset_alignment;
        Self { data: Vec::default(), alignment: alignment as usize }
    }

    pub fn builder<'a>(
        &'a mut self,
        device: &'a Device,
        layout: &'a DescriptorLayout,
    ) -> DescriptorBuilder<'a> {
        DescriptorBuilder::new(device, layout, self)
    }

    /// Reserve space for layout.
    /// Returns the offset where the layout should begin.
    fn reserve_for_layout(
        &mut self,
        device: &Device,
        layout: &DescriptorLayout,
    ) -> usize {
        let start = self.data.len();
        let alignment_offset = start.next_multiple_of(self.alignment) - start;
        let size = layout.size(device) as usize;
        self.data.extend(iter::repeat(0).take(size + alignment_offset));
        start + alignment_offset
    }
}

pub(crate) struct Descriptor {
    pub buffer_offset: vk::DeviceSize,
}
