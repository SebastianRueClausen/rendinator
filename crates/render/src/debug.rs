use std::slice;

use ash::vk;
use eyre::Result;

use crate::command::{self, Access, CommandBuffer, ImageBarrier};
use crate::device::Device;
use crate::resources::{Buffer, Image, Scratch};

pub(super) fn download_buffer(
    device: &Device,
    buffer: &Buffer,
) -> Result<Vec<u8>> {
    let scratch = Scratch::new(device, buffer.size)?;
    let byte_count = buffer.size as usize;

    command::quickie(device, |command_buffer| {
        let region = vk::BufferCopy::builder()
            .size(buffer.size)
            .src_offset(0)
            .dst_offset(0)
            .build();
        unsafe {
            device.cmd_copy_buffer(
                **command_buffer,
                **buffer,
                *scratch.buffer,
                slice::from_ref(&region),
            );
        }
        Ok(())
    })?;

    let bytes = unsafe {
        slice::from_raw_parts(scratch.memory.map(device)?, byte_count).to_vec()
    };

    scratch.memory.unmap(device);
    scratch.destroy(device);

    Ok(bytes)
}

pub(super) fn display_texture<'a>(
    device: &Device,
    command_buffer: &mut CommandBuffer<'a>,
    swapchain_image: &'a Image,
    image: &'a Image,
) {
    let swapchain_layout = command_buffer.image_layout(swapchain_image);
    let image_layout = command_buffer.image_layout(image);

    command_buffer
        .pipeline_barriers(
            device,
            &[
                ImageBarrier {
                    image: swapchain_image,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src: Access::ALL,
                    dst: Access::ALL,
                },
                ImageBarrier {
                    image,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    src: Access::ALL,
                    dst: Access::ALL,
                },
            ],
        )
        .blit_image(
            device,
            &command::ImageBlit {
                src: image,
                dst: swapchain_image,
                src_offsets: image.spanning_offsets(),
                dst_offsets: swapchain_image.spanning_offsets(),
                src_mip_level: 0,
                dst_mip_level: 0,
                filter: vk::Filter::LINEAR,
            },
        )
        .pipeline_barriers(
            device,
            &[
                ImageBarrier {
                    image: swapchain_image,
                    new_layout: swapchain_layout,
                    src: Access::ALL,
                    dst: Access::ALL,
                },
                ImageBarrier {
                    image: image,
                    new_layout: image_layout,
                    src: Access::ALL,
                    dst: Access::ALL,
                },
            ],
        );
}
