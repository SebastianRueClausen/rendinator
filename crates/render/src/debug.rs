use std::slice;

use ash::vk;
use eyre::Result;

use crate::hal;

#[allow(dead_code)]
pub(super) fn download_buffer(
    device: &hal::Device,
    buffer: &hal::Buffer,
) -> Result<Vec<u8>> {
    let scratch = hal::Scratch::new(device, buffer.size)?;
    let byte_count = buffer.size as usize;

    hal::command::quickie(device, |command_buffer| {
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

#[allow(dead_code)]
pub(super) fn display_texture<'a>(
    device: &hal::Device,
    command_buffer: &mut hal::CommandBuffer<'a>,
    swapchain_image: &'a hal::Image,
    image: &'a hal::Image,
    mip_level: u32,
) {
    let swapchain_layout = command_buffer.image_layout(swapchain_image);
    let image_layout = command_buffer.image_layout(image);

    command_buffer
        .pipeline_barriers(
            device,
            &[
                hal::ImageBarrier {
                    image: swapchain_image,
                    new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    mip_levels: hal::MipLevels::All,
                    src: hal::Access::ALL,
                    dst: hal::Access::ALL,
                },
                hal::ImageBarrier {
                    image,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    mip_levels: hal::MipLevels::Levels {
                        base: mip_level,
                        count: 1,
                    },
                    src: hal::Access::ALL,
                    dst: hal::Access::ALL,
                },
            ],
            &[],
        )
        .blit_image(
            device,
            &hal::command::ImageBlit {
                src: image,
                dst: swapchain_image,
                src_offsets: image.spanning_offsets(mip_level),
                dst_offsets: swapchain_image.spanning_offsets(0),
                src_mip_level: mip_level,
                dst_mip_level: 0,
                filter: vk::Filter::LINEAR,
            },
        )
        .pipeline_barriers(
            device,
            &[
                hal::ImageBarrier {
                    image: swapchain_image,
                    new_layout: swapchain_layout,
                    mip_levels: hal::MipLevels::All,
                    src: hal::Access::ALL,
                    dst: hal::Access::ALL,
                },
                hal::ImageBarrier {
                    image: image,
                    new_layout: image_layout,
                    mip_levels: hal::MipLevels::Levels {
                        base: mip_level,
                        count: 1,
                    },
                    src: hal::Access::ALL,
                    dst: hal::Access::ALL,
                },
            ],
            &[],
        );
}
