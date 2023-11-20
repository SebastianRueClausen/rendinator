use ash::vk;

use crate::command::{self, Access, CommandBuffer, ImageBarrier};
use crate::device::Device;
use crate::resources::Image;

pub(super) fn display_texture<'a>(
    device: &Device,
    command_buffer: &mut CommandBuffer<'a>,
    swapchain_image: &'a Image,
    image: &'a Image,
) {
    let prev_swapchain_layout = command_buffer.image_layout(swapchain_image);
    let prev_image_layout = command_buffer.image_layout(image);

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
                    new_layout: prev_swapchain_layout,
                    src: Access::ALL,
                    dst: Access::ALL,
                },
                ImageBarrier {
                    image: image,
                    new_layout: prev_image_layout,
                    src: Access::ALL,
                    dst: Access::ALL,
                },
            ],
        );
}
