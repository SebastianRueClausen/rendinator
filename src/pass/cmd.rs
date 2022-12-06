use super::{DescSet, GroupCount};
use crate::resource::{Buffer, Image, ImageLayout};
use rendi_res::Res;

use ash::vk::{self, DeviceSize};

#[derive(Debug)]
pub(super) enum Cmd {
    /// Bind descriptor sets.
    BindDescSets(Vec<Option<DescSet>>),
    /// Bind a new index buffer.
    BindIndexBuffer(Res<Buffer>),
    /// Bind a new vertex buffer.
    BindVertexBuffer(Res<Buffer>),
    /// Dispatch a compute shader.
    Dispatch(Res<GroupCount>),
    /// Draw command without index buffer.
    Draw(Draw),
    /// Draw command with index buffer.
    DrawIndexed(DrawIndexed),
    /// Draw command(s) from draw buffer without index buffer.
    DrawIndirect(DrawIndirect),
    /// Draw command(s) from draw buffer with index buffer.
    DrawIndexedIndirect(DrawIndirect),
    /// Draw command(s) from draw buffer without index buffer using draw count buffer.
    DrawIndirectCount(DrawIndirectCount),
    /// Draw command(s) from draw buffer with index buffer using draw count buffer.
    DrawIndexedIndirectCount(DrawIndirectCount),
    /// Buffer pipeline barrier.
    BufferBarrier(BufferBarrier),
    /// Image pipeline barrier.
    ImageBarrier(ImageBarrier),
}

/// Draw command of potentially multiple instances without an index buffer.
#[derive(Debug)]
pub struct Draw {
    /// The range into the bound vertex buffer to draw.
    pub vertex_range: DrawRange,
    /// The range of instances that should be drawn.
    pub instance_range: DrawRange,
}

/// Draw command of potentially multiple instances with an index buffer.
#[derive(Debug)]
pub struct DrawIndexed {
    /// The range into the index buffer to draw.
    pub index_range: DrawRange,
    /// The range of instances that should be drawn.
    pub instance_range: DrawRange,
    /// Offset added to each index in the index buffer.
    pub vertex_offset: u32,
}

/// Draw command(s) from `buffer`.
#[derive(Debug)]
pub struct DrawIndirect {
    /// The buffer of draw commands.
    pub buffer: Res<Buffer>,
    /// The byte offset into `buffer` of the first draw command.
    pub offset: DeviceSize,
    /// The amount draw commands to execute in `buffer`.
    pub draw_count: u32,
    /// The byte side skipped between draw commands.
    pub stride: u32,
}

/// Draw command(s) from `buffer` reading the count from `count_buffer`.
#[derive(Debug)]
pub struct DrawIndirectCount {
    /// Buffer containing the draw command count to execute.
    pub count_buffer: Res<Buffer>,
    /// The buffer of draw commands.
    pub buffer: Res<Buffer>,
    /// Offset of the count variable into `count_buffer`.
    pub count_offset: DeviceSize,
    /// The byte offset into `buffer` of the first draw command.
    pub offset: DeviceSize,
    /// A upper bound for draw commands.
    pub max_draw_count: u32,
    /// The byte side skipped between draw commands.
    pub stride: u32,
}

/// Buffer barrier for buffer access synchronization.
#[derive(Debug, Clone)]
pub struct BufferBarrier {
    pub buffer: Res<Buffer>,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
    pub src_stage: vk::PipelineStageFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
}

/// Image barrier for image access synchronization.
#[derive(Debug, Clone)]
pub struct ImageBarrier {
    pub image: Res<Image>,
    pub src_access: vk::AccessFlags2,
    pub dst_access: vk::AccessFlags2,
    pub src_stage: vk::PipelineStageFlags2,
    pub dst_stage: vk::PipelineStageFlags2,
    pub src_layout: ImageLayout,
    pub dst_layout: ImageLayout,
}

#[derive(Clone, Copy, Debug)]
pub struct DrawRange {
    start: u32,
    count: u32,
}
