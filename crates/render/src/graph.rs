use std::collections::HashMap;

use crate::hal;

pub struct Graph<'a> {
    buffers: HashMap<&'a hal::Buffer, hal::Access>,
    command_buffer: hal::CommandBuffer<'a>,
    pending_image_barriers: Vec<hal::ImageBarrier<'a>>,
    pending_buffer_barriers: Vec<hal::BufferBarrier<'a>>,
}

impl<'a> Graph<'a> {
    pub fn access_buffer(
        &mut self,
        buffer: &'a hal::Buffer,
        access: hal::Access,
    ) -> &mut Self {
        let current_access = self
            .buffers
            .entry(buffer)
            .and_modify(|prev_access| {
                if prev_access.writes() || access.writes() {
                    self.pending_buffer_barriers.push(hal::BufferBarrier {
                        buffer,
                        src: *prev_access,
                        dst: access,
                    });
                    *prev_access = hal::Access::default();
                }
            })
            .or_default();
        *current_access = *current_access | access;
        self
    }

    pub fn pass(
        &mut self,
        device: &hal::Device,
        f: impl FnOnce(&mut hal::CommandBuffer),
    ) -> &mut Self {
        self.command_buffer.pipeline_barriers(
            device,
            &self.pending_image_barriers,
            &self.pending_buffer_barriers,
        );
        self.pending_image_barriers.clear();
        self.pending_buffer_barriers.clear();
        f(&mut self.command_buffer);
        self
    }
}
