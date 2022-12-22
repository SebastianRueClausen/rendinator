use super::{Allocator, MemoryBlock, MemoryLayout, MemorySlice, MemoryType};
use crate::{Device, RenderError};
use ash::vk::DeviceSize;
use rendi_data_structs::SortedMap;
use rendi_res::Res;
use std::cell::Cell;

#[derive(Clone)]
struct BumpBlock {
    block: Res<MemoryBlock>,
    offset: Cell<DeviceSize>,
}

impl BumpBlock {
    fn new(block: Res<MemoryBlock>) -> Self {
        Self {
            block,
            offset: Cell::new(0),
        }
    }

    fn alloc(&self, layout: MemoryLayout) -> Option<MemorySlice> {
        let offset = super::align_up_to(self.offset.get(), layout.alignment());

        (layout.size() <= self.block.size().saturating_sub(offset)).then(|| unsafe {
            self.offset.set(offset);
            MemorySlice::new_unchecked(self.block.clone(), offset, offset + layout.size())
        })
    }
}

pub struct Bump {
    block_size: DeviceSize,
    blocks: SortedMap<MemoryType, BumpBlock>,
    device: Res<Device>,
}

impl Bump {
    #[must_use]
    pub fn new(device: Res<Device>, block_size: DeviceSize) -> Self {
        Self {
            blocks: SortedMap::new(),
            block_size,
            device,
        }
    }
}

impl Allocator for Bump {
    fn alloc(&mut self, layout: MemoryLayout, ty: MemoryType) -> Result<MemorySlice, RenderError> {
        #[allow(clippy::never_loop)]
        loop {
            let Some(slice) = self.blocks.get_mut(&ty).and_then(|bump_block| {
                bump_block.alloc(layout)
            }) else {
                self.blocks.insert(ty, {
                    let mut size = self.block_size;
                    while size < layout.size() {
                        size += self.block_size
                    }

                    let block = MemoryBlock::new(self.device.clone(), ty, size)?;
                    let block = Res::new_heap(block);

                    BumpBlock::new(block)
                });

                continue;
            };

            break Ok(slice);
        }
    }
}
