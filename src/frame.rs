use anyhow::Result;
use std::{ops, array, slice};

pub const FRAMES_IN_FLIGHT: usize = 2;

/// THe index of a frame in flight.
#[derive(Default, Clone, Copy)]
pub struct FrameIndex {
    index: u8,
}

impl FrameIndex {
    /// Enumerate all frame indices.
    pub fn enumerate() -> impl Iterator<Item = Self> {
        (0..FRAMES_IN_FLIGHT as u8).map(|index| Self {
            index
        })
    }

    /// The previous frame index.
    pub fn prev(self) -> Self {
        let index = self.index.wrapping_sub(1) % FRAMES_IN_FLIGHT as u8;

        Self {
            index,
        }
    }

    /// The next frame index.
    pub fn next(self) -> Self {
        let index = self.index.wrapping_add(1) % FRAMES_IN_FLIGHT as u8;

        Self {
            index,
        }
    }
}

/// A container for storing an element for each frame in flight.
pub struct PerFrame<T> {
    items: [T; FRAMES_IN_FLIGHT],
}

impl<T> PerFrame<T> {
    /// Create a new `PerFrame` and initialize each element with `func`.
    pub fn from_fn<F>(func: F) -> Self
    where F: Fn(FrameIndex) -> T
    {
        let items = array::from_fn(|index| {
            let frame_index = FrameIndex { index: index as u8 };
            func(frame_index)
        });

        Self { items }
    }

    /// Create a new `PerFrame` and try to initialize each element with `func`.
    ///
    /// Returns `Err` if initializing any of the elements fail.
    pub fn try_from_fn<F>(func: F) -> Result<Self>
    where F: Fn(FrameIndex) -> Result<T>
    {
        let items = array::try_from_fn(|index| {
            let frame_index = FrameIndex { index: index as u8 };
            func(frame_index)
        })?;

        Ok(Self { items })
    }

    /// Iterate over all elements.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.items.iter()
    }

    /// Get an unspecified element from the array.
    pub fn any(&self) -> &T {
        &self[FrameIndex::default()]
    }
}

impl<T> ops::Index<FrameIndex> for PerFrame<T> {
    type Output = T;

    fn index(&self, index: FrameIndex) -> &Self::Output {
        &self.items[index.index as usize]
    }
}

impl<T> ops::IndexMut<FrameIndex> for PerFrame<T> {
    fn index_mut(&mut self, index: FrameIndex) -> &mut Self::Output {
        &mut self.items[index.index as usize]
    }
}

impl<T: Clone> Clone for PerFrame<T> {
    fn clone(&self) -> Self {
        Self { items: self.items.clone() }
    }
}

impl<T> Into<[T; FRAMES_IN_FLIGHT]> for PerFrame<T> {
    fn into(self) -> [T; FRAMES_IN_FLIGHT] {
        self.items
    }
}

impl<'a, T> IntoIterator for &'a PerFrame<T> {
    type Item = &'a T;
    type IntoIter = slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.items.as_slice().iter()
    }
}

