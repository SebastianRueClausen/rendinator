#![warn(clippy::all)]
#![feature(unsize, coerce_unsized)]

pub mod bump;
pub mod dummy;
pub mod res;
pub mod slab;

pub use bump::Bump;
pub use dummy::DummyRes;
pub use res::Res;
pub use slab::Slabs;
