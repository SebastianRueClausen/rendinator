#![warn(clippy::all)]

pub mod bump;
pub mod res;
pub mod slab;
pub mod dummy;

pub use res::Res;
pub use slab::Slabs;
pub use bump::Bump;
pub use dummy::DummyRes;
