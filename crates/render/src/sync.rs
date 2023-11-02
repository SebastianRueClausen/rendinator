use ash::vk;
use eyre::{Context, Result};

use crate::device::Device;

pub(crate) struct Sync {
    pub acquire: vk::Semaphore,
    pub release: vk::Semaphore,
}

impl Sync {
    pub fn new(device: &Device) -> Result<Self> {
        Ok(Self {
            acquire: create_semaphore(device)?,
            release: create_semaphore(device)?,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_semaphore(self.acquire, None);
            device.destroy_semaphore(self.release, None);
        }
    }
}

fn create_semaphore(device: &Device) -> Result<vk::Semaphore> {
    let semaphore_info = vk::SemaphoreCreateInfo::builder().build();
    unsafe {
        device
            .create_semaphore(&semaphore_info, None)
            .wrap_err("failed to create semaphore")
    }
}
