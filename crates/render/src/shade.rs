use ash::vk;
use eyre::Result;

use super::hal;
use crate::render_targets::RenderTargets;
use crate::scene::Scene;

pub(super) struct ShadePhase {
    pipeline: hal::Pipeline,
    descriptor_layout: hal::DescriptorLayout,
}

impl ShadePhase {
    pub fn new(
        device: &hal::Device,
        scene: &Scene,
        render_targets: &RenderTargets,
    ) -> Result<Self> {
        let descriptor_layout = hal::DescriptorLayoutBuilder::default()
            // Constants.
            .binding(vk::DescriptorType::UNIFORM_BUFFER)
            // Scene buffers.
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            .binding(vk::DescriptorType::STORAGE_BUFFER)
            // Visibility and Depth.
            .binding(vk::DescriptorType::STORAGE_IMAGE)
            .binding(vk::DescriptorType::STORAGE_IMAGE)
            // G-Buffers.
            .binding(vk::DescriptorType::STORAGE_IMAGE)
            .binding(vk::DescriptorType::STORAGE_IMAGE)
            .binding(vk::DescriptorType::STORAGE_IMAGE)
            // Color buffer.
            .binding(vk::DescriptorType::STORAGE_IMAGE)
            .build(device)?;
        let layout = hal::PipelineLayout {
            descriptors: &[&descriptor_layout],
            push_constant: None,
        };
        let shader = hal::Shader::new(
            device,
            &hal::ShaderRequest {
                stage: vk::ShaderStageFlags::COMPUTE,
                source: vk_shader_macros::include_glsl!(
                    "src/shaders/shade/shade.comp.glsl",
                    kind: comp,
                ),
            },
        )?;
        let specializations = hal::Specializations::default();
        let shader_stage = hal::ShaderStage {
            shader: &shader,
            specializations: &specializations,
        };
        let pipeline = hal::Pipeline::compute(device, &layout, shader_stage)?;
        Ok(Self { pipeline, descriptor_layout })
    }

    pub fn destroy(&self, device: &hal::Device) {
        self.descriptor_layout.destroy(device);
        self.pipeline.destroy(device);
    }
}
