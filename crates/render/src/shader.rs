use std::ffi::CString;
use std::ops::Deref;
use std::slice;

use ash::vk;
use eyre::{Context, Result};

use crate::descriptor;
use crate::device::Device;

pub(crate) struct ShaderRequest<'a> {
    pub source: &'a [u32],
}

pub(crate) struct Shader {
    pub module: vk::ShaderModule,
}

impl Deref for Shader {
    type Target = vk::ShaderModule;

    fn deref(&self) -> &Self::Target {
        &self.module
    }
}

impl Shader {
    pub fn new(device: &Device, request: &ShaderRequest) -> Result<Shader> {
        let shader_info =
            vk::ShaderModuleCreateInfo::builder().code(request.source);
        let module = unsafe {
            device
                .create_shader_module(&shader_info, None)
                .wrap_err("failed to create shader module")?
        };
        Ok(Self { module })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.module, None);
        }
    }
}

pub(crate) enum PipelineStage<'a> {
    Compute { shader: &'a Shader },
}

pub(crate) struct PipelineRequest<'a> {
    pub stage: PipelineStage<'a>,
    pub bindings: &'a [descriptor::LayoutBinding],
    pub push_constant_range: Option<vk::PushConstantRange>,
}

pub(crate) struct Pipeline {
    pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_layout: descriptor::Layout,
    pub bind_point: vk::PipelineBindPoint,
}

impl Deref for Pipeline {
    type Target = vk::Pipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

impl Pipeline {
    pub fn new(device: &Device, request: &PipelineRequest) -> Result<Self> {
        let descriptor_layout =
            descriptor::Layout::new(device, request.bindings)?;

        let mut layout_info = vk::PipelineLayoutCreateInfo::builder()
            .set_layouts(slice::from_ref(descriptor_layout.deref()));
        if let Some(push_constant_range) = &request.push_constant_range {
            layout_info = layout_info
                .push_constant_ranges(slice::from_ref(push_constant_range));
        }

        let pipeline_layout = unsafe {
            device
                .create_pipeline_layout(&layout_info, None)
                .wrap_err("failed to create pipeline layout")?
        };

        let entry_point = CString::new("main").unwrap();

        let (pipeline, bind_point) = match request.stage {
            PipelineStage::Compute { shader } => {
                let stage = vk::PipelineShaderStageCreateInfo::builder()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(*shader.deref())
                    .name(&entry_point)
                    .build();
                let pipeline_info = vk::ComputePipelineCreateInfo::builder()
                    .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
                    .stage(stage)
                    .layout(pipeline_layout);
                let pipeline = unsafe {
                    *device
                        .create_compute_pipelines(
                            vk::PipelineCache::null(),
                            slice::from_ref(&pipeline_info),
                            None,
                        )
                        .map_err(|(_, err)| err)
                        .wrap_err("failed to create compute pipeline")?
                        .first()
                        .unwrap()
                };

                (pipeline, vk::PipelineBindPoint::COMPUTE)
            }
        };

        Ok(Self { pipeline, pipeline_layout, descriptor_layout, bind_point })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.descriptor_layout.destroy(device);
        }
    }
}
