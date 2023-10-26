use std::ffi::CString;
use std::ops::Deref;
use std::slice;

use ash::vk;
use eyre::{Context, Result};

use crate::descriptor;
use crate::device::Device;

pub(crate) struct ShaderRequest<'a> {
    pub source: &'a [u32],
    pub stage: vk::ShaderStageFlags,
}

pub(crate) struct Shader {
    pub module: vk::ShaderModule,
    stage: vk::ShaderStageFlags,
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
        Ok(Self { module, stage: request.stage })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_shader_module(self.module, None);
        }
    }
}

pub(crate) struct PipelineLayout<'a> {
    pub bindings: &'a [descriptor::LayoutBinding],
    pub push_constant: Option<vk::PushConstantRange>,
}

fn create_pipeline_layout(
    device: &Device,
    layout: &PipelineLayout,
) -> Result<(vk::PipelineLayout, descriptor::Layout)> {
    let descriptor_layout = descriptor::Layout::new(device, layout.bindings)?;

    let mut layout_info = vk::PipelineLayoutCreateInfo::builder()
        .set_layouts(slice::from_ref(descriptor_layout.deref()));
    if let Some(push_constant) = &layout.push_constant {
        layout_info =
            layout_info.push_constant_ranges(slice::from_ref(push_constant));
    }

    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&layout_info, None)
            .wrap_err("failed to create pipeline layout")?
    };

    Ok((pipeline_layout, descriptor_layout))
}

fn create_shader_info(
    shader: &Shader,
    entry_point: &CString,
) -> vk::PipelineShaderStageCreateInfo {
    vk::PipelineShaderStageCreateInfo::builder()
        .stage(shader.stage)
        .module(**shader)
        .name(&entry_point)
        .build()
}

pub(crate) enum PipelineStage<'a> {
    Compute { shader: &'a Shader },
    Graphics {},
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
    pub fn compute(
        device: &Device,
        shader: &Shader,
        layout: &PipelineLayout,
    ) -> Result<Self> {
        let (pipeline_layout, descriptor_layout) =
            create_pipeline_layout(device, layout)?;
        let entry_point = CString::new("main").unwrap();
        let shader_info = create_shader_info(shader, &entry_point);
        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .stage(create_shader_info(shader, &entry_point))
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
        Ok(Self {
            bind_point: vk::PipelineBindPoint::COMPUTE,
            pipeline,
            pipeline_layout,
            descriptor_layout,
        })
    }

    pub fn graphics<'a>(
        device: &Device,
        layout: &PipelineLayout,
        color_formats: &[vk::Format],
        depth_format: vk::Format,
        shaders: impl IntoIterator<Item = &'a Shader>,
    ) -> Result<Self> {
        let (pipeline_layout, descriptor_layout) =
            create_pipeline_layout(device, layout)?;
        let entry_point = CString::new("main").unwrap();
        let shader_infos: Vec<_> = shaders
            .into_iter()
            .map(|shader| create_shader_info(shader, &entry_point))
            .collect();
        let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
            .color_attachment_formats(color_formats)
            .depth_attachment_format(depth_format)
            .build();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_info = vk::PipelineInputAssemblyStateCreateInfo::builder()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewport_info = vk::PipelineViewportStateCreateInfo::builder()
            .viewport_count(1)
            .scissor_count(1);
        let rasterization_info =
            vk::PipelineRasterizationStateCreateInfo::builder()
                .line_width(1.0)
                .front_face(vk::FrontFace::CLOCKWISE)
                .cull_mode(vk::CullModeFlags::BACK);
        let multisampled_info =
            vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil_info =
            vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(vk::CompareOp::GREATER)
                .depth_test_enable(true)
                .depth_write_enable(true);
        let color_attachment_state =
            vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::RGBA);
        let color_blend_state =
            vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(slice::from_ref(&color_attachment_state));
        let dynamic_states =
            [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .layout(pipeline_layout)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisampled_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .push_next(&mut rendering_info);
        let pipeline = unsafe {
            *device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[pipeline_info.build()],
                    None,
                )
                .map_err(|(_, err)| err)
                .wrap_err("failed to create graphics pipeline")?
                .first()
                .unwrap()
        };
        Ok(Self {
            bind_point: vk::PipelineBindPoint::GRAPHICS,
            pipeline,
            pipeline_layout,
            descriptor_layout,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            self.descriptor_layout.destroy(device);
        }
    }
}
