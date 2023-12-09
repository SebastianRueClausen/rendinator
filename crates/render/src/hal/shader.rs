use std::ffi::CString;
use std::ops::Deref;
use std::slice;

use ash::vk::{self};
use eyre::{Context, Result};

use super::{DescriptorLayout, Device};

pub struct ShaderRequest<'a> {
    pub source: &'a [u32],
    pub stage: vk::ShaderStageFlags,
}

pub struct Shader {
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

#[derive(Default)]
pub struct PipelineLayout<'a> {
    pub descriptors: &'a [&'a DescriptorLayout],
    pub push_constant: Option<vk::PushConstantRange>,
}

fn create_pipeline_layout(
    device: &Device,
    layout: &PipelineLayout,
) -> Result<(vk::PipelineLayout, vk::ShaderStageFlags)> {
    let set_layouts: Vec<vk::DescriptorSetLayout> =
        layout.descriptors.iter().map(|layout| ***layout).collect();
    let mut layout_info =
        vk::PipelineLayoutCreateInfo::builder().set_layouts(&set_layouts);
    let push_constant_stages =
        if let Some(push_constant) = &layout.push_constant {
            layout_info = layout_info
                .push_constant_ranges(slice::from_ref(push_constant));
            push_constant.stage_flags
        } else {
            vk::ShaderStageFlags::default()
        };
    let pipeline_layout = unsafe {
        device
            .create_pipeline_layout(&layout_info, None)
            .wrap_err("failed to create pipeline layout")?
    };
    Ok((pipeline_layout, push_constant_stages))
}

fn create_shader_info(
    shader: &Shader,
    entry_point: &CString,
    specialization_info: &vk::SpecializationInfo,
) -> vk::PipelineShaderStageCreateInfo {
    vk::PipelineShaderStageCreateInfo::builder()
        .specialization_info(specialization_info)
        .stage(shader.stage)
        .module(**shader)
        .name(&entry_point)
        .build()
}

pub struct Pipeline {
    pipeline: vk::Pipeline,
    pub layout: vk::PipelineLayout,
    pub push_constant_stages: vk::ShaderStageFlags,
    pub bind_point: vk::PipelineBindPoint,
}

impl Deref for Pipeline {
    type Target = vk::Pipeline;

    fn deref(&self) -> &Self::Target {
        &self.pipeline
    }
}

#[derive(Default)]
pub struct Specializations {
    entries: Vec<vk::SpecializationMapEntry>,
    data: Vec<u8>,
}

impl Specializations {
    pub fn entry(mut self, data: &[u8]) -> Self {
        let entry = vk::SpecializationMapEntry::builder()
            .constant_id(self.entries.len() as u32)
            .offset(self.data.len() as u32)
            .size(data.len())
            .build();
        self.entries.push(entry);
        self.data.extend_from_slice(data);
        self
    }
}

pub struct ShaderStage<'a> {
    pub specializations: &'a Specializations,
    pub shader: &'a Shader,
}

pub struct GraphicsPipelineRequest<'a> {
    pub color_formats: &'a [vk::Format],
    pub depth_format: Option<vk::Format>,
    pub shaders: &'a [ShaderStage<'a>],
    pub cull_mode: vk::CullModeFlags,
}

impl Pipeline {
    pub fn compute(
        device: &Device,
        layout: &PipelineLayout,
        shader: ShaderStage,
    ) -> Result<Self> {
        let (pipeline_layout, push_constant_stages) =
            create_pipeline_layout(device, layout)?;
        let entry_point = CString::new("main").unwrap();
        let specialization_info = vk::SpecializationInfo::builder()
            .data(&shader.specializations.data)
            .map_entries(&shader.specializations.entries)
            .build();
        let pipeline_info = vk::ComputePipelineCreateInfo::builder()
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .stage(create_shader_info(
                shader.shader,
                &entry_point,
                &specialization_info,
            ))
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
            push_constant_stages,
            layout: pipeline_layout,
        })
    }

    pub fn graphics<'a>(
        device: &Device,
        layout: &PipelineLayout,
        request: &GraphicsPipelineRequest,
    ) -> Result<Self> {
        let (pipeline_layout, push_constant_stages) =
            create_pipeline_layout(device, layout)?;
        let entry_point = CString::new("main").unwrap();
        let specialization_infos: Vec<_> = request
            .shaders
            .iter()
            .map(|shader| {
                vk::SpecializationInfo::builder()
                    .data(&shader.specializations.data)
                    .map_entries(&shader.specializations.entries)
                    .build()
            })
            .collect();
        let shader_infos: Vec<_> = request
            .shaders
            .into_iter()
            .zip(specialization_infos.iter())
            .map(|(shader, specialization)| {
                create_shader_info(shader.shader, &entry_point, specialization)
            })
            .collect();
        let mut rendering_info = vk::PipelineRenderingCreateInfo::builder()
            .depth_attachment_format(request.depth_format.unwrap_or_default())
            .color_attachment_formats(request.color_formats)
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
                .cull_mode(request.cull_mode);
        let multisampled_info =
            vk::PipelineMultisampleStateCreateInfo::builder()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        let depth_stencil_info =
            vk::PipelineDepthStencilStateCreateInfo::builder()
                .depth_compare_op(vk::CompareOp::GREATER)
                .depth_test_enable(request.depth_format.is_some())
                .depth_write_enable(request.depth_format.is_some());
        let color_attachment_state =
            vk::PipelineColorBlendAttachmentState::builder()
                .color_write_mask(vk::ColorComponentFlags::RGBA)
                .blend_enable(true)
                .src_color_blend_factor(vk::BlendFactor::ONE)
                .dst_color_blend_factor(vk::BlendFactor::ONE_MINUS_SRC_ALPHA);
        let color_blend_state =
            vk::PipelineColorBlendStateCreateInfo::builder()
                .attachments(slice::from_ref(&color_attachment_state));
        let dynamic_states =
            [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::builder()
            .dynamic_states(&dynamic_states);
        let pipeline_info = vk::GraphicsPipelineCreateInfo::builder()
            .flags(vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT)
            .layout(pipeline_layout)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_info)
            .viewport_state(&viewport_info)
            .rasterization_state(&rasterization_info)
            .multisample_state(&multisampled_info)
            .depth_stencil_state(&depth_stencil_info)
            .color_blend_state(&color_blend_state)
            .dynamic_state(&dynamic_state)
            .stages(&shader_infos)
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
            push_constant_stages,
            layout: pipeline_layout,
            pipeline,
        })
    }

    pub fn destroy(&self, device: &Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
        }
    }
}
