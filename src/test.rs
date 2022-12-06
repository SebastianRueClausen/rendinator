#![allow(dead_code)]

use anyhow::Result;
use shaderc::{CompileOptions, Compiler, ShaderKind, SpirvVersion};

fn compile_glsl(kind: ShaderKind, code: &str) -> Result<Vec<u32>> {
    let mut options = CompileOptions::new().unwrap();
    options.set_target_spirv(SpirvVersion::V1_6);
    Ok(Compiler::new()
        .unwrap()
        .compile_into_spirv(code, kind, "shader.glsl", "main", Some(&options))?
        .as_binary()
        .to_vec())
}

pub fn compile_comp_glsl(code: &str) -> Result<Vec<u32>> {
    compile_glsl(ShaderKind::Compute, code)
}

pub fn compile_vert_glsl(code: &str) -> Result<Vec<u32>> {
    compile_glsl(ShaderKind::Vertex, code)
}

pub fn compile_frag_glsl(code: &str) -> Result<Vec<u32>> {
    compile_glsl(ShaderKind::Fragment, code)
}
