use anyhow::{anyhow, Result};

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescKind {
    UniformBuffer,
    StorageBuffer,
    SampledImage,
    StorageImage,
}

impl fmt::Display for DescKind {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            DescKind::UniformBuffer => "uniform buffer",
            DescKind::StorageBuffer => "storage buffer",
            DescKind::SampledImage => "sampled image",
            DescKind::StorageImage => "storage image",
        };

        write!(fmt, "{name}")
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescCount {
    Single,
    Bound(u32),
    Unbound,
}

impl fmt::Display for DescCount {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DescCount::Single => write!(fmt, "single"),
            DescCount::Bound(count) => write!(fmt, "array length {count}"),
            DescCount::Unbound => write!(fmt, "unbound array"),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindSlot {
    set: u32,
    binding: u32,
}

impl fmt::Display for BindSlot {
    fn fmt(&self, fmt: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(fmt, "({}, {})", self.set, self.binding)
    }
}

impl BindSlot {
    pub fn new(set: u32, binding: u32) -> Self {
        Self { set, binding }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescBind {
    kind: DescKind,
    count: DescCount,
}

#[repr(transparent)]
#[derive(Clone, Copy)]
struct Ins(u32);

impl Ins {
    #[inline]
    fn word_count(self) -> usize {
        (self.0 >> 16) as usize
    }

    #[inline]
    fn opcode(self) -> u32 {
        self.0 & 0xffff
    }
}
 
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecoKind {
    Binding,
    DescSet,
}

impl DecoKind {
    fn from_ins(val: u32) -> Option<Self> {
        let kind = match val {
            33 => DecoKind::Binding,
            34 => DecoKind::DescSet,
            _ => return None,
        };

        Some(kind)
    }
}

#[derive(Clone, Copy, Debug)]
struct Deco<'a> {
    id: u32, 
    kind: DecoKind,
    args: &'a [u32]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StorageClass {
    Uniform,
    UniformConstant,
    StorageBuffer,
}

impl StorageClass {
    fn from_ins(val: u32) -> Option<Self> {
        let kind = match val {
            0 => StorageClass::UniformConstant,
            2 => StorageClass::Uniform,
            12 => StorageClass::StorageBuffer,
            _ => return None,
        };

        Some(kind)
    }
}

#[derive(Clone, Copy, Debug)]
struct Var {
    id: u32,
    ty: u32,
    storage: StorageClass,
}

#[derive(Clone, Copy, Debug)]
struct Const {
    id: u32,
    val: u32,
}

#[derive(Clone, Copy, Debug)]
enum TypeKind {
    Image {
        dim: ImageDim,
        samples: ImageSamples,
    },
    ImageSampled {
        ty_id: u32,
    },
    Pointer {
        ty_id: u32,
    },
    Array {
        ty_id: u32,
        len_id: u32,
    }
}

#[derive(Clone, Copy, Debug)]
struct Type {
    id: u32,
    kind: TypeKind,
}

#[derive(Clone, Copy, Debug)]
enum ImageDim {
    D2,
    Cube,
    Buffer,
}

impl ImageDim {
    fn from_ins(val: u32) -> Option<Self> {
        let val = match val {
            1 => ImageDim::D2,
            3 => ImageDim::Cube,
            5 => ImageDim::Buffer,
            _ => return None,
        };

        Some(val)
    }
}

#[derive(Clone, Copy, Debug)]
enum ImageSamples {
    Single,
    Multi,
}

impl ImageSamples {
    fn from_ins(val: u32) -> Option<Self> {
        let val = match val {
            0 => ImageSamples::Single,
            1 => ImageSamples::Multi,
            _ => return None,
        };

        Some(val)
    }
}

/// Code and reflections for a compute shader.
pub struct ComputeProg<'a> {
    code: &'a [u32],
    desc_binds: HashMap<BindSlot, DescBind>,
}

impl<'a> ComputeProg<'a> {
    /// Create new source code for 
    pub fn new(code: &'a [u32]) -> Result<Self> {
        let desc_binds =  get_desc_binds(&code)?;

        Ok(Self { desc_binds, code })
    }

    /// Get a slice of the raw SPIR-V code of the shader.
    pub fn code(&self) -> &[u32] {
        &self.code
    }

    /// Get descriptor binding at a given `slot`.
    ///
    ///If there is no descriptor binding at `slot`, `None` is returned.
    pub fn desc_bind(&self, slot: BindSlot) -> Option<DescBind> {
        self.desc_binds.get(&slot).cloned()
    }
}

pub struct RasterProg<'a> {
    frag_code: &'a [u32],
    vert_code: &'a [u32],

    desc_binds: HashMap<BindSlot, DescBind>,
}

impl<'a> RasterProg<'a> {
    pub fn new(frag_code: &'a [u32], vert_code: &'a [u32]) -> Result<Self> {
        let frag_binds = get_desc_binds(frag_code)?;
        let mut vert_binds = get_desc_binds(vert_code)?; 

        // Go through each binding in the fragment shader and add it to the vertex bindings if not
        // present alreadt. If it is present, then make sure they match.
        for (slot, frag_bind) in frag_binds.into_iter() {
            let Some(vert_bind) = vert_binds.get(&slot) else {
                vert_binds.insert(slot, frag_bind.clone());
                continue;
            };

            if vert_bind.kind != frag_bind.kind {
                return Err(anyhow!(
                    "descriptor type at {slot} doesn't match in vertex and fragment shader, {} vs {}",
                    vert_bind.kind,
                    frag_bind.kind,
                )); 
            }

            if vert_bind.count != frag_bind.count {
                return Err(anyhow!(
                    "descriptor count at {slot} doesn't match in vertex and fragment shader, {} vs {}",
                    vert_bind.count,
                    frag_bind.count,
                )); 
            }
        }

        Ok(Self { desc_binds: vert_binds, frag_code, vert_code })
    }

    /// Get a slice of the raw SPIR-V code of the fragment shader.
    pub fn frag_code(&self) -> &[u32] {
        &self.frag_code
    }

    /// Get a slice of the raw SPIR-V code of the vertex shader.
    pub fn vert_code(&self) -> &[u32] {
        &self.vert_code
    }

    /// Get descriptor binding at a given `slot`.
    ///
    ///If there is no descriptor binding at `slot`, `None` is returned.
    pub fn desc_bind(&self, slot: BindSlot) -> Option<DescBind> {
        self.desc_binds.get(&slot).cloned()
    }
}

fn get_desc_binds(binary: &[u32]) -> Result<HashMap<BindSlot, DescBind>> {
    if binary.get(0).cloned() != Some(SPIRV_MAGIC_VALUE) {
        return Err(anyhow!("invalid magic value"));
    }

    let mut inss = &binary[5..];

    let mut decos = Vec::new();
    let mut vars = Vec::new();
    let mut consts = Vec::new();
    let mut types = Vec::new();

    while let Some(ins) = inss.first().map(|w| Ins(*w)) {
        macro_rules! eof {
            () => {
                println!("{}", std::line!());
                return Err(anyhow!("unexpected end of file"));
            }
        }

        macro_rules! skip {
            () => {
                (_, inss) = inss.split_at(ins.word_count());
                continue;
            }
        }

        let Some(bytes) = inss.get(..ins.word_count()) else {
            eof!();
        };

        match ins.opcode() {
            OP_CONST => {
                let Some(id) = bytes.get(2).cloned() else {
                    eof!();
                };

                let Some(val) = bytes.get(3).cloned() else {
                    eof!();
                };

                consts.push(Const { id, val });
            }
            OP_DECO => {
                let Some(id) = bytes.get(1).cloned() else {
                    eof!();
                };

                let Some(kind) = bytes.get(2).cloned() else {
                    eof!();
                };

                let Some(kind) = DecoKind::from_ins(kind) else {
                    skip!();
                };

                let Some(args) = bytes.get(3..) else {
                    eof!();
                };

                decos.push(Deco { id, kind, args });
            }
            OP_VAR => {
                let Some(ty) = bytes.get(1).cloned() else {
                    eof!();
                };

                let Some(id) = bytes.get(2).cloned() else {
                    eof!();
                };

                let Some(storage) = bytes.get(3).cloned() else {
                    eof!();
                };
        
                let Some(storage) = StorageClass::from_ins(storage) else {
                    skip!();
                };

                vars.push(Var { id, storage, ty });
            }
            OP_TYPE_IMAGE => {
                let Some(id) = bytes.get(1).cloned() else {
                    eof!();
                };

                let Some(dim) = bytes.get(3).cloned() else {
                    eof!();
                };

                let Some(dim) = ImageDim::from_ins(dim) else {
                    skip!();
                };

                let Some(samples) = bytes.get(6).cloned() else {
                    eof!();
                };

                let Some(samples) = ImageSamples::from_ins(samples) else {
                    skip!();
                };
    
                types.push(Type {
                    kind: TypeKind::Image { dim, samples },
                    id,
                });
            }
            OP_TYPE_SAMPLED_IMAGE => {
                let Some(id) = bytes.get(1).cloned() else {
                    eof!();
                };

                let Some(ty_id) = bytes.get(2).cloned() else {
                    eof!();
                };

                types.push(Type {
                    kind: TypeKind::ImageSampled { ty_id },
                    id, 
                });

            }
            OP_TYPE_POINTER => {
                let Some(id) = bytes.get(1).cloned() else {
                    eof!();
                };

                let Some(ty_id) = bytes.get(3).cloned() else {
                    eof!();
                };

                types.push(Type {
                    kind: TypeKind::Pointer { ty_id },
                    id, 
                });
            }
            OP_TYPE_ARRAY => {
                let Some(id) = bytes.get(1).cloned() else {
                    eof!();
                };

                let Some(ty_id) = bytes.get(2).cloned() else {
                    eof!();
                };

                let Some(len_id) = bytes.get(3).cloned() else {
                    eof!();
                };

                types.push(Type {
                    kind: TypeKind::Array { ty_id, len_id },
                    id,
                });
            }
            _ => ()
        }

        skip!();
    }

    let descs = vars
        .iter()
        .filter_map(|var| {
            let set = decos.iter().find(|deco| {
                deco.id == var.id && deco.kind == DecoKind::DescSet
            })?;

            let binding = decos.iter().find(|deco| {
                deco.id == var.id && deco.kind == DecoKind::Binding
            })?;
              
            let set = set.args.first().cloned()?;
            let binding = binding.args.first().cloned()?;

            let (kind, count) = match var.storage {
                StorageClass::Uniform => {
                    (DescKind::UniformBuffer, DescCount::Single)
                },
                StorageClass::StorageBuffer => {
                    (DescKind::StorageBuffer, DescCount::Single)
                }
                StorageClass::UniformConstant => {
                    let mut ty = types.iter().find(|ty| ty.id == var.ty)?;
                    let mut count = DescCount::Single;

                    let kind = loop {
                        match ty.kind {
                            TypeKind::Image { .. } => {
                                break DescKind::StorageImage;
                            }
                            TypeKind::ImageSampled { .. } => {
                                break DescKind::SampledImage;
                            }
                            TypeKind::Pointer { ty_id } => {
                                ty = types.iter().find(|ty| ty.id == ty_id)?;
                            }
                            TypeKind::Array { ty_id, len_id } => {
                                ty = types.iter().find(|ty| ty.id == ty_id)?;
                                
                                let len = consts.iter().find(|c| c.id == len_id)?;

                                // For some reason unbound descriptor arrays are marked by having
                                // the length of 1 in spirv from shaderc. This is not mentioned in
                                // the spirv spec.
                                if len.val == 1 {
                                    count = DescCount::Unbound;
                                } else {
                                    count = DescCount::Bound(len.val);
                                }
                            }
                        }
                    };

                    (kind, count)
                }
            };

            let slot = BindSlot::new(set, binding);
            let bind = DescBind { kind, count };

            Some((slot, bind))
        })
        .collect();

    Ok(descs)
}

const SPIRV_MAGIC_VALUE: u32 = 0x07230203;

const OP_CONST: u32 = 43;
const OP_VAR: u32 = 59;
const OP_DECO: u32 = 71;

const OP_TYPE_IMAGE: u32 = 25;
const OP_TYPE_SAMPLED_IMAGE: u32 = 27;
const OP_TYPE_ARRAY: u32 = 28;
const OP_TYPE_POINTER: u32 = 32;

#[cfg(test)]
mod test {
    use super::*;
    use shaderc::{Compiler, ShaderKind, CompileOptions, SpirvVersion};

    fn compile(kind: ShaderKind, src: &str) -> Vec<u32> {
        let mut options = CompileOptions::new().unwrap();
        options.set_target_spirv(SpirvVersion::V1_6);
        Compiler::new()
            .unwrap()
            .compile_into_spirv(src, kind, "shader.glsl", "main", Some(&options))
            .unwrap()
            .as_binary()
            .to_vec()
    }

    #[test]
    fn buffer_desc_slot() {
        let spv = compile(ShaderKind::Compute, r#"
            #version 450

            layout (set = 0, binding = 0) uniform Block1 {
                int val1;
                int val2;
            };

            layout (std430, set = 0, binding = 1) buffer Block2 {
                int val3;
            };

            void main() {}
        "#);

        let slots = get_desc_binds(&spv).unwrap();

        assert_eq!(slots[&BindSlot::new(0, 0)], DescBind {
            kind: DescKind::UniformBuffer, 
            count: DescCount::Single,
        });

        assert_eq!(slots[&BindSlot::new(0, 1)], DescBind {
            kind: DescKind::StorageBuffer, 
            count: DescCount::Single,
        });
    }

    #[test]
    fn image_desc_slot() {
        let spv = compile(ShaderKind::Compute, r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;
            layout (set = 0, binding = 1, r32f) writeonly uniform image2D val2;

            void main() {}
        "#);

        let slots = get_desc_binds(&spv).unwrap();

        assert_eq!(slots[&BindSlot::new(0, 0)], DescBind {
            kind: DescKind::SampledImage, 
            count: DescCount::Single,
        });

        assert_eq!(slots[&BindSlot::new(0, 1)], DescBind {
            kind: DescKind::StorageImage, 
            count: DescCount::Single,
        });
    }

    #[test]
    fn combined_image_buffer_desc_slot() {
        let spv = compile(ShaderKind::Compute, r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;
            layout (set = 1, binding = 0) writeonly uniform Block {
                int val2;
            };

            void main() {}
        "#);

        let slots = get_desc_binds(&spv).unwrap();

        assert_eq!(slots[&BindSlot::new(0, 0)], DescBind {
            kind: DescKind::SampledImage, 
            count: DescCount::Single,
        });

        assert_eq!(slots[&BindSlot::new(1, 0)], DescBind {
            kind: DescKind::UniformBuffer, 
            count: DescCount::Single,
        });
    }

    #[test]
    fn image_array() {
        let spv = compile(ShaderKind::Compute, r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1[];
            layout (set = 0, binding = 1) uniform sampler2D val2[2];

            void main() {}
        "#);

        let slots = get_desc_binds(&spv).unwrap();

        assert_eq!(slots[&BindSlot::new(0, 0)], DescBind {
            kind: DescKind::SampledImage, 
            count: DescCount::Unbound,
        });

        assert_eq!(slots[&BindSlot::new(0, 1)], DescBind {
            kind: DescKind::SampledImage, 
            count: DescCount::Bound(2),
        });
    }

    #[test]
    fn raster_different_types() {
        let frag = compile(ShaderKind::Fragment, r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;

            void main() {}
        "#);

        let vert = compile(ShaderKind::Vertex, r#"
            #version 450

            layout (set = 0, binding = 0) writeonly uniform image2D val1;

            void main() {}
        "#);

        let prog = RasterProg::new(&frag, &vert);

        assert!(prog.is_err());
    }

    #[test]
    fn raster_different_count() {
        let frag = compile(ShaderKind::Fragment, r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1[];

            void main() {}
        "#);

        let vert = compile(ShaderKind::Vertex, r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;

            void main() {}
        "#);

        let prog = RasterProg::new(&frag, &vert);

        assert!(prog.is_err());
    }

    #[test]
    fn raster_combine() {
        let frag = compile(ShaderKind::Fragment, r#"
            #version 450

            layout (set = 0, binding = 1) uniform sampler2D val1[];

            void main() {}
        "#);

        let vert = compile(ShaderKind::Vertex, r#"
            #version 450

            layout (set = 0, binding = 0) writeonly uniform image2D val1;

            void main() {}
        "#);

        let prog = RasterProg::new(&frag, &vert).unwrap();

        assert_eq!(prog.desc_bind(BindSlot::new(0, 1)), Some(DescBind {
            kind: DescKind::SampledImage,
            count: DescCount::Unbound,
        }));

        assert_eq!(prog.desc_bind(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::StorageImage,
            count: DescCount::Single,
        }));
    }
}
