use anyhow::{anyhow, Result};

use std::collections::HashMap;
use std::fmt;

/// The kind of the descriptor.
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

/// This indicates the number of descriptor bound to a slot. 
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescCount {
    /// A single descriptor.
    Single,

    /// An bound array of descriptors.
    Bound(u32),

    /// An unbound array of descriptors.
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

/// The slot where a [`DescBind`] is defined.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BindSlot {
    set: u32,
    binding: u32,
}

impl BindSlot {
    pub fn set(&self) -> u32 {
        self.set
    }

    pub fn binding(&self) -> u32 {
        self.binding
    }
}

impl From<(u32, u32)> for BindSlot {
    fn from((set, binding): (u32, u32)) -> Self {
        Self { set, binding }
    }
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

/// The access type of the descriptor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DescAccess {
    ReadWrite,
    Write,
    Read,
}

impl DescAccess {
    fn from_rw_flags(non_readable: bool, non_writeable: bool) -> Result<Self> {
        let access = match (non_readable, non_writeable) {
            (true, false) => DescAccess::Write,
            (false, true) => DescAccess::Read,
            (false, false) => DescAccess::ReadWrite,
            (true, true) => return Err(anyhow!(
                "descriptor both non-readable and non-writeable")
            ),
        };

        Ok(access)
    }
}

/// A descriptor binding as defined in a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescBind {
    access: DescAccess,
    kind: DescKind,
    count: DescCount,
}

impl DescBind {
    pub fn kind(&self) -> DescKind {
        self.kind
    }

    pub fn count(&self) -> DescCount {
        self.count
    }
}

/// A push constant range of a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PushConstRange {
    /// The start offset of the push contant. This is the offset into the struct where the first
    /// variable is stored.
    offset: u32,

    /// The size of the range from the offset.
    size: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecoKind {
    Binding,
    DescSet,
    Offset,
    NonReadable,
    NonWriteable,
}

impl DecoKind {
    fn from_ins(val: u32) -> Option<Self> {
        let kind = match val {
            24 => DecoKind::NonWriteable,
            25 => DecoKind::NonReadable,
            33 => DecoKind::Binding,
            34 => DecoKind::DescSet,
            35 => DecoKind::Offset,
            _ => return None,
        };

        Some(kind)
    }
}

/// SPIR-V decoration. This is an attribute given to variables and types.
#[derive(Clone, Copy, Debug)]
struct Deco<'a> {
    id: u32, 
    kind: DecoKind,
    args: &'a [u32]
}

/// SPIR-V member decoration. This is an attribute given to struct members.
#[derive(Clone, Copy, Debug)]
struct MemberDeco<'a> {
    struct_ty_id: u32,
    kind: DecoKind,
    member: u32,
    args: &'a [u32],
}

/// The storage class of an variable.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StorageClass {
    /// This usually means a uniform buffer. However when targeting older SPIR-V versions, shaderc
    /// also gives storage buffers this class.
    Uniform,

    /// This usually means either image of sampled image.
    UniformConstant,

    StorageBuffer,
    PushConstant,
}

impl StorageClass {
    fn from_ins(val: u32) -> Option<Self> {
        let kind = match val {
            0 => StorageClass::UniformConstant,
            2 => StorageClass::Uniform,
            9 => StorageClass::PushConstant,
            12 => StorageClass::StorageBuffer,
            _ => return None,
        };

        Some(kind)
    }
}

/// SPIR-V variable.
#[derive(Clone, Copy, Debug)]
struct Var {
    id: u32,
    ty: u32,
    storage: StorageClass,
}

/// SPIR-V constant.
#[derive(Clone, Copy, Debug)]
struct Const {
    id: u32,

    /// The raw bits of the constant value.
    val: u32,
}

/// SPIR-V type kind.
#[derive(Clone, Copy, Debug)]
enum TypeKind<'a> {
    Image {
        #[allow(unused)]
        dim: ImageDim,

        #[allow(unused)]
        samples: ImageSamples,
    },
    ImageSampled {
        /// The type ID of the underlying image.
        #[allow(unused)]
        ty_id: u32,
    },
    Pointer {
        /// The type ID if the type being pointed at.
        ty_id: u32,
    },
    Array {
        /// The type ID of element type.
        ty_id: u32,

        /// ID of the length constant.
        len_id: u32,
    },
    Struct {
        /// ID's of each member.
        member_ids: &'a [u32], 
    },
    Prim {
        /// The size of the type in bytes.
        bytes: u32,
    },
    Vector {
        /// Type ID of the element type.
        ty_id: u32,

        /// The Cardinality of the vector.
        card: u32
    },
}

/// SPIR-V type.
#[derive(Clone, Copy, Debug)]
struct Type<'a> {
    id: u32,
    kind: TypeKind<'a>,
}

/// The image dimensionality as defined by the SPIR-V spec.
#[derive(Clone, Copy, Debug)]
enum ImageDim {
    /// 2D image.
    D2,

    /// Image usable as a cube map.
    Cube,
}

impl ImageDim {
    fn from_ins(val: u32) -> Option<Self> {
        let val = match val {
            1 => ImageDim::D2,
            3 => ImageDim::Cube,
            _ => return None,
        };

        Some(val)
    }
}

/// Indicates whether the image is single or multi-sampled.
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
    push_const: Option<PushConstRange>,
}

impl<'a> ComputeProg<'a> {
    /// Create new source code for 
    pub fn new(code: &'a [u32]) -> Result<Self> {
        let (desc_binds, push_const) = get_desc_binds(&code)?;

        Ok(Self { desc_binds, code, push_const })
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

    /// Get push constant range of the the compute shader.
    pub fn push_const(&self) -> Option<PushConstRange> {
        self.push_const
    }
}

/// Code and reflection for fragment and vertex shader pair.
pub struct RasterProg<'a> {
    frag_code: &'a [u32],
    vert_code: &'a [u32],
    desc_binds: HashMap<BindSlot, DescBind>,

    vert_push_const: Option<PushConstRange>, 
    frag_push_const: Option<PushConstRange>, 
}

impl<'a> RasterProg<'a> {
    pub fn new(frag_code: &'a [u32], vert_code: &'a [u32]) -> Result<Self> {
        let (frag_binds, frag_push_const) = get_desc_binds(frag_code)?;
        let (mut vert_binds, vert_push_const) = get_desc_binds(vert_code)?; 

        // Go through each binding in the fragment shader and add it to the vertex bindings if not
        // present already. If it's present, then make sure they match.
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

        Ok(Self {
            frag_push_const,
            vert_push_const,
            desc_binds: vert_binds,
            frag_code,
            vert_code,
        })
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

    /// Get the push constant range of the fragment shader.
    pub fn frag_push_const(&self) -> Option<PushConstRange> {
        self.frag_push_const
    }

    /// Get the push constant range of the vertex shader.
    pub fn vert_push_const(&self) -> Option<PushConstRange> {
        self.vert_push_const
    }
}

/// SPIR-V instruction.
#[derive(Clone, Copy)]
struct Ins<'a> {
    opcode: u16,
    args: &'a [u32], 
}

impl<'a> Ins<'a> {
    /// Get the opcode of the instruction.
    #[inline]
    fn opcode(self) -> u16 {
        self.opcode
    }

    /// Get argument (word) at `index` from the start of instruction (this includes the first
    /// word containing the opcode). If `index` is outside the range of arguments, an error will be
    /// returned.
    #[inline]
    fn expect_arg(self, index: usize) -> Result<u32> {
        self.args.get(index).cloned().ok_or_else(|| {
            anyhow!("unexpected end of instruction")
        })
    }

    /// Get the arguments of the instruction. This includes the first word.
    #[inline]
    fn args(&self) -> &'a [u32] {
        self.args
    }
}

/// Iterator of SPIR-V instructions.
struct InsIter<'a> {
    binary: &'a [u32],
}

impl<'a> InsIter<'a> {
    fn new(binary: &'a [u32]) -> Result<Self> {
        if binary.get(0).cloned() != Some(SPIRV_MAGIC_VALUE) {
            return Err(anyhow!("invalid magic value"));
        }

        Ok(Self { binary: &binary[5..] })
    }
}

impl<'a> Iterator for InsIter<'a> {
    type Item = Ins<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let ins = self.binary
            .first()
            .cloned()
            .map(|w| {
                let opcode = (w & 0xff) as u16;
                let word_count = (w >> 16) as usize;

                let (args, binary) = self.binary.split_at(word_count);
                self.binary = binary;

                Ins { opcode, args }
            })?;

        Some(ins)
    }
}

/// Reflection of shader.
///
/// This is not a complete reflection. It only holds the data we care about.
#[derive(Default)]
struct Reflection<'a> {
    decos: Vec<Deco<'a>>,
    member_decos: Vec<MemberDeco<'a>>,
    types: Vec<Type<'a>>,
    vars: Vec<Var>,
    consts: Vec<Const>,
}

impl<'a> Reflection<'a> {
    fn new(binary: &'a [u32]) -> Result<Self> {
        let mut inss = InsIter::new(binary)?;
        let mut reflection = Reflection::default();

        while let Some(ins) = inss.next() {
            match ins.opcode() {
                OP_CONST => {
                    // NOTE: For now we treat every constant as an 32-bit unsigned integer since
                    // we only use it for array length.

                    let id = ins.expect_arg(2)?;
                    let val = ins.expect_arg(3)?;

                    reflection.consts.push(Const { id, val });
                }
                OP_DECO => {
                    let id = ins.expect_arg(1)?;
                    let Some(kind) = DecoKind::from_ins(ins.expect_arg(2)?) else {
                        continue;
                    };

                    let args = ins.args().get(3..).unwrap_or_default();

                    reflection.decos.push(Deco { id, kind, args });
                }
                OP_MEMBER_DECO => {
                    let struct_ty_id = ins.expect_arg(1)?;
                    let member = ins.expect_arg(2)?;

                    let Some(kind) = DecoKind::from_ins(ins.expect_arg(3)?) else {
                        continue;
                    };

                    let args = ins.args().get(4..).unwrap_or_default();
    
                    reflection.member_decos.push(MemberDeco { struct_ty_id, member, args, kind });
                }
                OP_VAR => {
                    let ty = ins.expect_arg(1)?;
                    let id = ins.expect_arg(2)?;
                    let storage = ins.expect_arg(3)?;
            
                    let Some(storage) = StorageClass::from_ins(storage) else {
                        continue;
                    };

                    reflection.vars.push(Var { id, storage, ty });
                }
                OP_TYPE_IMAGE => {
                    let id = ins.expect_arg(1)?;

                    let Some(dim) = ImageDim::from_ins(ins.expect_arg(3)?) else {
                        continue;
                    };

                    let Some(samples) = ImageSamples::from_ins(ins.expect_arg(6)?) else {
                        continue;
                    };
        
                    reflection.types.push(Type {
                        kind: TypeKind::Image { dim, samples },
                        id,
                    });
                }
                OP_TYPE_SAMPLED_IMAGE => {
                    let id = ins.expect_arg(1)?;
                    let ty_id = ins.expect_arg(2)?;

                    reflection.types.push(Type {
                        kind: TypeKind::ImageSampled { ty_id },
                        id, 
                    });

                }
                OP_TYPE_POINTER => {
                    let id = ins.expect_arg(1)?;
                    let ty_id = ins.expect_arg(3)?;

                    reflection.types.push(Type {
                        kind: TypeKind::Pointer { ty_id },
                        id, 
                    });
                }
                OP_TYPE_STRUCT => {
                    let id = ins.expect_arg(1)?;
                    let member_ids = ins.args().get(2..).unwrap_or_default();

                    reflection.types.push(Type {
                        kind: TypeKind::Struct { member_ids },
                        id,
                    });
                }
                OP_TYPE_ARRAY => {
                    let id = ins.expect_arg(1)?;
                    let ty_id = ins.expect_arg(2)?;
                    let len_id = ins.expect_arg(3)?;

                    reflection.types.push(Type {
                        kind: TypeKind::Array { ty_id, len_id },
                        id,
                    });
                }
                OP_TYPE_INT | OP_TYPE_FLOAT => {
                    let id = ins.expect_arg(1)?;
                    let bits = ins.expect_arg(2)?;
                    let bytes = bits / 8;

                    reflection.types.push(Type {
                        kind: TypeKind::Prim { bytes },
                        id,
                    });
                }
                OP_TYPE_VECTOR => {
                    let id = ins.expect_arg(1)?;
                    let ty_id = ins.expect_arg(2)?;
                    let card = ins.expect_arg(3)?;

                    reflection.types.push(Type {
                        kind: TypeKind::Vector { card, ty_id },
                        id,
                    });
                }
                _ => ()
            }
        }

        Ok(reflection)
    }

    /// Find type with `id`.
    fn find_type(&self, id: u32) -> Option<Type<'a>> {
        self.types.iter().find(|ty| ty.id == id).copied()
    }

    /// Find constant with `id`. Returns `None` if no type has that id.
    fn find_const(&self, id: u32) -> Option<Const> {
        self.consts.iter().find(|c| c.id == id).copied()
    }

    /// Find the byte size of a type with `id`. Returns `None` if either the type doesn't exist or
    /// the type doesn't have a size such as an image.
    fn find_type_size(&self, ty: Type<'a>) -> Option<u32> {
        let size = match ty.kind {
            TypeKind::Pointer { ty_id } => {
                self.find_type_size(self.find_type(ty_id)?)?
            }
            TypeKind::Array { ty_id, len_id, .. } => {
                let len = self.find_const(len_id)?.val;
                let elem_size = self.find_type_size(self.find_type(ty_id)?)?;

                len * elem_size
            }
            TypeKind::Struct { member_ids } => {
                // Get the largest offset of the members in the struct.
                let Some((offset, member)) = self.member_decos
                    .iter()
                    .filter_map(|deco| {
                        (deco.struct_ty_id == ty.id && deco.kind == DecoKind::Offset)
                            .then(|| {
                                deco.args
                                    .get(0)
                                    .copied()
                                    .map(|offset| {
                                        (offset, deco.member)
                                    })
                            })
                            .flatten()
                    })
                    .max_by_key(|(offset, _)| *offset)
                else {
                    return Some(0);
                };

                let member_ty = self.find_type(member_ids[member as usize])?;
                let size = self.find_type_size(member_ty)?;

                offset + size
            }
            TypeKind::Prim { bytes } => bytes,
            TypeKind::Vector { card, ty_id } => {
                card * self.find_type_size(self.find_type(ty_id)?)?
            }
            _ => {
                return None;
            }
        };

        Some(size)
    }

    /// Fetch through all layers of pointers. Returns `None` if underlying type isn't recognized.
    fn fetch_through_ptr(&'a self, mut ty: Type<'a>) -> Option<Type<'a>> {
        while let TypeKind::Pointer { ty_id } = ty.kind {
            let Some(point_ty) = self.find_type(ty_id) else {
                return None;
            };

            ty = point_ty;
        }

        Some(ty)
    }
}

fn get_desc_binds(
    binary: &[u32],
) -> Result<(HashMap<BindSlot, DescBind>, Option<PushConstRange>)> {
    let reflection = Reflection::new(binary)?;

    let mut descs = HashMap::new();
    let mut push_const = None;

    'var_loop: for var in &reflection.vars {
        // If it's not a type we save in `Reflection::types`, it's not a variable we care about.
        let Some(mut ty) = reflection
            .find_type(var.ty)
            .map(|ty| reflection.fetch_through_ptr(ty))
            .flatten()
        else {
            continue;
        };

        if var.storage == StorageClass::PushConstant {
            let Some(size) = reflection.find_type_size(ty) else {
                continue;
            };

            // Make sure the push constant is a struct. Then calculate the start offset of the
            // struct.
            let offset = if let TypeKind::Struct { member_ids } = ty.kind {
                let mut count = 0;

                let min = reflection.member_decos
                    .iter()
                    .filter_map(|deco| {
                        (deco.struct_ty_id == ty.id && deco.kind == DecoKind::Offset)
                            .then(|| deco.args.get(0).copied())
                            .flatten()
                    })
                    .inspect(|_| count += 1)
                    .min()
                    .unwrap_or(0);

                // This is not perfect. If for some weird reason one member has two offset
                // decorators, this would be incorrect.
                if count != member_ids.len() {
                    0
                } else {
                    min
                }
            } else {
                return Err(anyhow!("push constant is not a struct")); 
            };

            if push_const.is_some() {
                return Err(anyhow!("more than one push constant in shader"));
            }
            
            push_const = Some(PushConstRange { offset, size }); 

            continue;
        }

        let Some(set) = reflection.decos.iter().find(|deco| {
            deco.id == var.id && deco.kind == DecoKind::DescSet
        }) else {
            continue;
        };

        let Some(binding) = reflection.decos.iter().find(|deco| {
            deco.id == var.id && deco.kind == DecoKind::Binding
        }) else {
            continue;
        };

        let Some(set) = set.args.first().cloned() else {
            return Err(anyhow!("descriptor set decorator doesn't specify set"));
        };

        let Some(binding) = binding.args.first().cloned() else {
            return Err(anyhow!("descriptor binding decorator doesn't specify binding"));
        };

        let (kind, count, access) = match var.storage {
            StorageClass::PushConstant => unreachable!(),
            StorageClass::Uniform => {
                (DescKind::UniformBuffer, DescCount::Single, DescAccess::Read)
            },
            StorageClass::StorageBuffer => {
                let non_readable = reflection.member_decos.iter().any(|deco| {
                    deco.struct_ty_id == ty.id && deco.kind == DecoKind::NonReadable
                });

                let non_writeable = reflection.member_decos.iter().any(|deco| {
                    deco.struct_ty_id == ty.id && deco.kind == DecoKind::NonWriteable
                });

                let access = DescAccess::from_rw_flags(non_readable, non_writeable)?;

                (DescKind::StorageBuffer, DescCount::Single, access)
            }
            StorageClass::UniformConstant => {
                let mut count = DescCount::Single;

                let (kind, access) = loop {
                    match ty.kind {
                        TypeKind::Struct { .. }
                        | TypeKind::Prim { .. }
                        | TypeKind::Vector { .. } => {
                            // When a variable has storage class uniform constant is seems to
                            // always be an image.
                            
                            // continue 'var_loop;
                            
                            unreachable!(
                                "uniform constant variable is struct, primitive or vector"
                            );
                        }
                        TypeKind::Image { .. } => {
                            let non_readable = reflection.decos.iter().any(|deco| {
                                deco.id == var.id && deco.kind == DecoKind::NonReadable
                            });

                            let non_writeable = reflection.decos.iter().any(|deco| {
                                deco.id == var.id && deco.kind == DecoKind::NonWriteable
                            });

                            let access = DescAccess::from_rw_flags(non_readable, non_writeable)?;

                            break (DescKind::StorageImage, access);
                        }
                        TypeKind::ImageSampled { .. } => {
                            break (DescKind::SampledImage, DescAccess::Read);
                        }
                        TypeKind::Pointer { ty_id } => {
                            let Some(point_ty) = reflection.find_type(ty_id) else {
                                continue 'var_loop;
                            };

                            ty = point_ty;
                        }
                        TypeKind::Array { ty_id, len_id } => {
                            let Some(elem_ty) = reflection.find_type(ty_id) else {
                                continue 'var_loop;
                            };
    
                            ty = elem_ty;
                            
                            let Some(len) = reflection.find_const(len_id) else {
                                continue 'var_loop;
                            };

                            // For some reason unbound descriptor arrays are marked by having
                            // the length of 1 in SPIR-V from shaderc. This is not mentioned in
                            // the SPIR-V spec.
                            if len.val == 1 {
                                count = DescCount::Unbound;
                            } else {
                                count = DescCount::Bound(len.val);
                            }
                        }
                    }
                };

                (kind, count, access)
            }
        };

        let slot = BindSlot::new(set, binding);
        let bind = DescBind { kind, count, access };

        descs.insert(slot, bind);
    }

    Ok((descs, push_const))
}

const SPIRV_MAGIC_VALUE: u32 = 0x07230203;

const OP_CONST: u16 = 43;
const OP_VAR: u16 = 59;
const OP_DECO: u16 = 71;
const OP_MEMBER_DECO: u16 = 72;

const OP_TYPE_INT: u16 = 21;
const OP_TYPE_FLOAT: u16 = 22;
const OP_TYPE_VECTOR: u16 = 23;
const OP_TYPE_IMAGE: u16 = 25;
const OP_TYPE_SAMPLED_IMAGE: u16 = 27;
const OP_TYPE_ARRAY: u16 = 28;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;

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

        let (slots, _) = get_desc_binds(&spv).unwrap();

        assert_eq!(slots[&BindSlot::new(0, 0)], DescBind {
            kind: DescKind::UniformBuffer, 
            access: DescAccess::Read,
            count: DescCount::Single,
        });

        assert_eq!(slots[&BindSlot::new(0, 1)], DescBind {
            kind: DescKind::StorageBuffer, 
            access: DescAccess::ReadWrite,
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

        let (slots, _) = get_desc_binds(&spv).unwrap();

        assert_eq!(slots[&BindSlot::new(0, 0)], DescBind {
            kind: DescKind::SampledImage, 
            access: DescAccess::Read,
            count: DescCount::Single,
        });

        assert_eq!(slots[&BindSlot::new(0, 1)], DescBind {
            kind: DescKind::StorageImage, 
            access: DescAccess::Write,
            count: DescCount::Single,
        });
    }

    #[test]
    fn combined_image_buffer_desc_slot() {
        let spv = compile(ShaderKind::Compute, r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;
            layout (set = 1, binding = 0) uniform Block {
                int val2;
            };

            void main() {}
        "#);

        let (slots, _) = get_desc_binds(&spv).unwrap();

        assert_eq!(slots[&BindSlot::new(0, 0)], DescBind {
            kind: DescKind::SampledImage, 
            access: DescAccess::Read,
            count: DescCount::Single,
        });

        assert_eq!(slots[&BindSlot::new(1, 0)], DescBind {
            kind: DescKind::UniformBuffer, 
            access: DescAccess::Read,
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

        let (slots, _) = get_desc_binds(&spv).unwrap();

        assert_eq!(slots[&BindSlot::new(0, 0)], DescBind {
            kind: DescKind::SampledImage, 
            access: DescAccess::Read,
            count: DescCount::Unbound,
        });

        assert_eq!(slots[&BindSlot::new(0, 1)], DescBind {
            kind: DescKind::SampledImage, 
            access: DescAccess::Read,
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
            access: DescAccess::Read,
            count: DescCount::Unbound,
        }));

        assert_eq!(prog.desc_bind(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::StorageImage,
            access: DescAccess::Write,
            count: DescCount::Single,
        }));
    }

    #[test]
    fn push_const_range() {
        let code = compile(ShaderKind::Compute, r#"
            #version 450

            layout (push_constant) uniform Block {
                int val;
            };

            void main() {}
        "#);

        let prog = ComputeProg::new(&code).unwrap(); 

        assert_eq!(prog.push_const(), Some(PushConstRange {
            offset: 0,
            size: 4,
        }));
    }

    #[test]
    fn push_const_complex_range() {
        let code = compile(ShaderKind::Compute, r#"
            #version 450

            layout (push_constant) uniform Block {
                layout (offset = 8) int val1;
                int val2;
                layout (offset = 64) int val3;
            };

            void main() {}
        "#);

        let prog = ComputeProg::new(&code).unwrap(); 

        assert_eq!(prog.push_const(), Some(PushConstRange {
            offset: 8,
            size: 68,
        }));
    }

    #[test]
    fn push_const_vector_types_range() {
        let code = compile(ShaderKind::Compute, r#"
            #version 450

            layout (push_constant) uniform Block {
                vec4 val1;
                uvec4 val2;
                ivec4 val3;

                vec2 val4;
                uvec2 val5;
                ivec2 val6;
            };

            void main() {}
        "#);

        let prog = ComputeProg::new(&code).unwrap(); 

        assert_eq!(prog.push_const(), Some(PushConstRange {
            offset: 0,
            size: 72,
        }));
    }

    #[test]
    fn deco_read_write_buffer() {
        let code = compile(ShaderKind::Compute, r#"
            #version 450

            layout (set = 0, binding = 0) readonly buffer B1 {
                int val1;
            };
            layout (set = 1, binding = 0) writeonly buffer B2 {
                int val2;
            };
            layout (set = 2, binding = 0) buffer B3 {
                int val3;
            };

            void main() {}
        "#);

        let prog = ComputeProg::new(&code).unwrap(); 

        assert_eq!(prog.desc_bind(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::StorageBuffer, 
            access: DescAccess::Read,
            count: DescCount::Single,
        }));

        assert_eq!(prog.desc_bind(BindSlot::new(1, 0)), Some(DescBind {
            kind: DescKind::StorageBuffer, 
            access: DescAccess::Write,
            count: DescCount::Single,
        }));

        assert_eq!(prog.desc_bind(BindSlot::new(2, 0)), Some(DescBind {
            kind: DescKind::StorageBuffer, 
            access: DescAccess::ReadWrite,
            count: DescCount::Single,
        }));
    }

    #[test]
    fn deco_read_write_image() {
        let code = compile(ShaderKind::Compute, r#"
            #version 450

            layout (set = 0, binding = 0, r32f) readonly uniform image2D i1;
            layout (set = 1, binding = 0, r32f) writeonly uniform image2D i2;
            layout (set = 2, binding = 0, r32f) uniform image2D i3;

            void main() {}
        "#);

        let prog = ComputeProg::new(&code).unwrap(); 

        assert_eq!(prog.desc_bind(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::StorageImage, 
            access: DescAccess::Read,
            count: DescCount::Single,
        }));

        assert_eq!(prog.desc_bind(BindSlot::new(1, 0)), Some(DescBind {
            kind: DescKind::StorageImage, 
            access: DescAccess::Write,
            count: DescCount::Single,
        }));

        assert_eq!(prog.desc_bind(BindSlot::new(2, 0)), Some(DescBind {
            kind: DescKind::StorageImage, 
            access: DescAccess::ReadWrite,
            count: DescCount::Single,
        }));
    }
}
