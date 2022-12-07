use crate::{
    Access, BindSlot, BufferKind, DescBinding, DescCount, DescKind, PrimKind, PrimShape, PrimType,
    PushConstRange, ShaderBindings, ShaderInputs, ShaderStage,
};
use rendi_data_structs::SortedMap;

#[derive(Debug, Clone, thiserror::Error)]
pub enum ReflectError {
    #[error("failed parsing SPIR-V: {0}")]
    ParseError(String),

    #[error(
        "different descriptor kind in vertex and fragment shaders at {slot}: {vert} vs {frag}"
    )]
    DifferentDescKind {
        slot: BindSlot,
        vert: DescKind,
        frag: DescKind,
    },

    #[error(
        "different descriptor count in vertex and fragment shaders at {slot}: {vert} vs {frag}"
    )]
    DifferentDescCount {
        slot: BindSlot,
        vert: DescCount,
        frag: DescCount,
    },
}

#[derive(Clone, Copy)]
pub(crate) struct RwFlags {
    non_readable: bool,
    non_writeable: bool,
}

impl TryInto<Access> for RwFlags {
    type Error = ReflectError;

    fn try_into(self) -> Result<Access, Self::Error> {
        let access_flags = match (self.non_readable, self.non_writeable) {
            (true, false) => Access::WRITE,
            (false, true) => Access::READ,
            (false, false) => Access::READ_WRITE,
            (true, true) => {
                return Err(ReflectError::ParseError(
                    "descriptor can't be both readonly and writeonly".to_string(),
                ))
            }
        };

        Ok(access_flags)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DecoKind {
    // Deprecated kind used to indicate a storage buffer instead of a uniform buffer.
    BufferBlock,

    Binding,
    DescSet,
    Offset,
    NonReadable,
    NonWriteable,
    Location,
}

impl DecoKind {
    fn from_ins(val: u32) -> Option<Self> {
        let kind = match val {
            03 => DecoKind::BufferBlock,
            24 => DecoKind::NonWriteable,
            25 => DecoKind::NonReadable,
            30 => DecoKind::Location,
            33 => DecoKind::Binding,
            34 => DecoKind::DescSet,
            35 => DecoKind::Offset,
            _ => return None,
        };

        Some(kind)
    }
}

/// SPIR-V decoration. This is to give attributes to variables and types.
#[derive(Clone, Copy, Debug)]
struct Deco<'a> {
    id: u32,
    kind: DecoKind,
    args: &'a [u32],
}

/// SPIR-V decoration. This is to give attributes to struct members.
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

    /// This usually means either image or sampled image.
    UniformConstant,

    /// Vertex input or fragment input.
    Input,

    StorageBuffer,
    PushConstant,
}

impl StorageClass {
    fn from_ins(val: u32) -> Option<Self> {
        let kind = match val {
            0 => StorageClass::UniformConstant,
            1 => StorageClass::Input,
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
        /// The type ID of the type pointed to.
        ty_id: u32,
    },
    Array {
        /// The type ID of element type.
        ty_id: u32,

        /// ID of the length constant.
        len_id: u32,
    },
    RuntimeArray {
        ty_id: u32,
    },
    Struct {
        /// ID's of each member.
        member_ids: &'a [u32],
    },
    Prim {
        /// The size of the type in bytes.
        bytes: u32,
        kind: PrimKind,
    },
    Vector {
        /// Type ID of the element type.
        ty_id: u32,

        /// The Cardinality of the vector.
        card: u32,
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

/// Reflections for a compute shader.
pub struct ComputeReflection {
    push_const: Option<PushConstRange>,
    bindings: ShaderBindings,
}

impl ComputeReflection {
    /// Generate a reflection from `code`, which should be valid SPIR-V code of a compute shader.
    #[must_use]
    pub fn new(code: &[u32]) -> Result<Self, ReflectError> {
        let (bindings, _, push_const) = get_shader_info(code, ShaderStage::Compute)?;

        Ok(Self {
            bindings: ShaderBindings::new(bindings),
            push_const,
        })
    }

    /// Returns descriptor bindings of the shader.
    #[must_use]
    pub fn bindings(&self) -> &ShaderBindings {
        &self.bindings
    }

    /// Get push constant range of the the compute shader.
    #[must_use]
    pub fn push_const_range(&self) -> Option<PushConstRange> {
        self.push_const
    }
}

/// Reflection of a fragment and vertex shader.
pub struct RasterReflection {
    bindings: ShaderBindings,
    vert_inputs: ShaderInputs,
    vert_push_const: Option<PushConstRange>,
    frag_push_const: Option<PushConstRange>,
}

impl RasterReflection {
    /// Generate reflection of a vertex and fragment shader.
    /// Both `frag_code` and `vert_code` should be valid SPIR-V code of their respective shader
    /// stage.
    #[must_use]
    pub fn new(frag_code: &[u32], vert_code: &[u32]) -> Result<Self, ReflectError> {
        let (frag_binds, _, frag_push_const) = get_shader_info(frag_code, ShaderStage::Fragment)?;
        let (vert_binds, vert_inputs, vert_push_const) =
            get_shader_info(vert_code, ShaderStage::Vertex)?;

        let mut bindings = vert_binds.clone();

        // Go through each binding in the fragment shader and add it to the vertex bindings if not
        // present already. If it's present, then make sure they match.
        for (slot, frag_bind) in &frag_binds {
            let Some(vert_bind) = bindings.get_mut(slot) else {
                bindings.insert(*slot, *frag_bind);
                continue;
            };

            vert_bind.stage = ShaderStage::Raster;

            if vert_bind.kind() != frag_bind.kind() {
                return Err(ReflectError::DifferentDescKind {
                    vert: vert_bind.kind(),
                    frag: frag_bind.kind(),
                    slot: slot.clone(),
                });
            }

            if vert_bind.count() != frag_bind.count() {
                return Err(ReflectError::DifferentDescCount {
                    vert: vert_bind.count(),
                    frag: frag_bind.count(),
                    slot: slot.clone(),
                });
            }
        }

        let bindings = ShaderBindings::new(bindings);

        Ok(Self {
            frag_push_const,
            vert_push_const,
            vert_inputs,
            bindings,
        })
    }

    /// Returns descriptor bindings of the shader.
    #[must_use]
    pub fn bindings(&self) -> &ShaderBindings {
        &self.bindings
    }

    /// Returns the push constant range of the fragment shader.
    ///
    /// Returns `None` if there are none.
    #[must_use]
    pub fn frag_push_const_range(&self) -> Option<PushConstRange> {
        self.frag_push_const
    }

    /// Get the push constant range of the vertex shader.
    ///
    /// Returns `None` if there are none.
    #[must_use]
    pub fn vert_push_const_range(&self) -> Option<PushConstRange> {
        self.vert_push_const
    }

    /// Get the vertex inputs.
    #[must_use]
    pub fn vert_inputs(&self) -> &ShaderInputs {
        &self.vert_inputs
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
    fn expect_arg(self, index: usize) -> Result<u32, ReflectError> {
        self.args.get(index).cloned().ok_or_else(|| {
            ReflectError::ParseError(format!(
                "instruction too short, expected at least {} arguments",
                index + 1
            ))
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
    fn new(binary: &'a [u32]) -> Result<Self, ReflectError> {
        let Some(magic) = binary.first().cloned() else {
            return Err(ReflectError::ParseError("empty binary input".to_string()));
        };

        if magic != SPIRV_MAGIC_VALUE {
            return Err(ReflectError::ParseError(format!(
                "invalid magic value, expected {SPIRV_MAGIC_VALUE} but got {magic}"
            )));
        }

        Ok(Self {
            binary: &binary[5..],
        })
    }
}

impl<'a> Iterator for InsIter<'a> {
    type Item = Ins<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let ins = self.binary.first().cloned().map(|w| {
            let opcode = w as u16;
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
    fn new(binary: &'a [u32]) -> Result<Self, ReflectError> {
        let inss = InsIter::new(binary)?;
        let mut reflection = Reflection::default();

        for ins in inss {
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

                    reflection.member_decos.push(MemberDeco {
                        struct_ty_id,
                        member,
                        args,
                        kind,
                    });
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
                OP_TYPE_RUNTIME_ARRAY => {
                    let id = ins.expect_arg(1)?;
                    let ty_id = ins.expect_arg(2)?;

                    reflection.types.push(Type {
                        kind: TypeKind::RuntimeArray { ty_id },
                        id,
                    });
                }
                OP_TYPE_INT => {
                    let id = ins.expect_arg(1)?;
                    let bits = ins.expect_arg(2)?;
                    let bytes = bits / 8;

                    let kind = match ins.expect_arg(3)? {
                        0 => PrimKind::Uint,
                        1 => PrimKind::Int,
                        _ => continue,
                    };

                    reflection.types.push(Type {
                        kind: TypeKind::Prim { bytes, kind },
                        id,
                    });
                }
                OP_TYPE_FLOAT => {
                    let id = ins.expect_arg(1)?;
                    let bits = ins.expect_arg(2)?;
                    let bytes = bits / 8;

                    reflection.types.push(Type {
                        kind: TypeKind::Prim {
                            kind: PrimKind::Float,
                            bytes,
                        },
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
                _ => (),
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
            TypeKind::Pointer { ty_id } => self.find_type_size(self.find_type(ty_id)?)?,
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
                                    .first()
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
            TypeKind::Prim { bytes, .. } => bytes,
            TypeKind::Vector { card, ty_id } => {
                let ty = self.find_type(ty_id)?;

                card * self.find_type_size(ty)?
            }
            _ => {
                return None;
            }
        };

        Some(size)
    }

    fn rw_flags(&self, id: u32) -> RwFlags {
        RwFlags {
            non_readable: self.find_deco(DecoKind::NonReadable, id).is_some(),
            non_writeable: self.find_deco(DecoKind::NonWriteable, id).is_some(),
        }
    }

    fn member_rw_flags(&self, id: u32) -> RwFlags {
        RwFlags {
            non_readable: self.find_member_deco(DecoKind::NonReadable, id).is_some(),
            non_writeable: self.find_member_deco(DecoKind::NonWriteable, id).is_some(),
        }
    }

    /// Fetch through all layers of pointers. Returns `None` if underlying type isn't recognized.
    fn fetch_through_ptr(&self, mut ty: Type<'a>) -> Option<Type<'a>> {
        while let TypeKind::Pointer { ty_id } = ty.kind {
            let Some(point_ty) = self.find_type(ty_id) else {
                return None;
            };

            ty = point_ty;
        }

        Some(ty)
    }

    fn find_deco(&self, kind: DecoKind, id: u32) -> Option<Deco> {
        self.decos
            .iter()
            .find(|deco| deco.id == id && deco.kind == kind)
            .copied()
    }

    fn find_member_deco(&self, kind: DecoKind, id: u32) -> Option<MemberDeco> {
        self.member_decos
            .iter()
            .find(|deco| deco.struct_ty_id == id && deco.kind == kind)
            .copied()
    }
}
fn get_shader_info(
    binary: &[u32],
    stage: ShaderStage,
) -> Result<
    (
        SortedMap<BindSlot, DescBinding>,
        ShaderInputs,
        Option<PushConstRange>,
    ),
    ReflectError,
> {
    let reflection = Reflection::new(binary)?;

    let mut inputs = Vec::new();
    let mut desc_binds = Vec::new();
    let mut push_const = None;

    'var_loop: for var in &reflection.vars {
        // If it's not a type we save in `Reflection::types`, it's not a variable we care about.
        let Some(mut ty) = reflection
            .find_type(var.ty)
            .and_then(|ty| reflection.fetch_through_ptr(ty))
        else {
            continue;
        };

        let is_buffer_block = reflection.find_deco(DecoKind::BufferBlock, ty.id).is_some();

        let get_bind_slot = || -> Result<BindSlot, ReflectError> {
            let (Some(set), Some(binding)) = (
                reflection.find_deco(DecoKind::DescSet, var.id),
                reflection.find_deco(DecoKind::Binding, var.id),
            ) else {
                return Err(ReflectError::ParseError(
                    "descriptor set doesn't specify set and binding".into()
                ));
            };

            let (Some(set), Some(binding)) = (
                set.args.first().cloned(),
                binding.args.first().cloned(),
            ) else {
                return Err(ReflectError::ParseError(
                    "invalid descriptor set set and binding".into()
                ));
            };

            Ok((set, binding).into())
        };

        match var.storage {
            StorageClass::PushConstant => {
                let Some(size) = reflection.find_type_size(ty) else {
                    return Err(ReflectError::ParseError(
                        "failed to get size of push constant".into()
                    ))
                };

                // Make sure the push constant is a struct. Then calculate the start offset of the
                // struct.
                let offset = if let TypeKind::Struct { member_ids } = ty.kind {
                    let mut count = 0;

                    let min = reflection
                        .member_decos
                        .iter()
                        .filter_map(|deco| {
                            (deco.struct_ty_id == ty.id && deco.kind == DecoKind::Offset)
                                .then(|| deco.args.first().copied())
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
                    return Err(ReflectError::ParseError(
                        "push constant is not a struct".into(),
                    ));
                };

                if push_const.is_some() {
                    return Err(ReflectError::ParseError(
                        "more than one push constant in shader".into(),
                    ));
                }

                push_const = Some(PushConstRange { offset, size });
            }
            StorageClass::Input => {
                let get_input_kind = |bytes: u32, kind: PrimKind| {
                    Ok(match (bytes, kind) {
                        (4, PrimKind::Float) => PrimKind::Float,
                        (8, PrimKind::Float) => PrimKind::Double,
                        (4, PrimKind::Int) => PrimKind::Int,
                        (4, PrimKind::Uint) => PrimKind::Uint,
                        _ => {
                            return Err(ReflectError::ParseError(
                                "shader input has invalid type".into(),
                            ))
                        }
                    })
                };

                let (kind, shape) = match ty.kind {
                    TypeKind::Vector { ty_id, card } => {
                        let Some(TypeKind::Prim { bytes, kind}) = reflection
                            .find_type(ty_id)
                            .map(|ty| ty.kind)
                        else {
                            return Err(ReflectError::ParseError(
                                "failed to find a type for shader input".into()
                            ));
                        };

                        let kind = get_input_kind(bytes, kind)?;
                        let shape = match card {
                            2 => PrimShape::Vec2,
                            3 => PrimShape::Vec3,
                            4 => PrimShape::Vec4,
                            _ => {
                                continue;
                            }
                        };

                        (kind, shape)
                    }
                    TypeKind::Prim { bytes, kind } => {
                        let kind = get_input_kind(bytes, kind)?;

                        (kind, PrimShape::Scalar)
                    }
                    kind => {
                        return Err(ReflectError::ParseError(format!(
                            "shader input has invalid type: {kind:?}"
                        )));
                    }
                };

                let Some(location) = reflection
                    .find_deco(DecoKind::Location, var.id)
                    .and_then(|deco| deco.args.first().cloned())
                else {
                    continue;
                };

                inputs.push((location, PrimType { kind, shape }));
            }
            StorageClass::Uniform => {
                let (kind, access_flags) = if is_buffer_block {
                    let rw_flags = reflection.member_rw_flags(ty.id);
                    let access_flags: Access = rw_flags.try_into()?;

                    (DescKind::Buffer(BufferKind::Storage), access_flags)
                } else {
                    (DescKind::Buffer(BufferKind::Uniform), Access::READ)
                };

                desc_binds.push((
                    get_bind_slot()?,
                    DescBinding {
                        count: DescCount::Single,
                        access_flags,
                        kind,
                        stage,
                    },
                ));
            }
            StorageClass::StorageBuffer => {
                let rw_flags = reflection.member_rw_flags(ty.id);
                let access_flags: Access = rw_flags.try_into()?;

                desc_binds.push((
                    get_bind_slot()?,
                    DescBinding {
                        kind: DescKind::Buffer(BufferKind::Storage),
                        count: DescCount::Single,
                        access_flags,
                        stage,
                    },
                ));
            }
            StorageClass::UniformConstant => {
                let mut count = DescCount::Single;

                let (kind, access_flags) = loop {
                    match ty.kind {
                        TypeKind::Struct { .. }
                        | TypeKind::Prim { .. }
                        | TypeKind::Vector { .. } => {
                            unreachable!(
                                "uniform constant variable is struct, primitive or vector"
                            );
                        }
                        TypeKind::Image { .. } => {
                            let rw_flags = reflection.rw_flags(var.id);
                            let access_flags: Access = rw_flags.try_into()?;

                            break (DescKind::StorageImage, access_flags);
                        }
                        TypeKind::ImageSampled { .. } => {
                            break (DescKind::SampledImage, Access::READ);
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
                                count = DescCount::UnboundArray;
                            } else {
                                count = DescCount::BoundArray(len.val);
                            }
                        }
                        TypeKind::RuntimeArray { ty_id } => {
                            // Runtime arrays as a type for descriptors seems to mean unbound array.

                            let Some(elem_ty) = reflection.find_type(ty_id) else {
                                continue 'var_loop;
                            };

                            ty = elem_ty;
                            count = DescCount::UnboundArray;
                        }
                    }
                };

                desc_binds.push((
                    get_bind_slot()?,
                    DescBinding {
                        access_flags,
                        stage,
                        kind,
                        count,
                    },
                ));
            }
        };
    }

    Ok((
        SortedMap::from_unsorted(desc_binds),
        ShaderInputs::new(inputs),
        push_const,
    ))
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
const OP_TYPE_RUNTIME_ARRAY: u16 = 29;
const OP_TYPE_STRUCT: u16 = 30;
const OP_TYPE_POINTER: u16 = 32;

#[cfg(test)]
mod test {
    use super::*;
    use shaderc::{CompileOptions, Compiler, ShaderKind, SpirvVersion};

    pub fn compile(kind: ShaderKind, src: &str) -> Vec<u32> {
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
        let spv = compile(
            ShaderKind::Compute,
            r#"
            #version 450

            layout (set = 0, binding = 0) uniform Block1 {
                int val1;
                int val2;
            };

            layout (std430, set = 0, binding = 1) buffer Block2 {
                int val3;
            };

            void main() {}
        "#,
        );

        let (desc_binds, _, _) = get_shader_info(&spv, ShaderStage::Compute).unwrap();

        assert_eq!(
            desc_binds.get(&BindSlot::new(0, 0)),
            Some(&DescBinding {
                kind: DescKind::Buffer(BufferKind::Uniform),
                access_flags: Access::READ,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );

        assert_eq!(
            desc_binds.get(&BindSlot::new(0, 1)),
            Some(&DescBinding {
                kind: DescKind::Buffer(BufferKind::Storage),
                access_flags: Access::READ_WRITE,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );
    }

    #[test]
    fn image_desc_slot() {
        let spv = compile(
            ShaderKind::Compute,
            r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;
            layout (set = 0, binding = 1, r32f) writeonly uniform image2D val2;

            void main() {}
        "#,
        );

        let (desc_binds, _, _) = get_shader_info(&spv, ShaderStage::Compute).unwrap();

        assert_eq!(
            desc_binds.get(&BindSlot::new(0, 0)),
            Some(&DescBinding {
                kind: DescKind::SampledImage,
                access_flags: Access::READ,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );

        assert_eq!(
            desc_binds.get(&BindSlot::new(0, 1)),
            Some(&DescBinding {
                kind: DescKind::StorageImage,
                access_flags: Access::WRITE,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );
    }

    #[test]
    fn combined_image_buffer_desc_slot() {
        let spv = compile(
            ShaderKind::Compute,
            r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;
            layout (set = 1, binding = 0) uniform Block {
                int val2;
            };

            void main() {}
        "#,
        );

        let (desc_binds, _, _) = get_shader_info(&spv, ShaderStage::Compute).unwrap();

        assert_eq!(
            desc_binds.get(&BindSlot::new(0, 0)),
            Some(&DescBinding {
                kind: DescKind::SampledImage,
                access_flags: Access::READ,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );

        assert_eq!(
            desc_binds.get(&BindSlot::new(1, 0)),
            Some(&DescBinding {
                kind: DescKind::Buffer(BufferKind::Uniform),
                access_flags: Access::READ,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );
    }

    #[test]
    fn image_array() {
        let spv = compile(
            ShaderKind::Compute,
            r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1[];
            layout (set = 0, binding = 1) uniform sampler2D val2[2];

            void main() {}
        "#,
        );

        let (desc_binds, _, _) = get_shader_info(&spv, ShaderStage::Compute).unwrap();

        assert_eq!(
            desc_binds.get(&BindSlot::new(0, 0)),
            Some(&DescBinding {
                kind: DescKind::SampledImage,
                access_flags: Access::READ,
                count: DescCount::UnboundArray,
                stage: ShaderStage::Compute,
            })
        );

        assert_eq!(
            desc_binds.get(&BindSlot::new(0, 1)),
            Some(&DescBinding {
                kind: DescKind::SampledImage,
                access_flags: Access::READ,
                count: DescCount::BoundArray(2),
                stage: ShaderStage::Compute,
            })
        );
    }

    #[test]
    fn raster_different_types() {
        let frag = compile(
            ShaderKind::Fragment,
            r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;

            void main() {}
        "#,
        );

        let vert = compile(
            ShaderKind::Vertex,
            r#"
            #version 450

            layout (set = 0, binding = 0) writeonly uniform image2D val1;

            void main() {}
        "#,
        );

        let prog = RasterReflection::new(&frag, &vert);

        assert!(prog.is_err());
    }

    #[test]
    fn raster_different_count() {
        let frag = compile(
            ShaderKind::Fragment,
            r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1[];

            void main() {}
        "#,
        );

        let vert = compile(
            ShaderKind::Vertex,
            r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;

            void main() {}
        "#,
        );

        let prog = RasterReflection::new(&frag, &vert);

        assert!(prog.is_err());
    }

    #[test]
    fn raster_combine_overlap() {
        let frag = compile(
            ShaderKind::Fragment,
            r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;

            void main() {}
        "#,
        );

        let vert = compile(
            ShaderKind::Vertex,
            r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;

            void main() {}
        "#,
        );

        let prog = RasterReflection::new(&frag, &vert).unwrap();

        assert_eq!(
            prog.bindings().binding(BindSlot::new(0, 0)),
            Some(&DescBinding {
                kind: DescKind::SampledImage,
                access_flags: Access::READ,
                count: DescCount::Single,
                stage: ShaderStage::Raster,
            })
        );
    }

    #[test]
    fn raster_combine() {
        let frag = compile(
            ShaderKind::Fragment,
            r#"
            #version 450

            layout (set = 0, binding = 1) uniform sampler2D val1[];

            void main() {}
        "#,
        );

        let vert = compile(
            ShaderKind::Vertex,
            r#"
            #version 450

            layout (set = 0, binding = 0) writeonly uniform image2D val1;

            void main() {}
        "#,
        );

        let prog = RasterReflection::new(&frag, &vert).unwrap();

        assert_eq!(
            prog.bindings().binding(BindSlot::new(0, 1)),
            Some(&DescBinding {
                kind: DescKind::SampledImage,
                access_flags: Access::READ,
                count: DescCount::UnboundArray,
                stage: ShaderStage::Fragment,
            })
        );

        assert_eq!(
            prog.bindings().binding(BindSlot::new(0, 0)),
            Some(&DescBinding {
                kind: DescKind::StorageImage,
                access_flags: Access::WRITE,
                count: DescCount::Single,
                stage: ShaderStage::Vertex,
            })
        );
    }

    #[test]
    fn push_const_range() {
        let code = compile(
            ShaderKind::Compute,
            r#"
            #version 450

            layout (push_constant) uniform Block {
                int val;
            };

            void main() {}
        "#,
        );

        let prog = ComputeReflection::new(&code).unwrap();

        assert_eq!(
            prog.push_const_range(),
            Some(PushConstRange { offset: 0, size: 4 })
        );
    }

    #[test]
    fn push_const_complex_range() {
        let code = compile(
            ShaderKind::Compute,
            r#"
            #version 450

            layout (push_constant) uniform Block {
                layout (offset = 8) int val1;
                int val2;
                layout (offset = 64) int val3;
            };

            void main() {}
        "#,
        );

        let prog = ComputeReflection::new(&code).unwrap();

        assert_eq!(
            prog.push_const_range(),
            Some(PushConstRange {
                offset: 8,
                size: 68,
            })
        );
    }

    #[test]
    fn push_const_vector_types_range() {
        let code = compile(
            ShaderKind::Compute,
            r#"
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
        "#,
        );

        let prog = ComputeReflection::new(&code).unwrap();

        assert_eq!(
            prog.push_const_range(),
            Some(PushConstRange {
                offset: 0,
                size: 72,
            })
        );
    }

    #[test]
    fn deco_read_write_buffer() {
        let code = compile(
            ShaderKind::Compute,
            r#"
            #version 450

            layout (set = 0, binding = 0) readonly buffer B1 {
                mat4 val1;
            };
            layout (set = 1, binding = 0) writeonly buffer B2 {
                double val2;
            };
            layout (set = 2, binding = 0) buffer B3 {
                int val3;
            };

            void main() {}
        "#,
        );

        let prog = ComputeReflection::new(&code).unwrap();

        assert_eq!(
            prog.bindings().binding(BindSlot::new(0, 0)),
            Some(&DescBinding {
                kind: DescKind::Buffer(BufferKind::Storage),
                access_flags: Access::READ,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );

        assert_eq!(
            prog.bindings().binding(BindSlot::new(1, 0)),
            Some(&DescBinding {
                kind: DescKind::Buffer(BufferKind::Storage),
                access_flags: Access::WRITE,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );

        assert_eq!(
            prog.bindings().binding(BindSlot::new(2, 0)),
            Some(&DescBinding {
                kind: DescKind::Buffer(BufferKind::Storage),
                access_flags: Access::READ_WRITE,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );
    }

    #[test]
    fn deco_read_write_image() {
        let code = compile(
            ShaderKind::Compute,
            r#"
            #version 450

            layout (set = 0, binding = 0, r32f) readonly uniform image2D i1;
            layout (set = 1, binding = 0, r32f) writeonly uniform image2D i2;
            layout (set = 2, binding = 0, r32f) uniform image2D i3;

            void main() {}
        "#,
        );

        let prog = ComputeReflection::new(&code).unwrap();

        assert_eq!(
            prog.bindings().binding(BindSlot::new(0, 0)),
            Some(&DescBinding {
                kind: DescKind::StorageImage,
                access_flags: Access::READ,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );

        assert_eq!(
            prog.bindings().binding(BindSlot::new(1, 0)),
            Some(&DescBinding {
                kind: DescKind::StorageImage,
                access_flags: Access::WRITE,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );

        assert_eq!(
            prog.bindings().binding(BindSlot::new(2, 0)),
            Some(&DescBinding {
                kind: DescKind::StorageImage,
                access_flags: Access::READ_WRITE,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );
    }

    #[test]
    fn vertex_input() {
        let spv = compile(
            ShaderKind::Vertex,
            r#"
            #version 450

            layout (location = 0) in vec2 i1;
            layout (location = 1) in vec3 i2;
            layout (location = 2) in vec4 i3;

            layout (location = 3) in dvec2 i4;
            layout (location = 4) in dvec3 i5;
            layout (location = 5) in dvec4 i6;

            layout (location = 6) in ivec2 i7;
            layout (location = 7) in ivec3 i8;
            layout (location = 8) in ivec4 i9;

            layout (location = 9) in uvec2 i10;
            layout (location = 10) in uvec3 i11;
            layout (location = 11) in uvec4 i12;

            layout (location = 12) in float i14;
            layout (location = 13) in double i13;
            layout (location = 14) in int i15;
            layout (location = 15) in uint i16;

            void main() {}
        "#,
        );

        let (_, inputs, _) = get_shader_info(&spv, ShaderStage::Vertex).unwrap();

        let kinds = [
            PrimKind::Float,
            PrimKind::Double,
            PrimKind::Int,
            PrimKind::Uint,
        ];

        let shapes = [PrimShape::Vec2, PrimShape::Vec3, PrimShape::Vec4];

        for (i, kind) in kinds.into_iter().enumerate() {
            for (j, shape) in shapes.into_iter().enumerate() {
                let location = (i * 3 + j) as u32;

                assert_eq!(inputs.get(location), Some(PrimType { kind, shape }));
            }
        }

        assert_eq!(
            inputs.get(12),
            Some(PrimType {
                kind: PrimKind::Float,
                shape: PrimShape::Scalar,
            })
        );

        assert_eq!(
            inputs.get(13),
            Some(PrimType {
                kind: PrimKind::Double,
                shape: PrimShape::Scalar,
            })
        );

        assert_eq!(
            inputs.get(14),
            Some(PrimType {
                kind: PrimKind::Int,
                shape: PrimShape::Scalar,
            })
        );

        assert_eq!(
            inputs.get(15),
            Some(PrimType {
                kind: PrimKind::Uint,
                shape: PrimShape::Scalar,
            })
        );
    }

    #[test]
    fn unbound_image_array() {
        let spv = compile(
            ShaderKind::Compute,
            r#"
            #version 450
            #pragma shader_stage(compute)

            #extension GL_EXT_nonuniform_qualifier: require

            layout (local_size_x = 32, local_size_y = 32) in;

            layout (set = 0, binding = 0) uniform sampler2D sampled_images[];
            layout (set = 1, binding = 0, r32f) uniform writeonly image2D storage_images[];

            layout (push_constant) uniform Consts {
              // Size of the target image.
              uvec2 size;

              // Index of the target pyramid level.
              uint target;
            };

            void main() {
              const uvec2 pos = gl_GlobalInvocationID.xy;
             
              const vec4 samples = textureGather(sampled_images[target], (vec2(pos) + vec2(0.5)) / vec2(size), 0);
              const float depth = max(samples.x, max(samples.y, max(samples.z, samples.w)));

              imageStore(storage_images[0], ivec2(0), vec4(0.0));
            }
        "#,
        );

        let prog = ComputeReflection::new(&spv).unwrap();

        assert_eq!(
            prog.bindings().binding(BindSlot::new(0, 0)),
            Some(&DescBinding {
                kind: DescKind::SampledImage,
                access_flags: Access::READ,
                count: DescCount::UnboundArray,
                stage: ShaderStage::Compute,
            })
        );

        assert_eq!(
            prog.bindings().binding(BindSlot::new(1, 0)),
            Some(&DescBinding {
                kind: DescKind::StorageImage,
                access_flags: Access::WRITE,
                count: DescCount::UnboundArray,
                stage: ShaderStage::Compute,
            })
        );
    }

    #[test]
    fn cluster_update() {
        let spv = compile(
            ShaderKind::Compute,
            r#"
            #version 450
            #pragma shader_stage(compute)

            struct DirLight {
              vec4 dir;
              vec4 irradiance;
            };

            struct LightInfo {
              DirLight dir_light;

              uvec4 subdivisions;

              uvec2 cluster_size;
              vec2 depth_factors;

              uint point_light_count;

              uint pad1;
              uint pad2;
              uint pad3;
            };

            struct Aabb {
              vec4 min_point;
              vec4 max_point;
            };

            struct Proj {
              mat4 mat;
              mat4 inverse_proj;

              vec2 surface_size;

              float z_near;
              float z_far;
            };

            layout (std140, set = 0, binding = 0) readonly uniform ProjBuf {
              Proj proj;
            };

            layout (std140, set = 1, binding = 0) readonly uniform LightInfoBuf {
              LightInfo light_info;
            };

            layout (std430, set = 1, binding = 1) writeonly buffer Aabbs {
              Aabb aabbs[];
            };

            vec4 screen_to_view(const vec2 screen, const float z) {
              const vec2 coords = screen / proj.surface_size.xy;
              const vec4 clip = vec4(vec2(coords.x, 1.0 - coords.y) * 2.0 - 1.0, z, 1);
              const vec4 view = proj.inverse_proj * clip;

              return view / view.w;
            }

            uint cluster_index(const uvec3 coords) {
              return coords.z * light_info.subdivisions.x * light_info.subdivisions.y
                + coords.y * light_info.subdivisions.x
                + coords.x;
            }

            void main() {
              const uvec3 cluster_coords = gl_WorkGroupID;
              const uint cluster_index = cluster_index(cluster_coords);

              const vec2 screen_min = vec2(cluster_coords.xy * light_info.cluster_size.xy);
              const vec2 screen_max = vec2((cluster_coords.xy + 1.0) * light_info.cluster_size.xy);

              vec3 view_min = screen_to_view(screen_min, 1.0).xyz;
              vec3 view_max = screen_to_view(screen_max, 1.0).xyz;

              view_min.y = -view_min.y;
              view_max.y = -view_max.y;

              const float z_far_over_z_near = proj.z_far / proj.z_near;

              const float view_near = -proj.z_near * pow(
                z_far_over_z_near,
                cluster_coords.z / float(light_info.subdivisions.z)
              );

              const float view_far = -proj.z_near * pow(
                z_far_over_z_near,
                (cluster_coords.z + 1) / float(light_info.subdivisions.z)
              );

              const vec3 min_near = view_min * view_near / view_min.z;
              const vec3 max_near = view_max * view_near / view_max.z;

              const vec3 min_far = view_min * view_far / view_min.z;
              const vec3 max_far = view_max * view_far / view_max.z;

              aabbs[cluster_index] = Aabb(
                vec4(min(min_near, min(max_near, min(min_far, max_far))), 1.0),
                vec4(max(min_near, max(max_near, max(min_far, max_far))), 1.0)
              );
            }
        "#,
        );

        let prog = ComputeReflection::new(&spv).unwrap();

        assert_eq!(
            prog.bindings().binding(BindSlot::new(1, 1)),
            Some(&DescBinding {
                kind: DescKind::Buffer(BufferKind::Storage),
                access_flags: Access::WRITE,
                count: DescCount::Single,
                stage: ShaderStage::Compute,
            })
        );
    }
}
