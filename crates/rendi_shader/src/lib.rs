use anyhow::{anyhow, Result};
use ash::vk;

use std::cmp::Ordering;
use std::fmt;
use std::slice;

/// The base kind of an shader input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputKind {
    Int,
    Uint,
    Float,
    Double,
}

impl InputKind {
    pub fn bytes(self) -> u32 {
        match self {
            InputKind::Int | InputKind::Uint | InputKind::Float => 4,
            InputKind::Double => 8,
        }
    }
}

/// The shape of an shader input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputShape {
    Scalar = 1,
    Vec2 = 2,
    Vec3 = 3,
    Vec4 = 4,
}

impl InputShape {
    pub fn cardinality(self) -> u32 {
        self as u32
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Input {
    kind: InputKind,
    shape: InputShape,
}

impl Input {
    pub fn bytes(self) -> u32 {
        self.shape.cardinality() * self.kind.bytes()
    }

    pub fn kind(self) -> InputKind {
        self.kind
    }

    pub fn shape(self) -> InputShape {
        self.shape
    }

    pub fn format(self) -> vk::Format {
        use { InputKind::*, InputShape::* };

        match (self.kind(), self.shape()) {
            // Int.
            (Int, Scalar) => vk::Format::R32_SINT,
            (Int, Vec2) => vk::Format::R32G32_SINT,
            (Int, Vec3) => vk::Format::R32G32B32_SINT,
            (Int, Vec4) => vk::Format::R32G32B32A32_SINT,

            // Uint.
            (Uint, Scalar) => vk::Format::R32_UINT,
            (Uint, Vec2) => vk::Format::R32G32_UINT,
            (Uint, Vec3) => vk::Format::R32G32B32_UINT,
            (Uint, Vec4) => vk::Format::R32G32B32A32_UINT,

            // Float.
            (Float, Scalar) => vk::Format::R32_SFLOAT,
            (Float, Vec2) => vk::Format::R32G32_SFLOAT,
            (Float, Vec3) => vk::Format::R32G32B32_SFLOAT,
            (Float, Vec4) => vk::Format::R32G32B32A32_SFLOAT,

            // Double.
            (Double, Scalar) => vk::Format::R64_SFLOAT,
            (Double, Vec2) => vk::Format::R64G64_SFLOAT,
            (Double, Vec3) => vk::Format::R64G64B64_SFLOAT,
            (Double, Vec4) => vk::Format::R64G64B64A64_SFLOAT,
        }
    }
}

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DescCount {
    /// A single descriptor.
    Single,

    /// A bound array of descriptors.
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

impl PartialOrd for BindSlot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for BindSlot {
    fn cmp(&self, other: &Self) -> Ordering {
        let a = (self.set as u64) << 32 | (self.binding as u64);
        let b = (other.set as u64) << 32 | (other.binding as u64);

        a.cmp(&b)
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

#[derive(Clone, Copy)]
struct RwFlags {
    non_readable: bool,
    non_writeable: bool,
}

bitflags::bitflags! {
    pub struct AccessFlags: u8 {
        const READ = 0b01;
        const WRITE = 0b10;
        const READ_WRITE = Self::READ.bits | Self::WRITE.bits;
    }
}

impl AccessFlags {
    fn from_rw_flags(flags: RwFlags) -> Result<Self> {
        let access_flags = match (flags.non_readable, flags.non_writeable) {
            (true, false) => AccessFlags::WRITE,
            (false, true) => AccessFlags::READ,
            (false, false) => AccessFlags::READ_WRITE,
            (true, true) => return Err(anyhow!(
                "descriptor both non-readable and non-writeable")
            ),
        };

        Ok(access_flags)
    }
}

/// A descriptor binding as defined in a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescBind {
    access_flags: AccessFlags,
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

    pub fn access_flags(&self) -> AccessFlags {
        self.access_flags
    }
}

/// A push constant range of a shader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PushConstRange {
    /// The start offset of the push contant. This is the offset into the struct where the first
    /// variable is stored.
    pub offset: u32,

    /// The size of the range from the offset.
    pub size: u32,
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
    args: &'a [u32]
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

#[derive(Clone, Copy, Debug)]
enum PrimKind {
    Int,
    Uint,
    Float,
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
pub struct ComputeReflection {
    desc_binds: DescBinds,
    push_const: Option<PushConstRange>,
}

impl ComputeReflection {
    /// Create new source code for 
    pub fn new(code: &[u32]) -> Result<Self> {
        let (desc_binds, _, push_const) = get_shader_info(&code)?;

        Ok(Self { desc_binds, push_const })
    }

    /// Get descriptor bindings.
    pub fn desc_binds(&self) -> &DescBinds {
        &self.desc_binds
    }

    /// Get push constant range of the the compute shader.
    pub fn push_const_range(&self) -> Option<PushConstRange> {
        self.push_const
    }
}

/// Code and reflection for fragment and vertex shader pair.
pub struct RasterReflection {
    desc_binds: DescBinds,
    vert_inputs: Inputs,

    vert_push_const: Option<PushConstRange>, 
    frag_push_const: Option<PushConstRange>, 
}

impl RasterReflection {
    pub fn new(frag_code: &[u32], vert_code: &[u32]) -> Result<Self> {
        let (frag_binds, _, frag_push_const) = get_shader_info(frag_code)?;
        let (vert_binds, vert_inputs, vert_push_const) = get_shader_info(vert_code)?;

        let mut desc_binds = Vec::from_iter(vert_binds.iter().cloned());

        // Go through each binding in the fragment shader and add it to the vertex bindings if not
        // present already. If it's present, then make sure they match.
        for (slot, frag_bind) in &frag_binds {
            let Some(vert_bind) = vert_binds.get(*slot) else {
                desc_binds.push((*slot, *frag_bind));
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

        let desc_binds = DescBinds::new(desc_binds);

        Ok(Self {
            frag_push_const,
            vert_push_const,
            vert_inputs,
            desc_binds,
        })
    }

    /// Get descriptor bindings.
    pub fn desc_binds(&self) -> &DescBinds {
        &self.desc_binds
    }

    /// Get the push constant range of the fragment shader.
    pub fn frag_push_const_range(&self) -> Option<PushConstRange> {
        self.frag_push_const
    }

    /// Get the push constant range of the vertex shader.
    pub fn vert_push_const_range(&self) -> Option<PushConstRange> {
        self.vert_push_const
    }

    /// Get the vertex inputs.
    pub fn vert_inputs(&self) -> &Inputs {
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
            non_readable: self
                .find_deco(DecoKind::NonReadable, id)
                .is_some(),
            non_writeable: self
                .find_deco(DecoKind::NonWriteable, id)
                .is_some(),
        }
    }

    fn member_rw_flags(&self, id: u32) -> RwFlags {
        RwFlags {
            non_readable: self
                .find_member_deco(DecoKind::NonReadable, id)
                .is_some(),
            non_writeable: self
                .find_member_deco(DecoKind::NonWriteable, id)
                .is_some(),
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
            .find(|deco| {
                deco.id == id && deco.kind == kind
            })
            .copied()
    }

    fn find_member_deco(&self, kind: DecoKind, id: u32) -> Option<MemberDeco> {
        self.member_decos
            .iter()
            .find(|deco| {
                deco.struct_ty_id == id && deco.kind == kind
            })
            .copied()
    }
}

#[derive(Debug)]
pub struct DescBinds {
    binds: Vec<(BindSlot, DescBind)>,
}

impl DescBinds {
    fn new(mut binds: Vec<(BindSlot, DescBind)>) -> Self {
        binds.sort_by_key(|(slot, _)| *slot); 
        Self { binds }
    }
    
    pub fn get(&self, slot: BindSlot) -> Option<DescBind> {
        self.binds
            .binary_search_by_key(&slot, |(slot, _)| *slot)
            .ok()
            .map(|index| {
                self.binds[index].1
            })
    }

    /// Iterate over all the slots and descriptor bindings.
    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<'a> IntoIterator for &'a DescBinds {
    type Item = &'a (BindSlot, DescBind);
    type IntoIter = slice::Iter<'a, (BindSlot, DescBind)>;

    fn into_iter(self) -> Self::IntoIter {
        self.binds.iter()
    }
}

pub struct Inputs {
    inputs: Vec<(u32, Input)>,
}

impl Inputs {
    fn new(mut inputs: Vec<(u32, Input)>) -> Self {
        inputs.sort_by_key(|(loc, _)| *loc);
        Self { inputs }
    }

    pub fn count(&self) -> u32 {
        self.inputs.len() as u32
    }

    pub fn has_any(&self) -> bool {
        self.count() != 0
    }
    
    pub fn get(&self, loc: u32) -> Option<Input> {
        self.inputs.binary_search_by_key(&loc, |(loc, _)| *loc).ok().map(|index| {
            self.inputs[index].1
        })
    }

    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<'a> IntoIterator for &'a Inputs {
    type Item = &'a (u32, Input);
    type IntoIter = slice::Iter<'a, (u32, Input)>;

    fn into_iter(self) -> Self::IntoIter {
        self.inputs.iter()
    }
}

fn get_shader_info(binary: &[u32]) -> Result<(DescBinds, Inputs, Option<PushConstRange>)> {
    let reflection = Reflection::new(binary)?;

    let mut inputs = Vec::new();
    let mut desc_binds = Vec::new();
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

        let is_buffer_block = reflection
            .find_deco(DecoKind::BufferBlock, ty.id)
            .is_some();

        let get_bind_slot = || -> Result<BindSlot> {
            let (Some(set), Some(binding)) = (
                reflection.find_deco(DecoKind::DescSet, var.id),
                reflection.find_deco(DecoKind::Binding, var.id),
            ) else {
                return Err(anyhow!("descriptor set doesn't specify set and binding"));
            };

            let (Some(set), Some(binding)) = (
                set.args.first().cloned(),
                binding.args.first().cloned(),
            ) else {
                return Err(anyhow!("invalid descriptor set set and binding")); 
            };

            Ok((set, binding).into())
        };

        match var.storage {
            StorageClass::PushConstant => {
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
            }
            StorageClass::Input => {
                let get_input_kind = |bytes: u32, kind: PrimKind| Some(match (bytes, kind) {
                    (4, PrimKind::Float) => InputKind::Float,
                    (8, PrimKind::Float) => InputKind::Double,
                    (_, PrimKind::Uint) => InputKind::Uint,
                    (_, PrimKind::Int) => InputKind::Int,
                    _ => return None,
                });

                let (kind, shape) = match ty.kind {
                    TypeKind::Vector { ty_id, card } => {
                        let Some(TypeKind::Prim { bytes, kind}) = reflection
                            .find_type(ty_id)
                            .map(|ty| ty.kind)
                        else {
                            continue;
                        };

                        let Some(kind) = get_input_kind(bytes, kind) else {
                            continue;
                        };

                        let shape = match card {
                            2 => InputShape::Vec2,
                            3 => InputShape::Vec3,
                            4 => InputShape::Vec4,
                            _ => {
                                continue; 
                            }
                        };

                        (kind, shape)
                    }
                    TypeKind::Prim { bytes, kind } => {
                        let Some(kind) = get_input_kind(bytes, kind) else {
                            continue;
                        };

                        (kind, InputShape::Scalar)
                    }
                    _ => {
                        continue;
                    }
                };

                let Some(location) = reflection
                    .find_deco(DecoKind::Location, var.id)
                    .map(|deco| deco.args.first().cloned())
                    .flatten()
                else {
                    continue;
                };

                inputs.push((location, Input { kind, shape }));
            }
            StorageClass::Uniform => {
                let (kind, access_flags) = if is_buffer_block {
                    let rw_flags = reflection.member_rw_flags(ty.id);
                    let access_flags = AccessFlags::from_rw_flags(rw_flags)?;

                    (DescKind::StorageBuffer, access_flags)
                } else {
                    (DescKind::UniformBuffer, AccessFlags::READ)
                };

                desc_binds.push((get_bind_slot()?, DescBind {
                    count: DescCount::Single,
                    access_flags,
                    kind,
                }));
            }
            StorageClass::StorageBuffer => {
                let rw_flags = reflection.member_rw_flags(ty.id);
                let access_flags = AccessFlags::from_rw_flags(rw_flags)?;

                desc_binds.push((get_bind_slot()?, DescBind {
                    kind: DescKind::StorageBuffer,
                    count: DescCount::Single,
                    access_flags,
                }));
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
                            let access_flags = AccessFlags::from_rw_flags(rw_flags)?;

                            break (DescKind::StorageImage, access_flags);
                        }
                        TypeKind::ImageSampled { .. } => {
                            break (DescKind::SampledImage, AccessFlags::READ);
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
                        TypeKind::RuntimeArray { ty_id } => {
                            // Runtime arrays as a type for descriptors seems to mean unbound array.

                            let Some(elem_ty) = reflection.find_type(ty_id) else {
                                continue 'var_loop;
                            };

                            ty = elem_ty;
                            count = DescCount::Unbound;
                        }
                    }
                };

                desc_binds.push((get_bind_slot()?, DescBind {
                    access_flags,
                    kind,
                    count,
                }));
            }
        };
    }

    Ok((DescBinds::new(desc_binds), Inputs::new(inputs), push_const))
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

        let (desc_binds, _, _) = get_shader_info(&spv).unwrap();

        assert_eq!(desc_binds.get(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::UniformBuffer, 
            access_flags: AccessFlags::READ,
            count: DescCount::Single,
        }));

        assert_eq!(desc_binds.get(BindSlot::new(0, 1)), Some(DescBind {
            kind: DescKind::StorageBuffer, 
            access_flags: AccessFlags::READ_WRITE,
            count: DescCount::Single,
        }));
    }

    #[test]
    fn image_desc_slot() {
        let spv = compile(ShaderKind::Compute, r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1;
            layout (set = 0, binding = 1, r32f) writeonly uniform image2D val2;

            void main() {}
        "#);

        let (desc_binds, _, _) = get_shader_info(&spv).unwrap();

        assert_eq!(desc_binds.get(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::SampledImage, 
            access_flags: AccessFlags::READ,
            count: DescCount::Single,
        }));

        assert_eq!(desc_binds.get(BindSlot::new(0, 1)), Some(DescBind {
            kind: DescKind::StorageImage, 
            access_flags: AccessFlags::WRITE,
            count: DescCount::Single,
        }));
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

        let (desc_binds, _, _) = get_shader_info(&spv).unwrap();

        assert_eq!(desc_binds.get(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::SampledImage, 
            access_flags: AccessFlags::READ,
            count: DescCount::Single,
        }));

        assert_eq!(desc_binds.get(BindSlot::new(1, 0)), Some(DescBind {
            kind: DescKind::UniformBuffer, 
            access_flags: AccessFlags::READ,
            count: DescCount::Single,
        }));
    }

    #[test]
    fn image_array() {
        let spv = compile(ShaderKind::Compute, r#"
            #version 450

            layout (set = 0, binding = 0) uniform sampler2D val1[];
            layout (set = 0, binding = 1) uniform sampler2D val2[2];

            void main() {}
        "#);

        let (desc_binds, _, _) = get_shader_info(&spv).unwrap();

        assert_eq!(desc_binds.get(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::SampledImage, 
            access_flags: AccessFlags::READ,
            count: DescCount::Unbound,
        }));

        assert_eq!(desc_binds.get(BindSlot::new(0, 1)), Some(DescBind {
            kind: DescKind::SampledImage, 
            access_flags: AccessFlags::READ,
            count: DescCount::Bound(2),
        }));
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

        let prog = RasterReflection::new(&frag, &vert);

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

        let prog = RasterReflection::new(&frag, &vert);

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

        let prog = RasterReflection::new(&frag, &vert).unwrap();

        assert_eq!(prog.desc_binds().get(BindSlot::new(0, 1)), Some(DescBind {
            kind: DescKind::SampledImage,
            access_flags: AccessFlags::READ,
            count: DescCount::Unbound,
        }));

        assert_eq!(prog.desc_binds().get(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::StorageImage,
            access_flags: AccessFlags::WRITE,
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

        let prog = ComputeReflection::new(&code).unwrap(); 

        assert_eq!(prog.push_const_range(), Some(PushConstRange {
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

        let prog = ComputeReflection::new(&code).unwrap(); 

        assert_eq!(prog.push_const_range(), Some(PushConstRange {
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

        let prog = ComputeReflection::new(&code).unwrap(); 

        assert_eq!(prog.push_const_range(), Some(PushConstRange {
            offset: 0,
            size: 72,
        }));
    }

    #[test]
    fn deco_read_write_buffer() {
        let code = compile(ShaderKind::Compute, r#"
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
        "#);

        let prog = ComputeReflection::new(&code).unwrap(); 

        assert_eq!(prog.desc_binds().get(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::StorageBuffer, 
            access_flags: AccessFlags::READ,
            count: DescCount::Single,
        }));

        assert_eq!(prog.desc_binds().get(BindSlot::new(1, 0)), Some(DescBind {
            kind: DescKind::StorageBuffer, 
            access_flags: AccessFlags::WRITE,
            count: DescCount::Single,
        }));

        assert_eq!(prog.desc_binds().get(BindSlot::new(2, 0)), Some(DescBind {
            kind: DescKind::StorageBuffer, 
            access_flags: AccessFlags::READ_WRITE,
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

        let prog = ComputeReflection::new(&code).unwrap(); 

        assert_eq!(prog.desc_binds().get(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::StorageImage, 
            access_flags: AccessFlags::READ,
            count: DescCount::Single,
        }));

        assert_eq!(prog.desc_binds().get(BindSlot::new(1, 0)), Some(DescBind {
            kind: DescKind::StorageImage, 
            access_flags: AccessFlags::WRITE,
            count: DescCount::Single,
        }));

        assert_eq!(prog.desc_binds().get(BindSlot::new(2, 0)), Some(DescBind {
            kind: DescKind::StorageImage, 
            access_flags: AccessFlags::READ_WRITE,
            count: DescCount::Single,
        }));
    }

    #[test]
    fn vertex_input() {
        let spv = compile(ShaderKind::Vertex, r#"
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
        "#);

        let (_, inputs, _) = get_shader_info(&spv).unwrap();

        let kinds = [InputKind::Float, InputKind::Double, InputKind::Int, InputKind::Uint];
        let shapes = [InputShape::Vec2, InputShape::Vec3, InputShape::Vec4];

        for (i, kind) in kinds.into_iter().enumerate() {
            for (j, shape) in shapes.into_iter().enumerate() {
                let location = (i * 3 + j) as u32;

                assert_eq!(inputs.get(location), Some(Input {
                    kind,
                    shape,
                }));
            }
        }

        assert_eq!(inputs.get(12), Some(Input {
            kind: InputKind::Float,
            shape: InputShape::Scalar,
        }));

        assert_eq!(inputs.get(13), Some(Input {
            kind: InputKind::Double,
            shape: InputShape::Scalar,
        }));

        assert_eq!(inputs.get(14), Some(Input {
            kind: InputKind::Int,
            shape: InputShape::Scalar,
        }));

        assert_eq!(inputs.get(15), Some(Input {
            kind: InputKind::Uint,
            shape: InputShape::Scalar,
        }));
    }

    #[test]
    fn unbound_image_array() {
        let spv = compile(ShaderKind::Compute, r#"
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
        "#);

        let prog = ComputeReflection::new(&spv).unwrap(); 

        assert_eq!(prog.desc_binds().get(BindSlot::new(0, 0)), Some(DescBind {
            kind: DescKind::SampledImage, 
            access_flags: AccessFlags::READ,
            count: DescCount::Unbound,
        }));

        assert_eq!(prog.desc_binds().get(BindSlot::new(1, 0)), Some(DescBind {
            kind: DescKind::StorageImage, 
            access_flags: AccessFlags::WRITE,
            count: DescCount::Unbound,
        }));
    }

    #[test]
    fn cluster_update() {
        let spv = compile(ShaderKind::Compute, r#"
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
        "#);

        let prog = ComputeReflection::new(&spv).unwrap(); 

        assert_eq!(prog.desc_binds().get(BindSlot::new(1, 1)), Some(DescBind {
            kind: DescKind::StorageBuffer, 
            access_flags: AccessFlags::WRITE,
            count: DescCount::Single,
        }));
    }
}
