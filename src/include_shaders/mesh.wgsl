#define_import_path mesh
#import util

const TRIANGLE_INDEX_BITS = 22u;
const TRIANGLE_INDEX_MASK = 0x3fffffu;

struct BoundingSphere {
    center: vec3f,
    radius: f32,
};

struct Primitive {
    transform: mat4x4f,
    inverse_transpose_transform: mat4x4f,
    bounding_sphere: BoundingSphere,
};

struct Material {
    albedo_texture: u32,
    normal_texture: u32,
    specular_texture: u32,
    emissive_texture: u32,
    base_color: vec4f,
    emissive: vec4f,
    metallic: f32,
    roughness: f32,
    ior: f32,
    padding: array<u32, 1>,
};

struct Vertex {
    raw: array<u32, 4>,
};

struct TangentFrame {
    normal: vec3f,
    tangent: vec3f,
    bitangent_sign: f32,
};

fn deterministic_orthonormal_vector(normal: vec3f) -> vec3f {
    return normalize(select(
        vec3f(0.0, -normal.z, normal.y),
        vec3f(-normal.y, normal.x, 0.0),
        abs(normal.x) > abs(normal.z),
    ));
}

fn tangent_frame(vertex: Vertex) -> TangentFrame {
    let UV_MASK = 0x3ffu;
    let ANGLE_MASK = 0x7ffu;

    var tangent_frame: TangentFrame;
    let encoded = vertex.raw[3];

    let uv = vec2f(
        util::dequantize_unorm(10u, encoded & UV_MASK),
        util::dequantize_unorm(10u, (encoded >> 10u) & UV_MASK),
    );

    tangent_frame.normal = util::octahedron_decode(uv);

    let orthonormal = deterministic_orthonormal_vector(tangent_frame.normal);
    let angle = util::dequantize_unorm(11u, (encoded >> 20u) & ANGLE_MASK) * util::TAU;

    tangent_frame.tangent = orthonormal * cos(angle) + cross(tangent_frame.normal, orthonormal) * sin(angle);
    tangent_frame.bitangent_sign = select(-1.0, 1.0, encoded >> 31u == 1u);

    return tangent_frame;
}

fn position(bounding_sphere: BoundingSphere, vertex: Vertex) -> vec3f {
    let xy = unpack2x16snorm(vertex.raw[1]);
    let z = unpack2x16snorm(vertex.raw[2]).x;
    return (vec3f(xy, z) * bounding_sphere.radius) + bounding_sphere.center;
}

fn material(vertex: Vertex) -> u32 {
    return vertex.raw[2] >> 16u;
}

fn texcoords(vertex: Vertex) -> vec2f {
    return unpack2x16float(vertex.raw[0]);
}
