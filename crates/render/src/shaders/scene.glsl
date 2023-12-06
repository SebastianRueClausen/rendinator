#ifndef SCENE
#define SCENE

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_shader_8bit_storage: require
#extension GL_EXT_shader_explicit_arithmetic_types_int16: require
#extension GL_EXT_shader_explicit_arithmetic_types_float16: require

#include "util.glsl"

struct BoundingSphere {
    vec3 center;
    float radius;
};

struct TangentFrame {
    vec3 normal;
    vec3 tangent;
    float bitangent_sign;
};

struct Vertex {
    float16_t texcoord[2];
    int16_t position[3];
    uint16_t material;
    uint tangent_frame;
};

vec3 deterministic_orthonormal_vector(vec3 normal) {
    if (abs(normal.x) > abs(normal.z)) {
        return vec3(-normal.y, normal.x, 0.0);
    } else {
        return vec3(0.0, -normal.z, normal.y);
    }
}

TangentFrame decode_tangent_frame(uint encoded) {
    const uint UV_MASK = 0x3ff;
    const uint ANGLE_MASK = 0x7ff;
    TangentFrame tangent_frame;

    vec2 uv = vec2(
        dequantize_unorm(10, encoded & UV_MASK),
        dequantize_unorm(10, (encoded >> 10) & UV_MASK)
    );

    tangent_frame.normal = octahedron_decode(uv);
    vec3 orthonormal = deterministic_orthonormal_vector(tangent_frame.normal);
    float angle = dequantize_unorm(11, (encoded >> 20) & ANGLE_MASK) * TAU;

    tangent_frame.tangent = orthonormal
        * cos(angle) + cross(tangent_frame.normal, orthonormal)
        * sin(angle);
    tangent_frame.bitangent_sign = encoded >> 31 == 1 ? 1.0 : -1.0;
    return tangent_frame;
}

vec3 decode_position(BoundingSphere bounding_sphere, int16_t encoded[3]) {
    float x = dequantize_snorm(16, encoded[0]);
    float y = dequantize_snorm(16, encoded[1]);
    float z = dequantize_snorm(16, encoded[2]);
    vec3 position = vec3(x, y, z);

    /*
    mat4 matrix = mat4(
        bounding_sphere.radius, 0.0, 0.0, 0.0,
        0.0, bounding_sphere.radius, 0.0, 0.0,
        0.0, 0.0, bounding_sphere.radius, 0.0,
        bounding_sphere.center.x,
        bounding_sphere.center.y,
        bounding_sphere.center.z,
        1.0
    );
    */

    return (position * bounding_sphere.radius) + bounding_sphere.center;
}

vec2 decode_texcoord(float16_t encoded[2]) {
    return vec2(
        float(encoded[0]),
        float(encoded[1])
    );
}

struct Meshlet {
    BoundingSphere bounding_sphere;
    int8_t cone_axis[3];
    int8_t cone_cutoff;
    uint data_offset;
    uint8_t vertex_count;
    uint8_t triangle_count;
};

struct Lod {
    uint index_offset;
    uint index_count;
    uint meshlet_offset;
    uint meshlet_count;
};

struct Mesh {
    float x, y, z;
    float radius;
    uint vertex_offset;
    uint vertex_count;
    uint material;
    uint lod_count;
    Lod lods[8];
};

struct Material {
    uint albedo_texture;
    uint normal_texture;
    uint specular_texture;
    uint emissive_texture;
    vec4 base_color;
    vec4 emissive;
    float metallic;
    float roughness;
    float ior;
    uint padding;
};

struct Instance {
    mat4 transform;
    mat4 normal_transform;
};

struct Draw {
    float x, y, z;
    float radius;
    uint mesh_index;
    uint instance_index;
    uint visible;
};

struct DrawCommand {
    DrawIndexedIndirectCommand command;
    float x, y, z;
    float radius;
};

#endif