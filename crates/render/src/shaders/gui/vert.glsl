#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require

#include "../constants.glsl"

struct Vertex {
    float x, y;
    float u, v;
    uint color;
};

layout (location = 0) out vec2 texcoord;
layout (location = 1) out vec4 color;

layout (binding = 0) uniform ConstantData {
    Constants constants;
};

layout (binding = 1) buffer Vertices {
    Vertex vertices[];
};

layout (binding = 2) uniform sampler2D textures[];

layout (push_constant) uniform PushConstants {
    vec2 screen_size_in_points;
    uint texture_index;
};

vec4 unpack_color(uint color) {
    return vec4(
        float(color & 255),
        float((color >> 8) & 255),
        float((color >> 16) & 255),
        float((color >> 24) & 255)
    ) / 255.0;
}

vec4 translate_position(vec2 position) {
    return vec4(
        2.0 * (position.x / screen_size_in_points.x) - 1.0,
        2.0 * (position.y / screen_size_in_points.y) - 1.0,
        0.0,
        1.0
    );
}

void main() {
    Vertex vertex = vertices[gl_VertexIndex];
    color = unpack_color(vertex.color);
    texcoord = vec2(vertex.u, vertex.v);
    gl_Position = translate_position(vec2(vertex.x, vertex.y));
}