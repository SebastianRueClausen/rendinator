#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require

#include "../constants.glsl"


layout (location = 0) in vec2 texcoord;
layout (location = 1) in vec4 color;

layout (location = 0) out vec4 result;

layout (binding = 0) uniform ConstantData {
    Constants constants;
};

layout (binding = 2) uniform sampler2D textures[];

layout (push_constant) uniform PushConstants {
    vec2 screen_size_in_points;
    uint texture_index;
};

void main() {
    result = color * texture(textures[texture_index], texcoord);
}