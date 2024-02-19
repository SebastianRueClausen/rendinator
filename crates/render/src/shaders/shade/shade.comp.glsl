#version 450
#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_control_flow_attributes: require
#extension GL_EXT_nonuniform_qualifier: require

#include "../constants.glsl"
#include "../scene.glsl"

layout (binding = 0) uniform ConstantData {
    Constants constants;
};

layout (binding = 1) readonly buffer Draws {
    Draw draws[];
};

layout (binding = 2) readonly buffer Materials {
    Material materials[];
};

layout (binding = 3, rg32ui) readonly uniform uimage2D visibility_buffer;
layout (binding = 4, r32f) readonly uniform image2D depth_buffer;

layout (binding = 5, r11f_g11f_b10f) readonly uniform image2D gbuffer0;
layout (binding = 6, rgba8) readonly uniform image2D gbuffer1;
layout (binding = 7, rgba8) readonly uniform image2D gbuffer2;

layout (binding = 8, r11f_g11f_b10f) writeonly uniform image2D color_buffer;

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

void main() {
    uvec2 pixel_index = gl_GlobalInvocationID.xy;
    uvec2 target_size = imageSize(visibility_buffer).xy;

    if (any(greaterThanEqual(pixel_index, target_size))) {
        return;
    }

    uvec2 visibility = imageLoad(visibility_buffer, ivec2(pixel_index)).xy;
    float depth = imageLoad(depth_buffer, ivec2(pixel_index)).x;

    Draw draw = draws[visibility.y];
}