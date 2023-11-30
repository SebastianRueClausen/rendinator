#ifndef CONSTANTS
#define CONSTANTS

#include "light.glsl"

struct Constants {
    mat4 proj;
    mat4 view;
    mat4 proj_view;
    vec4 camera_position;
    vec4 frustrum_planes[6];
    DirectionalLight sun;
    uvec2 screen_size;
    float z_near;
    float z_far;
};

#endif