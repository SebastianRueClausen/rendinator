#ifndef CONSTANTS
#define CONSTANTS

#include "light.glsl"

struct Constants {
    mat4 proj;
    mat4 view;
    mat4 proj_view;
    vec4 camera_position;
    DirectionalLight sun;
    uvec2 screen_size;
};

#endif