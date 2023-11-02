#version 450

layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (binding = 0) uniform writeonly image2D surface;
layout (binding = 1) buffer Data {
    uint data[];
};

void main() {
    uvec2 pos = gl_GlobalInvocationID.xy;
    uvec2 surface_size = imageSize(surface).xy;
    if (pos.x > surface_size.x || pos.y > surface_size.y) {
        return;
    }
    imageStore(surface, ivec2(pos), vec4(1.0));
}