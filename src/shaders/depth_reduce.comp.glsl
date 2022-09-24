#version 450
#pragma shader_stage(compute)

#extension GL_EXT_nonuniform_qualifier: require

layout (local_size_x = 32, local_size_y = 32) in;

layout (set = 0, binding = 0) uniform sampler2D sampled_images[];
layout (set = 1, binding = 0, r32f) uniform writeonly image2D storage_images[];

layout (push_constant, std140) uniform Consts {
	uvec2 size;
	uint target;
};

void main() {
	const uvec2 pos = gl_GlobalInvocationID.xy;
	const float depth = texture(sampled_images[target], (vec2(pos) + vec2(0.5)) / vec2(size)).r;
	imageStore(storage_images[target], ivec2(pos), vec4(depth));
}
