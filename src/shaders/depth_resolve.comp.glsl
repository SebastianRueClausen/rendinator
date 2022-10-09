#version 450
#pragma shader_stage(compute)

layout (local_size_x = 16, local_size_y = 16) in;

layout (set = 0, binding = 0) uniform sampler2DMS depth_image;
layout (set = 0, binding = 1, r32f) uniform writeonly image2D depth_staging;

layout (push_constant, std140) uniform Consts {
	uvec2 size;
	uint sample_count;
};

void main() {
	const uvec2 pos = gl_GlobalInvocationID.xy;

	if (pos.x > size.x || pos.y > size.y) return;

	float depth = 0.0;
	for (int i = 0; i < sample_count; i++) {
		depth = max(depth, texelFetch(depth_image, ivec2(pos), i).r);	
	}

#ifdef DEPTH_VISUALIZE
	depth = pow(depth, 20);
#endif

	imageStore(depth_staging, ivec2(pos), vec4(depth));
}
