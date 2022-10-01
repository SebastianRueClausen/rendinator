#version 450
#pragma shader_stage(compute)

#extension GL_EXT_scalar_block_layout: require

#include "light.glsl"
#include "camera.glsl"

const uint THREAD_COUNT = 64;

layout (local_size_x = THREAD_COUNT, local_size_y = 1, local_size_z = 1) in;

layout (std140, set = 0, binding = 1) readonly uniform ViewBuf {
	View view;
};

layout (std430, set = 1, binding = 0) readonly uniform LightInfoBuf {
	LightInfo light_info;		
};

layout (std430, set = 1, binding = 2) readonly buffer LightBuf {
	PointLight point_lights[];
};

layout (std430, set = 1, binding = 3) writeonly buffer LightPosBuf {
	LightPos light_positions[];
};

void main() {
	const uint light_index = gl_LocalInvocationIndex + THREAD_COUNT * gl_WorkGroupID.x;

	if (light_index < light_info.point_light_count) {
		const PointLight light = point_lights[light_index];

		light_positions[light_index] = LightPos(
			(view.mat * light.pos).xyz,
			light.radius
		);
	}
}
