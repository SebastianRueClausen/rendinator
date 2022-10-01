#version 450
#pragma shader_stage(compute)

#include "light.glsl"

const uint THREADS_PER_CLUSTER = 64;

layout(local_size_x = THREADS_PER_CLUSTER, local_size_y = 1, local_size_z = 1) in;

layout (std140, set = 1, binding = 0) readonly uniform LightInfoBuf {
	LightInfo light_info;	
};

layout (std430, set = 1, binding = 1) readonly buffer AabbBuf {
	Aabb aabbs[];
};

layout (std430, set = 1, binding = 2) readonly buffer LightBuf {
	PointLight point_lights[];
};

layout (std430, set = 1, binding = 3) readonly buffer LightPosBuf {
	LightPos light_positions[];
};

layout (std430, set = 1, binding = 4) writeonly buffer LightMaskBuf {
	LightMask light_masks[];
};

shared LightMask shared_mask;

bool sphere_intersects_aabb(const Aabb aabb, const Sphere sphere) {
	const vec3 closest = max(vec3(aabb.min_point), min(sphere.pos, vec3(aabb.max_point)));
	const vec3 dist = closest - sphere.pos;

	return dot(dist, dist) <= sphere.radius * sphere.radius;
}

void main() {
	if (gl_LocalInvocationID.x == 0) {
		for (uint i = 0; i < LIGHT_MASK_WORD_COUNT; ++i) {
			shared_mask.mask[i] = 0;
		}
	}

	memoryBarrierShared();

	const uint cluster_index = cluster_index(gl_NumWorkGroups, gl_WorkGroupID);
	const Aabb aabb = aabbs[cluster_index];

	const uint start = gl_LocalInvocationID.x;

	for (uint i = start; i < light_info.point_light_count; i += THREADS_PER_CLUSTER) {
		const LightPos light = light_positions[i];

		const Sphere sphere = Sphere(light.pos_radius.xyz, light.pos_radius.w);

		if (sphere_intersects_aabb(aabb, sphere)) {
			const uint word = i / 32;
			const uint bit = i % 32;

			atomicOr(shared_mask.mask[word], 1 << bit);
		}
	}

	memoryBarrierShared();
	barrier();

	if (gl_LocalInvocationID.x == 0) {
		light_masks[cluster_index] = shared_mask;	
	}
}
