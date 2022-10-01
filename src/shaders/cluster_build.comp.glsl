#version 450
#pragma shader_stage(compute)

#include "light.glsl"
#include "camera.glsl"

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout (std140, set = 0, binding = 0) readonly uniform ProjBuf {
	Proj proj;
};

layout (std140, set = 1, binding = 0) readonly uniform LightInfoBuf {
	LightInfo light_info;
};

layout (std430, set = 1, binding = 1) writeonly buffer AabbBuf {
	Aabb aabbs[];
};

vec4 screen_to_view(const vec2 screen, const float z) {
	const vec2 coords = screen / proj.surface_size.xy;
	const vec4 clip = vec4(vec2(coords.x, 1.0 - coords.y) * 2.0 - 1.0, z, 1);
	const vec4 view = proj.inverse_proj * clip;

	return view / view.w;
}

uint cluster_index(const uvec3 coords) {
	return coords.z * light_info.subdivisions.x * light_info.subdivisions.y
		+ coords.y * light_info.subdivisions.x
		+ coords.x;
}

void main() {
	const uvec3 cluster_coords = gl_WorkGroupID;
	const uint cluster_index = cluster_index(cluster_coords);

	const vec2 screen_min = vec2(cluster_coords.xy * light_info.cluster_size.xy);
	const vec2 screen_max = vec2((cluster_coords.xy + 1.0) * light_info.cluster_size.xy);

	vec3 view_min = screen_to_view(screen_min, 1.0).xyz;
	vec3 view_max = screen_to_view(screen_max, 1.0).xyz;

	view_min.y = -view_min.y;
	view_max.y = -view_max.y;

	const float z_far_over_z_near = proj.z_far / proj.z_near;

	const float view_near = -proj.z_near * pow(
		z_far_over_z_near,
		cluster_coords.z / float(light_info.subdivisions.z)
	);

	const float view_far = -proj.z_near * pow(
		z_far_over_z_near,
		(cluster_coords.z + 1) / float(light_info.subdivisions.z)
	);

	const vec3 min_near = view_min * view_near / view_min.z;
	const vec3 max_near = view_max * view_near / view_max.z;

	const vec3 min_far = view_min * view_far / view_min.z;
	const vec3 max_far = view_max * view_far / view_max.z;

	aabbs[cluster_index] = Aabb(
		vec4(min(min_near, min(max_near, min(min_far, max_far))), 1.0),
		vec4(max(min_near, max(max_near, max(min_far, max_far))), 1.0)
	);
}
