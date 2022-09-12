#version 450
#pragma shader_stage(fragment)

#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require

#include "light.glsl"
#include "tonemap.glsl"

#ifdef CLUSTER_DEBUG
#include "cluster_debug.glsl"
#endif

#define NORMAL_DEBUG

#include "brdf.glsl"

layout (std140, set = 0, binding = 0) readonly uniform Proj {
	mat4 proj;
	mat4 inverse_proj;
	vec2 screen_dimensions;
};

layout (std140, set = 0, binding = 1) readonly uniform View {
	vec4 eye;
	mat4 view;
	mat4 proj_view;
};

layout (std140, set = 1, binding = 0) readonly uniform Cluster {
	ClusterInfo cluster_info;
};

layout (std430, set = 1, binding = 2) readonly buffer Lights {
	uint point_light_count;

	DirLight dir_light;
	PointLight point_lights[];
};

layout (std430, set = 1, binding = 4) readonly buffer LightMasks {
	LightMask light_masks[];
};

layout (set = 2, binding = 5) uniform sampler2D textures[];

layout (location = 0) in vec2 in_texcoord;
layout (location = 1) in vec3 in_world_normal;
layout (location = 2) in vec3 in_world_tangent;
layout (location = 3) in vec3 in_world_bitangent;
layout (location = 4) in vec4 in_world_position;
layout (location = 5) in float in_view_z;
layout (location = 6) flat in uvec3 in_textures;

layout (location = 0) out vec4 out_color;

uvec3 cluster_coords(vec2 coords, float view_z) {
	uvec2 ij = uvec2(coords / cluster_info.cluster_size.xy);
	uint k = uint(log(-view_z) * cluster_info.depth_factors.x - cluster_info.depth_factors.y);
	return uvec3(ij, k);
}

void main() {
	const vec4 color = texture(textures[in_textures.x], in_texcoord);
	const vec2 specular_params = texture(textures[in_textures.y], in_texcoord).ba;
	vec3 normal = texture(textures[in_textures.z], in_texcoord).rgb * 2.0 - 1.0;

	const vec3 albedo = color.rgb;

	const float metallic = specular_params.r;
	const float rough = clamp(geometric_aa(normal, specular_params.g * specular_params.g), 0.05, 1.0);

	normal = normalize(
		normal.x * in_world_tangent.xyz
			+ normal.y * in_world_bitangent
			+ normal.z * normalize(in_world_normal)
	);

	const vec3 view_dir = normalize(eye.xyz - in_world_position.xyz);
	const float norm_dot_view = clamp(dot(normal, view_dir), 0.0001, 1.0);

	const vec3 diffuse_albedo = (1.0 - metallic) * albedo;
	const vec3 f0 = mix(vec3(0.04), albedo, metallic);

	vec3 radiance = vec3(0.0);

	//
	// Calculate lighting from directional light.
	//
	
	{
		const vec3 light_dir = dir_light.dir.xyz;
		const vec3 half_vec = normalize(view_dir + light_dir);

		const float norm_dot_half = clamp(dot(normal, half_vec), 0.0, 1.0);
		const float norm_dot_light = clamp(dot(normal, light_dir), 0.0, 1.0);
		const float view_dot_half = clamp(dot(view_dir, half_vec), 0.0, 1.0);
		const float light_dot_view = clamp(dot(light_dir, view_dir), 0.0, 1.0);
		const float light_dot_half = clamp(dot(light_dir, half_vec), 0.0, 1.0);

		const float d = norm_dist(norm_dot_half, rough);
		const float v = visibility(norm_dot_view, norm_dot_light, rough);
		const vec3 f = fresnel_schlick(f0, view_dot_half);

		const vec3 specular = d * v * f;
		const vec3 diffuse = diffuse_albedo * burley_diffuse(norm_dot_view, norm_dot_light, light_dot_half, rough);

		radiance += (diffuse + specular) * norm_dot_light * dir_light.irradiance.xyz;
	}

	//
	// Calculate lighting from point lights.
	//

	const uvec3 cluster_coords = cluster_coords(gl_FragCoord.xy, in_view_z);
	const uint cluster_index = cluster_index(cluster_info.subdivisions.xyz, cluster_coords);

	const LightMask light_mask = light_masks[cluster_index];
	
	uint light_count = 0;
	for (uint i = 0; i < LIGHT_MASK_WORD_COUNT; ++i) {
		uint word = light_mask.mask[i];

		while (word != 0) {
			const uint bit_index = findLSB(word);
			const uint light_index = i * 32 + bit_index;

			word &= ~uint(1 << bit_index);
			light_count += 1;

			const PointLight light = point_lights[light_index];

			const float light_dist = length(light.pos.xyz - in_world_position.xyz);

			const vec3 light_dir = normalize(light.pos.xyz - in_world_position.xyz);
			const vec3 half_vec = normalize(view_dir + light_dir);

			const vec3 irradiance = vec3(light.lum) / (4.0 * PI * light_dist * light_dist);

			const float norm_dot_half = clamp(dot(normal, half_vec), 0.0, 1.0);
			const float norm_dot_light = clamp(dot(normal, light_dir), 0.0, 1.0);
			const float view_dot_half = clamp(dot(view_dir, half_vec), 0.0, 1.0);
			const float light_dot_view = clamp(dot(light_dir, view_dir), 0.0, 1.0);

			const float d = norm_dist(norm_dot_half, rough);
			const float v = visibility(norm_dot_view, norm_dot_light, rough);
			const vec3 f = fresnel_schlick(f0, view_dot_half);

			const vec3 specular = d * v * f;
			const vec3 diffuse = diffuse_albedo * lambert_diffuse();

			radiance += (specular + diffuse) * norm_dot_light * irradiance;
		}
	}

	const vec3 ambient = vec3(0.1) * albedo;

	out_color = vec4(aces_approx_tonemap(radiance + ambient), 1.0);

#ifdef CLUSTER_DEBUG
	out_color = debug_cluster_overlay(out_color, cluster_coords, light_count);
#endif

#ifdef NORMAL_DEBUG
	out_color = vec4(normal * 0.5 + vec3(0.5), 1.0);
#endif
}
