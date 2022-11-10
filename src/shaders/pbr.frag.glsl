#version 450
#pragma shader_stage(fragment)

#extension GL_GOOGLE_include_directive: require
#extension GL_EXT_nonuniform_qualifier: require
#extension GL_EXT_scalar_block_layout: require
#extension GL_EXT_shader_16bit_storage: require

#include "light.glsl"
#include "tonemap.glsl"
#include "camera.glsl"
#include "mesh.glsl"

#ifdef CLUSTER_DEBUG
#include "cluster_debug.glsl"
#endif

#include "brdf.glsl"

layout (std140, set = 0, binding = 0) readonly uniform ProjBuf {
	Proj proj;
};

layout (std140, set = 0, binding = 1) readonly uniform ViewBuf {
	View view;
};

// Not used.
layout (std430, set = 1, binding = 0) writeonly buffer DrawBuf {
	DrawCommand draw_commands[];
};

// Not used.
layout (std430, set = 1, binding = 1) buffer DrawCountBuf {
	uint command_count;
	uint primitive_count;
};

// Not used.
layout (std430, set = 1, binding = 2) buffer DrawFlagBuf {
	uint draw_flags[];
};

// Not used.
layout (std430, set = 2, binding = 0) readonly buffer Instances {
	InstanceData instances[];
};

// Not used.
layout (std430, set = 2, binding = 1) readonly buffer Primitives {
	Primitive primitives[];	
};

// Not used
layout (std430, set = 2, binding = 2) readonly buffer Verts {
	Vert verts;
};

layout (set = 2, binding = 3) uniform sampler2D textures[];

layout (std430, set = 3, binding = 0) readonly uniform LightInfoBuf {
	LightInfo light_info;
};

// Not used.
layout (std430, set = 3, binding = 1) readonly buffer AabbBuf {
	Aabb aabbs[];
};

layout (std430, set = 3, binding = 2) readonly buffer LightBuf {
	PointLight point_lights[];
};

// Not used.
layout (std430, set = 3, binding = 3) readonly buffer LightPosBuf {
	LightPos light_positions[];
};

layout (std430, set = 3, binding = 4) readonly buffer LightMaskBuf {
	LightMask light_masks[];
};

layout (location = 0) in vec2 in_texcoord;
layout (location = 1) in vec3 in_world_normal;
layout (location = 2) in vec4 in_world_tangent;
layout (location = 3) in vec4 in_world_position;
layout (location = 5) flat in uvec3 in_textures;

layout (location = 0) out vec4 out_color;

uvec3 cluster_coords(const vec2 coords, const float view_z) {
	const uvec2 ij = uvec2(coords / light_info.cluster_size.xy);
	const uint k = uint(log(-view_z) * light_info.depth_factors.x - light_info.depth_factors.y);
	return uvec3(ij, k);
}

vec3 get_normal() {
	const vec3 tangent_normal = texture(textures[in_textures.z], in_texcoord).xyz * 2 - 1;

	const vec3 tangent = normalize(in_world_tangent.xyz);
	const vec3 normal = normalize(in_world_normal);
	const vec3 bitangent = in_world_tangent.w * cross(tangent, normal);

	const mat3 tbn = mat3(tangent, bitangent, normal);

	return normalize(tbn * tangent_normal);	
}

float linearize_depth(const float z) {
    return proj.z_near * proj.z_far / (proj.z_far + z * (proj.z_near - proj.z_far));
}

void main() {
	const vec4 color = texture(textures[in_textures.x], in_texcoord);
	const vec3 albedo = color.rgb;
	const vec2 specular_params = texture(textures[in_textures.y], in_texcoord).rg;

	const vec3 normal = get_normal();

	const vec3 view_dir = normalize(view.eye.xyz - in_world_position.xyz);
	const float norm_dot_view = clamp(dot(normal, view_dir), 0.0001, 1.0);

	const float metallic = specular_params.g;
	const float rough = clamp(geometric_aa(normal, specular_params.r * specular_params.r), 0.05, 1.0);

	const vec3 diffuse_albedo = (1.0 - metallic) * albedo;
	const vec3 f0 = mix(vec3(0.04), albedo, metallic);

	vec3 radiance = vec3(0.0);

	//
	// Calculate lighting from directional light.
	//
	
	{
		const vec3 light_dir = light_info.dir_light.dir.xyz;
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

		radiance += (diffuse + specular) * norm_dot_light * light_info.dir_light.irradiance.xyz;
	}

	//
	// Calculate lighting from point lights.
	//

	const float depth = -linearize_depth(gl_FragCoord.z);
	const uvec3 cluster_coords = cluster_coords(gl_FragCoord.xy, depth);
	const uint cluster_index = cluster_index(light_info.subdivisions.xyz, cluster_coords);

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

			const vec3 irradiance = light.lum_radius.xyz / (4.0 * PI * light_dist * light_dist);

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

#ifdef METALLIC_DEBUG
	out_color = vec4(vec3(specular_params.g), 1.0);
#endif

#ifdef ROUGHNESS_DEBUG
	out_color = vec4(vec3(specular_params.r), 1.0);
#endif

#ifdef NORMAL_DEBUG
	out_color = vec4(normal * 0.5 + vec3(0.5), 1.0);
#endif
}
