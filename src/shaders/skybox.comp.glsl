#version 450
#pragma shader_stage(compute)

#extension GL_GOOGLE_include_directive: require

#include "atmosphere.glsl"
#include "light.glsl"
#include "tonemap.glsl"

const uint TEXTURE_SIZE = 64;

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout (set = 0, binding = 0, rgba16f) writeonly uniform image2DArray skybox;

layout (std430, set = 0, binding = 1) readonly buffer Lights {
	uint point_light_count;

	DirLight dir_light;
	PointLight point_lights[];
};

const mat3 CUBE_MAP_FACES[6] = {
	mat3(vec3(0, 0, -1), vec3(0, -1, 0), vec3(-1, 0, 0)),	
	mat3(vec3(0, 0, 1), vec3(0, -1, 0), vec3(1, 0, 0)),	

	mat3(vec3(1, 0, 0), vec3(0, 0, 1), vec3(0, -1, 0)),	
	mat3(vec3(1, 0, 0), vec3(0, 0, -1), vec3(0, 1, 0)),	

	mat3(vec3(1, 0, 0), vec3(0, -1, 0), vec3(0, 0, -1)),	
	mat3(vec3(-1, 0, 0), vec3(0, -1, 0), vec3(0, 0, 1)),	
};

vec3 sky_color(const vec3 ray_dir, const vec3 sun_dir) {
	const vec3 origin = vec3(0.0);
	const vec3 irradiance = dir_light.irradiance.xyz;

	return int_scattering(origin, ray_dir, sun_dir, irradiance);
}

void main() {
	const uvec3 coords = gl_GlobalInvocationID;

	const uint face = coords.z;
	const vec2 uv = (coords.xy + 0.5) / TEXTURE_SIZE;

	vec3 sun_dir = dir_light.dir.xyz;
	sun_dir.y = -sun_dir.y;

	const vec3 ray_dir = normalize(CUBE_MAP_FACES[face] * vec3(uv * 2.0 - 1.0, -1.0));
	const vec3 color = aces_approx_tonemap(sky_color(ray_dir, sun_dir));

	imageStore(skybox, ivec3(coords), vec4(color, 1.0));
}

