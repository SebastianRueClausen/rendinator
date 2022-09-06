#version 450
#pragma shader_stage(vertex)

#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require

#include "mesh.glsl"

layout (set = 0, binding = 1) uniform View {
	vec4 eye;
	mat4 view;
	mat4 proj_view;
};

layout (std430, set = 2, binding = 0) buffer Instances {
	InstanceData instance_data[];
};

layout (std430, set = 2, binding = 1) buffer DrawBuffer {
	DrawCommand draw_commands[];
};

layout (location = 0) in vec3 in_position;
layout (location = 1) in vec3 in_normal;
layout (location = 2) in vec2 in_texcoord;
layout (location = 3) in vec4 in_tangent;

layout (location = 0) out vec2 out_texcoord;
layout (location = 1) out vec3 out_world_normal;
layout (location = 2) out vec4 out_world_tangent;
layout (location = 3) out vec3 out_world_bitangent;
layout (location = 4) out vec4 out_world_position;
layout (location = 5) out float out_view_z;
layout (location = 6) out uvec3 out_textures;

void main() {
	const vec4 position = vec4(in_position, 1.0);
	const InstanceData instance = instance_data[gl_InstanceIndex];
	const vec4 world = instance.transform * position;

	const DrawCommand command = draw_commands[gl_DrawIDARB];

	out_textures = uvec3(command.albedo_map, command.specular_map, command.normal_map);
	out_texcoord = in_texcoord;
	out_world_normal = normalize(mat3(instance.inverse_transpose_transform) * in_normal);
	out_world_tangent = normalize(vec4(mat3(instance.transform) * in_tangent.xyz, in_tangent.w));
	out_world_bitangent = out_world_tangent.w * cross(out_world_tangent.xyz, out_world_normal);
	out_world_position = world;
	out_view_z = (view * world).z;

	gl_Position = proj_view * world;
}
