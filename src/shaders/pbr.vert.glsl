#version 450
#pragma shader_stage(vertex)

#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require
#extension GL_EXT_shader_16bit_storage: require

#include "mesh.glsl"
#include "camera.glsl"

layout (set = 0, binding = 1) uniform ViewBuf {
	View view;
};

layout (std430, set = 1, binding = 0) buffer DrawBuf {
	DrawCommand draw_commands[];
};

layout (std430, set = 2, binding = 0) buffer InstanceBuf {
	InstanceData instance_data[];
};

layout (std430, set = 2, binding = 2) readonly buffer VertBuf {
	Vert verts[];
};

layout (location = 0) out vec2 out_texcoord;
layout (location = 1) out vec3 out_world_normal;
layout (location = 2) out vec4 out_world_tangent;
layout (location = 3) out vec4 out_world_position;
layout (location = 5) out uvec3 out_textures;

void main() {
	const DrawCommand command = draw_commands[gl_DrawIDARB];

	const vec4 position = vec4(verts[gl_VertexIndex].position);

	const InstanceData instance = instance_data[gl_InstanceIndex];
	const vec4 world = instance.transform * position;

	const vec3 normal = vec3(verts[gl_VertexIndex].normal);
	const vec4 tangent = vec4(verts[gl_VertexIndex].tangent);

	out_textures = uvec3(command.albedo_map, command.specular_map, command.normal_map);
	out_texcoord = vec2(verts[gl_VertexIndex].texcoord);

	out_world_normal = normalize(mat3(instance.transform) * normal);
	out_world_tangent = vec4(normalize(mat3(instance.transform) * tangent.xyz), tangent.w);

	out_world_position = world;

	gl_Position = view.proj_view * world;
}
