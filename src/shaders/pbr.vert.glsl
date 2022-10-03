#version 450
#pragma shader_stage(vertex)

#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require
#extension GL_EXT_shader_16bit_storage: require

#define TB_BRANCHLESS

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
layout (location = 2) out vec3 out_world_tangent;
layout (location = 3) out vec3 out_world_bitangent;
layout (location = 4) out vec4 out_world_position;
layout (location = 5) out float out_view_z;
layout (location = 6) out uvec3 out_textures;

vec3 decode_normal(const vec2 encoded) {
    const float scale = 1.7777;

    const vec3 nn = vec3(encoded, 0.0)
			* vec3(2.0 * scale, 2.0 * scale, 0.0)
			+ vec3(-scale, -scale, 1.0);

    const float g = 2.0 / dot(nn.xyz, nn.xyz);
    const vec3 n = vec3(g * nn.x, g * nn.y, g - 1.0);

    return vec3(n);	
}

void main() {
	const DrawCommand command = draw_commands[gl_DrawIDARB];

	vec4 position = vec4(verts[gl_VertexIndex].position);

	const float bitangent_sign = sign(position.w);
	const float tangent_angle = abs(position.w);

	position.w = 1.0;

	const InstanceData instance = instance_data[gl_InstanceIndex];
	const vec4 world = instance.transform * position;

	const vec3 normal = decode_normal(vec2(verts[gl_VertexIndex].normal));

#ifdef TB_BRANCHLESS
	const vec3 tb1 = vec3(-normal.y, normal.x, 0.0);
	const vec3 tb2 = vec3(0.0, -normal.z, normal.y);

	const bool tb_switch = abs(normal.x) > abs(normal.z);
	const vec3 tb = float(tb_switch) * tb1 + float(!tb_switch) * tb2;
#else
	const vec3 tb = abs(normal.x) > abs(normal.z) ? vec3(-normal.y, normal.x, 0.0) : vec3(0.0, -normal.z, normal.y);
#endif	
	
	const vec3 tangent = vec3(tb * cos(tangent_angle) + cross(normal, tb) * sin(tangent_angle));

	out_textures = uvec3(command.albedo_map, command.specular_map, command.normal_map);
	out_texcoord = vec2(verts[gl_VertexIndex].texcoord);

	out_world_normal = normalize(mat3(instance.inverse_transpose_transform) * normal);
	out_world_tangent = normalize(mat3(instance.transform) * tangent.xyz);

	out_world_bitangent = bitangent_sign * cross(out_world_tangent, out_world_normal);

	out_world_position = world;
	out_view_z = (view.mat * world).z;

	gl_Position = view.proj_view * world;
}
