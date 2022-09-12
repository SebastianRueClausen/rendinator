#version 450
#pragma shader_stage(vertex)

#extension GL_GOOGLE_include_directive: require
#extension GL_ARB_shader_draw_parameters: require
#extension GL_EXT_shader_16bit_storage: require

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

layout (std430, set = 2, binding = 4) readonly buffer Verts {
	Vert verts[];
};

vec3 decode_normal(const vec2 encoded) {
    const float scale = 1.7777;

    const vec3 nn = vec3(encoded, 0.0)
			* vec3(2.0 * scale, 2.0 * scale, 0.0)
			+ vec3(-scale, -scale, 1.0);

    const float g = 2.0 / dot(nn.xyz, nn.xyz);
    const vec3 n = vec3(g * nn.x, g * nn.y, g - 1.0);

    return vec3(n);	
}

layout (location = 0) out vec2 out_texcoord;
layout (location = 1) out vec3 out_world_normal;
layout (location = 2) out vec4 out_world_tangent;
layout (location = 3) out vec3 out_world_bitangent;
layout (location = 4) out vec4 out_world_position;
layout (location = 5) out float out_view_z;
layout (location = 6) out uvec3 out_textures;

void main() {
	const DrawCommand command = draw_commands[gl_DrawIDARB];

	const vec4 position = verts[gl_VertexIndex].position;
	const InstanceData instance = instance_data[gl_InstanceIndex];
	const vec4 world = instance.transform * position;

	const vec3 normal = decode_normal(vec2(verts[gl_VertexIndex].normal));
	const vec4 tangent = vec4(verts[gl_VertexIndex].tangent);

	out_textures = uvec3(command.albedo_map, command.specular_map, command.normal_map);
	out_texcoord = vec2(verts[gl_VertexIndex].texcoord);
	out_world_normal = normalize(mat3(instance.inverse_transpose_transform) * normal);
	out_world_tangent = normalize(vec4(mat3(instance.transform) * tangent.xyz, tangent.w));

	out_world_bitangent = out_world_tangent.w * cross(out_world_tangent.xyz, out_world_normal);

	out_world_position = world;
	out_view_z = (view * world).z;

	gl_Position = proj_view * world;
}
