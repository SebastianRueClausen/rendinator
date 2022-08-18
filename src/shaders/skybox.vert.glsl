#version 450
#pragma shader_stage(vertex)

layout (std140, push_constant) uniform PushConsts {
	mat4 transform;
};

layout (location = 0) in vec3 in_position;
layout (location = 0) out vec3 out_texcoord;

void main() {
	out_texcoord = in_position;
	out_texcoord.y = -out_texcoord.y;

	const vec4 pos = transform * vec4(in_position, 1.0);
	gl_Position = pos.xyww;
}
