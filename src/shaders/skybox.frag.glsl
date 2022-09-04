#version 450
#pragma shader_stage(fragment)

layout (binding = 0) uniform samplerCube box_sampler;

layout (location = 0) in vec3 in_texcoord;
layout (location = 0) out vec4 out_color;

void main() {
	out_color = texture(box_sampler, in_texcoord);	
	out_color.w = 1.0;
}
