#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_16bit_storage: require

#include "mesh.glsl"

const uint THREAD_COUNT = 64;

layout (local_size_x = THREAD_COUNT) in;

layout (std140, push_constant) uniform PushConst {
	FrustrumInfo frustrum_info;		
};

layout (std430, set = 0, binding = 0) readonly buffer Instances {
	InstanceData instances[];
};

layout (std430, set = 0, binding = 1) writeonly buffer DrawBuffer {
	DrawCommand draw_commands[];
};

layout (std430, set = 0, binding = 2) readonly buffer Primitives {
	Primitive primitives[];	
};

layout (std430, set = 0, binding = 3) buffer DrawCount {
	uint command_count;
	uint primitive_count;
};

void main() {
	const uint draw_id = gl_GlobalInvocationID.x;

	if (draw_id >= primitive_count) return;

	const Primitive primitive = primitives[draw_id];
	const InstanceData instance = instances[primitive.instance];

	// Convert primitive center to world space.
	// const vec3 center = (instance.transform * vec4(primitive.position, 1)).xyz;
	const vec3 center = vec3(10.0, 10.0, 10.0);
	const float radius = 200.0;

	// TODO: Handle scaling.
	
	const bool visible = center.z * frustrum_info.right - abs(center.x) * frustrum_info.left > -radius
		&& center.z * frustrum_info.bottom - abs(center.x) * frustrum_info.top > -radius
		&& center.z + radius > frustrum_info.z_near && center.z - radius < frustrum_info.z_far;

	if (visible) {
		const uint command_id = atomicAdd(command_count, 1);

		draw_commands[command_id] = DrawCommand(
			primitive.index_count,
			1,
			primitive.first_index,
			primitive.vertex_offset,
			primitive.instance,
			primitive.albedo_map,
			primitive.specular_map,
			primitive.normal_map
		);
	}
}
