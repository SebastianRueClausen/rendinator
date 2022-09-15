#version 450
#pragma shader_stage(compute)

#extension GL_EXT_shader_16bit_storage: require
#extension GL_EXT_scalar_block_layout: require

#include "mesh.glsl"

const uint THREAD_COUNT = 64;

layout (local_size_x = THREAD_COUNT) in;

layout (std430, push_constant) uniform PushConst {
	CullInfo cull_info;
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
	const vec3 center = (instance.transform * primitive.position).xyz;
	const float radius = primitive.radius;

	// TODO: Handle scaling.
	
	const bool visible = center.z * cull_info.frust_right - abs(center.x) * cull_info.frust_left > -radius
		&& center.z * cull_info.frust_bottom - abs(center.x) * cull_info.frust_top > -radius
		&& center.z + radius > cull_info.z_near && center.z - radius < cull_info.z_far;

	if (visible) {
		const uint command_id = atomicAdd(command_count, 1);

		const float lod_factor = log2(length(center) / cull_info.lod_base) / log2(cull_info.lod_step);
		const uint lod_index = min(uint(max(lod_factor + 1, 0)), primitive.lod_count - 1);

		const Lod lod = primitive.lods[lod_index];	

		draw_commands[command_id] = DrawCommand(
			lod.index_count,
			1,
			lod.first_index,
			primitive.vertex_offset,
			primitive.instance,
			primitive.albedo_map,
			primitive.specular_map,
			primitive.normal_map
		);
	}
}
