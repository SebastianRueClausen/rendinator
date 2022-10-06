#ifndef MESH_GLSL
#define MESH_GLSL

const uint MAX_LOD_COUNT = 8;

struct DrawCommand {
	uint index_count;
	uint instance_count;
	uint first_index;
	uint vertex_offset;
	uint first_instance;

	uint albedo_map;
	uint specular_map;
	uint normal_map;
};

struct Lod {
	uint first_index;
	uint index_count;
};

struct Primitive {
	vec4 center;
	float radius;

	uint _pad;

	Lod lods[MAX_LOD_COUNT];

	uint instance;

	uint vertex_offset;
	uint lod_count;

	uint albedo_map;
	uint specular_map;
	uint normal_map;
};

struct InstanceData {
	mat4 transform;
	mat4 inverse_transpose_transform;
};

struct Vert {
	f16vec4 position;
	f16vec2 texcoord;
	f16vec2 normal;
	f16vec4 tangent;
};

#endif
