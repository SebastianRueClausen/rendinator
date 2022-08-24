
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

struct Primitive {
	vec3 position;
	float radius;

	uint vertex_offset;
	uint first_index;
	uint index_count;
	uint instance;

	uint albedo_map;
	uint specular_map;
	uint normal_map;
};

struct FrustrumInfo {
	float z_near;
	float z_far;

	float left;
	float right;
	float top;
	float bottom;
};

struct InstanceData {
	mat4 transform;
	mat4 inverse_transpose_transform;
};

