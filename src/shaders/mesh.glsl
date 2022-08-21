
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
