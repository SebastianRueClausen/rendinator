#import util
#import mesh
#import consts

@group(0) @binding(0)
var<uniform> consts: consts::Consts;

@group(1) @binding(0)
var<storage, read> primitives: array<mesh::Primitive>;

@group(1) @binding(2)
var<storage, read> indices: array<u32>;

@group(1) @binding(3)
var<storage, read> vertices: array<mesh::Vertex>;

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(0) @interpolate(flat) triangle_index: u32,
    @location(1) @interpolate(flat) primitive_index: u32,
};

@vertex
fn vertex(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) primitive_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let transform = primitives[primitive_index].transform;
    let bounding_sphere = primitives[primitive_index].bounding_sphere;

    let index = indices[vertex_index];
    let position = mesh::position(bounding_sphere, vertices[index]);

    var jittered_proj_view = consts.proj_view;
    jittered_proj_view[2][0] += consts.jitter.x;
    jittered_proj_view[2][1] += consts.jitter.y;

    let world_position = transform * vec4f(position, 1.0);
    out.clip_position = jittered_proj_view * world_position;

    out.triangle_index = vertex_index / 3u;
    out.primitive_index = primitive_index;

    return out;
}

@fragment
fn fragment(in: VertexOutput) -> @location(0) u32 {
    return ((in.primitive_index + 1u) << mesh::TRIANGLE_INDEX_BITS) | (in.triangle_index + 1u);
}
