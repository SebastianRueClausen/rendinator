
#import mesh
#import light

@group(0) @binding(0)
var<storage, read> shadow_cascades: array<light::ShadowCascade>;

@group(1) @binding(0)
var<storage, read> primitives: array<mesh::Primitive>;

@group(1) @binding(2)
var<storage, read> indices: array<u32>;

@group(1) @binding(3)
var<storage, read> vertices: array<mesh::Vertex>;

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
};

var<push_constant> cascade_index: u32;

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

    let proj_view = shadow_cascades[cascade_index].matrix;

    let world_position = transform * vec4f(position, 1.0);
    out.clip_position = proj_view * world_position;

    return out;
}

@fragment
fn fragment(in: VertexOutput) {}
