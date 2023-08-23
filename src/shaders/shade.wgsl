#import mesh
#import pbr
#import consts
#import util
#import light

@group(0) @binding(0)
var<uniform> consts: consts::Consts;

@group(0) @binding(1)
var texture_sampler: sampler;

@group(1) @binding(0)
var<storage, read> primitives: array<mesh::Primitive>;

@group(1) @binding(1)
var<storage, read> materials: array<mesh::Material>;

@group(1) @binding(2)
var<storage, read> indices: array<u32>;

@group(1) @binding(3)
var<storage, read> vertices: array<mesh::Vertex>;

@group(1) @binding(4)
var textures: binding_array<texture_2d<f32>>;

@group(2) @binding(0)
var visibility_buffer: texture_2d<u32>;

@group(2) @binding(1)
var color_buffer: texture_storage_2d<rgba16float, read_write>;

var<push_constant> ray_matrix: mat4x4f;

struct Triangle {
    p0: vec3f,
    p1: vec3f,
    p2: vec3f,
}

fn intersection(tri: Triangle, ray: util::Ray) -> vec3f {
    let edge_to_origin = ray.origin - tri.p0;
    let edge_2 = tri.p2 - tri.p0;
    let edge_1 = tri.p1 - tri.p0;
    let r = cross(ray.direction, edge_2);
    let s = cross(edge_to_origin, edge_1);
    let inverse_det = 1.0 / dot(r, edge_1);
    let v1 = dot(r, edge_to_origin);
    let v2 = dot(s, ray.direction);
    let b = v1 * inverse_det;
    let c = v2 * inverse_det;
    let a = 1.0 - b - c;
    return vec3f(a, b, c);
}

struct Barycentric {
    lambda: vec3f,
    ddx: vec3f,
    ddy: vec3f,
};

fn barycentric(world_positions: array<vec3f, 3>, ndc: vec2f, screen_size: vec2f) -> Barycentric {
    var ray: util::Ray;
    var bary: Barycentric;

    ray.origin = consts.camera_pos.xyz;
    ray.direction = (ray_matrix * vec4f(ndc, 1.0, 1.0)).xyz;

    var tri: Triangle;
    tri.p0 = world_positions[0];
    tri.p1 = world_positions[1];
    tri.p2 = world_positions[2];

    bary.lambda = intersection(tri, ray);
    let texel_size = 2.0 / screen_size;

    ray.direction = (ray_matrix * vec4f(ndc.x + texel_size.x, ndc.y, 1.0, 1.0)).xyz;
    let hx = intersection(tri, ray);

    ray.direction = (ray_matrix * vec4f(ndc.x, ndc.y + texel_size.y, 1.0, 1.0)).xyz;
    let hy = intersection(tri, ray);

    bary.ddx = bary.lambda - hx;
    bary.ddy = bary.lambda - hy;

    return bary;
}

fn interp_1d(lambda: vec3f, values: vec3f) -> f32 {
    return dot(values, lambda);
}

fn interp_2d(lambda: vec3f, values: array<vec2f, 3>) -> vec2f {
    return fma(vec2f(lambda[0]), values[0], fma(vec2f(lambda[1]), values[1], lambda[2] * values[2]));
}

fn interp_3d(lambda: vec3f, values: array<vec3f, 3>) -> vec3f {
    return fma(vec3f(lambda[0]), values[0], fma(vec3f(lambda[1]), values[1], lambda[2] * values[2]));
}

@compute
@workgroup_size(8, 8)
fn shade(@builtin(global_invocation_id) invocation_id: vec3u) {
    if any(invocation_id.xy >= consts.surface_size) {
        return;
    }

    let texel_id = vec2i(invocation_id.xy);

    var ndc = (vec2f(texel_id.xy) + 0.5) / vec2f(consts.surface_size) * 2.0 - 1.0;
    ndc.y *= -1.0;

    let visibility = textureLoad(visibility_buffer, texel_id, 0).x;

    let primitive_index = (visibility >> mesh::TRIANGLE_INDEX_BITS) - 1u;
    var triangle_index = visibility & mesh::TRIANGLE_INDEX_MASK;

    if triangle_index == 0u {
        textureStore(color_buffer, texel_id, vec4f(0.8));
        return;
    }

    triangle_index -= 1u;

    let primitive = primitives[primitive_index];
    let base_vertex = triangle_index * 3u;

    let vertices = array(
        vertices[indices[base_vertex + 0u]],
        vertices[indices[base_vertex + 1u]],
        vertices[indices[base_vertex + 2u]],
    );

    let world_positions = array<vec3f, 3>(
        (primitive.transform * vec4f(mesh::position(primitive.bounding_sphere, vertices[0]), 1.0)).xyz,
        (primitive.transform * vec4f(mesh::position(primitive.bounding_sphere, vertices[1]), 1.0)).xyz,
        (primitive.transform * vec4f(mesh::position(primitive.bounding_sphere, vertices[2]), 1.0)).xyz,
    );

    let bary = barycentric(world_positions, ndc, vec2f(consts.surface_size));

    let material = materials[mesh::material(vertices[0])];
    let position = interp_3d(bary.lambda, world_positions);

    let tangent_frames = array<mesh::TangentFrame, 3>(
        mesh::tangent_frame(vertices[0]),
        mesh::tangent_frame(vertices[1]),
        mesh::tangent_frame(vertices[2]),
    );

    let normal_transform = mat3x3f(
        primitive.inverse_transpose_transform[0].xyz,
        primitive.inverse_transpose_transform[1].xyz,
        primitive.inverse_transpose_transform[2].xyz,
    );

    let tangent_transform = mat3x3f(
        primitive.transform[0].xyz,
        primitive.transform[1].xyz,
        primitive.transform[2].xyz,
    );

    let bitangent_sign = interp_1d(bary.lambda, vec3f(
        tangent_frames[0].bitangent_sign,
        tangent_frames[1].bitangent_sign,
        tangent_frames[2].bitangent_sign,
    ));

    var normal = normalize(normal_transform * interp_3d(bary.lambda, array<vec3f, 3>(
        tangent_frames[0].normal,
        tangent_frames[1].normal,
        tangent_frames[2].normal,
    )));

    var tangent = normalize(tangent_transform * interp_3d(bary.lambda, array<vec3f, 3>(
        tangent_frames[0].tangent,
        tangent_frames[1].tangent,
        tangent_frames[2].tangent,
    )));

    let bitangent = sign(bitangent_sign) * cross(normal, tangent);

    let texcoords = array<vec2f, 3>(
        mesh::texcoords(vertices[0]),
        mesh::texcoords(vertices[1]),
        mesh::texcoords(vertices[2])
    );

    let uv = interp_2d(bary.lambda, texcoords);
    let uv_ddx = interp_2d(bary.ddx, texcoords);
    let uv_ddy = interp_2d(bary.ddy, texcoords);

    var tangent_space_normal = util::octahedron_decode(
        textureSampleGrad(
            textures[material.normal_texture],
            texture_sampler,
            uv,
            uv_ddx,
            uv_ddy,
            vec2<i32>(0, 0)
        ).xy,
    );

    normal = normalize(
        tangent_space_normal.x * tangent
            + tangent_space_normal.y * bitangent
            + tangent_space_normal.z * normal,
    );

    // Setup shade data.
    var shade: pbr::ShadeParameters;

    var emissive = textureSampleGrad(
        textures[material.emissive_texture],
        texture_sampler,
        uv,
        uv_ddx,
        uv_ddy,
        vec2i(0, 0),
    ).rgb;

    emissive *= material.emissive.rgb;

    shade.albedo = textureSampleGrad(
        textures[material.albedo_texture],
        texture_sampler,
        uv,
        uv_ddx,
        uv_ddy,
        vec2i(0, 0),
    ).rgb;

    let specular_params = textureSampleGrad(
        textures[material.specular_texture],
        texture_sampler,
        uv,
        uv_ddx,
        uv_ddy,
        vec2i(0, 0),
    );

    shade.albedo *= material.base_color.rgb;

    let roughness = specular_params.g * material.roughness;
    shade.metallic = specular_params.r * material.metallic;
    shade.roughness = roughness * roughness;

    var dielectric_specular = (material.ior - 1.0) / (material.ior + 1.0);
    dielectric_specular *= dielectric_specular;

    shade.fresnel_min = mix(vec3f(dielectric_specular), shade.albedo, shade.metallic);
    shade.fresnel_max = saturate(dot(shade.fresnel_min, vec3f(50.0 * 0.33)));

    shade.view_direction = normalize(consts.camera_pos.xyz - position);
    shade.normal_dot_view = clamp(dot(normal, shade.view_direction), 0.0001, 1.0);

    var directional_light: light::DirectionalLight;
    directional_light.direction = vec4f(0.0, 1.0, 0.0, 1.0);
    directional_light.irradiance = vec4f(4.0);

    var light: pbr::LightParameters;
    light.specular_intensity = 1.0;
    light.light_direction = directional_light.direction.xyz;
    light.half_vector = normalize(shade.view_direction + light.light_direction);
    light.normal_dot_half = saturate(dot(normal, light.half_vector));
    light.normal_dot_light = saturate(dot(normal, light.light_direction));
    light.view_dot_half = saturate(dot(shade.view_direction, light.half_vector));
    light.light_dot_view = saturate(dot(light.light_direction, shade.view_direction));
    light.light_dot_half = saturate(dot(light.light_direction, light.half_vector));

    let diffuse_color = shade.albedo * (1.0 - shade.metallic);
    let specular = pbr::specular(shade, light);
    let diffuse = diffuse_color * pbr::burley_diffuse(shade, light);

    let radiance = (diffuse + specular)
        * light.normal_dot_light
        * directional_light.irradiance.xyz;

    let ambient = shade.albedo * 0.2;

    let final_color = vec4f(radiance + ambient + emissive, 1.0);
    textureStore(color_buffer, texel_id, final_color);
}
