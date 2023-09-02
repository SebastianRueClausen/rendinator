#import consts
#import light
#import util

struct ShadowParams {
    cascade_size: u32,
    lambda: f32,
    near_offset: f32,
}

@group(0) @binding(0)
var<uniform> consts: consts::Consts;

@group(1) @binding(0)
var<storage, read_write> shadow_cascades: array<light::ShadowCascade>;

@group(1) @binding(1)
var depth_pyramid: texture_2d<f32>;

fn look_at(eye: vec3f, center: vec3f, up: vec3f) -> mat4x4f {
    let dir = normalize(center - eye);
    let s = normalize(cross(dir, up));
    let u = cross(s, dir);

    return mat4x4f(
        vec4f(s.x, u.x, -dir.x, 0.0),
        vec4f(s.y, u.y, -dir.y, 0.0),
        vec4f(s.z, u.z, -dir.z, 0.0),
        vec4f(-dot(eye, s), -dot(eye, u), dot(eye, dir), 1.0),
    );
}

fn ortho_proj(
    left: f32,
    right: f32,
    bottom: f32,
    top: f32,
    near: f32,
    far: f32,
) -> mat4x4f {
    let rcp_width = 1.0 / (right - left);
    let rcp_height = 1.0 / (top - bottom);
    let r = 1.0 / (near - far);
    return mat4x4f(
        vec4f(rcp_width + rcp_width, 0.0, 0.0, 0.0),
        vec4f(0.0, rcp_height + rcp_height, 0.0, 0.0),
        vec4f(0.0, 0.0, r, 0.0),
        vec4f(
            -(left + right) * rcp_width,
            -(top + bottom) * rcp_height,
            r * near,
            1.0,
        ),
    );
}

fn update_corners(cascade_count: u32, depth_ratio: f32) {
    let center = consts.camera_pos.xyz;
    let front = consts.camera_front.xyz;

    let up = vec3f(0.0, 1.0, 0.0);
    let right = normalize(cross(front, up));

    for (var i = 0u; i < cascade_count; i += 1u) {
        let cascade = &shadow_cascades[i];

        let far = center + front * (*cascade).far;
        let near = center + front * (*cascade).far;

        let near_height = tan(consts.camera_fov / 2.0) * (*cascade).near;
        let near_width = near_height * depth_ratio;

        let far_height = tan(consts.camera_fov / 2.0) * (*cascade).far;
        let far_width = far_height * depth_ratio;

        (*cascade).corners[0] = vec4f(near - up * near_height - right * near_width, 1.0);
        (*cascade).corners[1] = vec4f(near + up * near_height - right * near_width, 1.0);
        (*cascade).corners[2] = vec4f(near - up * near_height + right * near_width, 1.0);
        (*cascade).corners[3] = vec4f(near + up * near_height + right * near_width, 1.0);

        (*cascade).corners[4] = vec4f(far - up * far_height - right * far_width, 1.0);
        (*cascade).corners[5] = vec4f(far + up * far_height - right * far_width, 1.0);
        (*cascade).corners[6] = vec4f(far - up * far_height + right * far_width, 1.0);
        (*cascade).corners[7] = vec4f(far + up * far_height + right * far_width, 1.0);
    }
}

fn update_matrices(cascade_count: u32, shadow_map_size: u32, light_dir: vec3f, near_offset: f32) {
    for (var i = 0u; i < cascade_count; i += 1u) {
        let cascade = &shadow_cascades[i];

        for (var j = 0u; j < 8u; j += 1u) {
            (*cascade).center += (*cascade).corners[j];
        }

        (*cascade).center /= 8.0;

        var radius = 0.0;
        for (var j = 0u; j < 8u; j += 1u) {
            let distance = length((*cascade).corners[j].xyz - (*cascade).center.xyz);
            radius = max(radius, distance);
        }

        radius = ceil(radius * 16.0) / 16.0;

        let cascade_max = vec3f(radius);
        let cascade_min = vec3f(-radius);

        let cascade_extent = cascade_max - cascade_min;

        let shadow_camera_pos = (*cascade).center.xyz - light_dir * near_offset;
        var shadow_proj = ortho_proj(
            cascade_min.x,
            cascade_max.x,
            cascade_min.y,
            cascade_max.y,
            -near_offset,
            near_offset + cascade_extent.z,
        );

        let shadow_view = look_at(shadow_camera_pos, (*cascade).center.xyz, vec3f(0.0, 1.0, 0.0));
        let shadow_proj_view = shadow_proj * shadow_view;

        var shadow_origin = shadow_proj_view * vec4f(0.0, 0.0, 0.0, 1.0);
        shadow_origin = shadow_origin * (f32(shadow_map_size) / 2.0);

        var origin_offset = round(shadow_origin) - shadow_origin;
        origin_offset = origin_offset * (2.0 / f32(shadow_map_size));

        shadow_proj[3][0] += origin_offset.x;
		shadow_proj[3][1] += origin_offset.y;

        (*cascade).matrix = shadow_proj * shadow_view;
    }
}

@compute
@workgroup_size(1, 1, 1)
fn main() {
    let cascade_count = arrayLength(&shadow_cascades);

    let depth_pyramid_mip_count = textureNumLevels(depth_pyramid);
    let max_mip_level = i32(depth_pyramid_mip_count - 1u);

    let depth_min_max = textureLoad(depth_pyramid, vec2u(0u), max_mip_level);

    let near = util::linearize_depth(
        consts.frustrum_z_planes.x,
        consts.frustrum_z_planes.y,
        depth_min_max.x,
    );

    let far = util::linearize_depth(
        consts.frustrum_z_planes.x,
        consts.frustrum_z_planes.y,
        depth_min_max.y,
    );

    let lambda = 0.3;
    let ratio = far / near;

    for (var i = 0u; i < cascade_count; i += 1u) {
        if i == 0u {
            shadow_cascades[0].near = near;
        } else {
            let fraction = f32(i) / f32(cascade_count);

            let split = lambda * (
                near * pow(ratio, fraction)) + (1.0 - lambda) * (near + (far - near) * fraction
            );

            shadow_cascades[i].near = split;
            shadow_cascades[i - 1u].far = split * 1.0005;

            if i == cascade_count - 1u {
                shadow_cascades[i].far = far;
            }
        }
    }

    let shadow_map_size = 1024u;
    let near_offset = 250.0;

    update_corners(cascade_count, ratio);
    update_matrices(cascade_count, shadow_map_size, consts.sun.direction.xyz, near_offset);
}
