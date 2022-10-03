glslc pbr.vert.glsl -o ../../assets/shaders/pbr.vert.spv
glslc pbr.frag.glsl -o ../../assets/shaders/pbr.frag.spv

glslc draw_cull.comp.glsl -o ../../assets/shaders/draw_cull.comp.spv
glslc depth_reduce.comp.glsl -o ../../assets/shaders/depth_reduce.comp.spv
glslc depth_resolve.comp.glsl -o ../../assets/shaders/depth_resolve.comp.spv

glslc sdf.vert.glsl -o ../../assets/shaders/sdf.vert.spv
glslc sdf.frag.glsl -o ../../assets/shaders/sdf.frag.spv

glslc shadow_pass.vert.glsl -o ../../assets/shaders/shadow_pass.vert.spv
glslc shadow_pass.frag.glsl -o ../../assets/shaders/shadow_pass.frag.spv

glslc skybox.vert.glsl -o ../../assets/shaders/skybox.vert.spv
glslc skybox.frag.glsl -o ../../assets/shaders/skybox.frag.spv
glslc skybox.comp.glsl -o ../../assets/shaders/skybox.comp.spv

glslc cluster_build.comp.glsl -o ../../assets/shaders/cluster_build.comp.spv
glslc cluster_update.comp.glsl -o ../../assets/shaders/cluster_update.comp.spv
glslc light_update.comp.glsl -o ../../assets/shaders/light_update.comp.spv
