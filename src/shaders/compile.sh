glslc pbr.vert.glsl --target-env=vulkan1.2 -o ../../assets/shaders/pbr.vert.spv
glslc pbr.frag.glsl --target-env=vulkan1.2 -o ../../assets/shaders/pbr.frag.spv

glslc draw_cull.comp.glsl --target-env=vulkan1.2 -o ../../assets/shaders/draw_cull.comp.spv
glslc depth_reduce.comp.glsl --target-env=vulkan1.2 -o ../../assets/shaders/depth_reduce.comp.spv
glslc depth_resolve.comp.glsl --target-env=vulkan1.2 -o ../../assets/shaders/depth_resolve.comp.spv

glslc sdf.vert.glsl --target-env=vulkan1.2 -o ../../assets/shaders/sdf.vert.spv
glslc sdf.frag.glsl --target-env=vulkan1.2 -o ../../assets/shaders/sdf.frag.spv

glslc skybox.vert.glsl --target-env=vulkan1.2 -o ../../assets/shaders/skybox.vert.spv
glslc skybox.frag.glsl --target-env=vulkan1.2 -o ../../assets/shaders/skybox.frag.spv
glslc skybox.comp.glsl --target-env=vulkan1.2 -o ../../assets/shaders/skybox.comp.spv

glslc cluster_build.comp.glsl --target-env=vulkan1.2 -o ../../assets/shaders/cluster_build.comp.spv
glslc cluster_update.comp.glsl --target-env=vulkan1.2 -o ../../assets/shaders/cluster_update.comp.spv
glslc light_update.comp.glsl --target-env=vulkan1.2 -o ../../assets/shaders/light_update.comp.spv
