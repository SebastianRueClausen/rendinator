use eyre::{Context, Result};
use std::path::Path;

mod gltf;
mod normals;

pub fn load_gltf(path: &Path) -> Result<asset::Scene> {
    let importer = gltf::Importer::new(&path)
        .wrap_err_with(|| format!("failed creating gltf importer for file at {:?}", path))?;
    let scene = importer
        .load_scene()
        .wrap_err_with(|| format!("failed loading gltf scene for file at {:?}", path))?;
    Ok(scene)
}
