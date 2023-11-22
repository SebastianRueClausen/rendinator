use std::fmt::Debug;
use std::path::Path;
use std::sync::{Arc, Mutex};

use eyre::{Context, Result};

mod gltf;
mod normals;

#[derive(Clone, Copy, Debug, Default)]
pub enum Stage {
    #[default]
    Files,
    Meshes,
    Textures,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Progress {
    pub percentage: f32,
    pub stage: Stage,
}

pub fn load_gltf(
    path: impl AsRef<Path> + Debug,
    progress: Option<Arc<Mutex<Progress>>>,
) -> Result<asset::Scene> {
    let importer =
        gltf::Importer::new(path.as_ref(), progress).wrap_err_with(|| {
            format!("failed creating gltf importer for file at {:?}", path)
        })?;
    let scene = importer.load_scene().wrap_err_with(|| {
        format!("failed loading gltf scene for file at {:?}", path)
    })?;
    Ok(scene)
}
