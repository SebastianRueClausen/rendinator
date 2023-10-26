use eyre::Result;
use glam::{Vec2, Vec3, Vec4};

pub fn generate_normals(positions: &[Vec3], indices: &[u32]) -> Vec<Vec3> {
    let mut normals = vec![Vec3::ZERO; positions.len()];

    for triangle in indices.chunks(3) {
        let [idx0, idx1, idx2] = triangle else {
            panic!("indices isn't multiple of 3");
        };

        let pos0 = positions[*idx0 as usize];
        let pos1 = positions[*idx1 as usize];
        let pos2 = positions[*idx2 as usize];

        let normal = (pos1 - pos0).cross(pos2 - pos0);
        normals[*idx0 as usize] += normal;
        normals[*idx1 as usize] += normal;
        normals[*idx2 as usize] += normal;
    }

    for normal in &mut normals {
        *normal = normal.normalize();
    }

    normals
}

pub fn generate_tangents(
    positions: &[Vec3],
    texcoords: &[Vec2],
    normals: &[Vec3],
    indices: &[u32],
) -> Result<Vec<Vec4>> {
    let tangents = vec![Vec4::ZERO; positions.len()];

    let mut generator = TangentGenerator {
        positions,
        texcoords,
        normals,
        indices,
        tangents,
    };

    if !mikktspace::generate_tangents(&mut generator) {
        return Err(eyre::eyre!("failed to generate tangents"));
    }

    for tangent in &mut generator.tangents {
        tangent[3] *= -1.0;
    }

    Ok(generator.tangents)
}

struct TangentGenerator<'a> {
    positions: &'a [Vec3],
    texcoords: &'a [Vec2],
    normals: &'a [Vec3],
    indices: &'a [u32],
    tangents: Vec<Vec4>,
}

impl<'a> TangentGenerator<'a> {
    fn index(&self, face: usize, vertex: usize) -> usize {
        self.indices[face * 3 + vertex] as usize
    }
}

impl<'a> mikktspace::Geometry for TangentGenerator<'a> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.index(face, vert)].into()
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.index(face, vert)].into()
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.texcoords[self.index(face, vert)].into()
    }

    fn set_tangent_encoded(&mut self, tangent: [f32; 4], face: usize, vert: usize) {
        let index = self.index(face, vert);
        self.tangents[index] = tangent.into();
    }
}
