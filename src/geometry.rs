use std::cmp::Ordering;
use std::collections::HashMap;
use std::str::FromStr;
use std::usize;

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct VertexData {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
    pub uvw: [f32; 3],
    pub ao: f32
}

#[derive(Clone, Debug)]
pub struct MeshData {
    pub vertices: Vec<VertexData>,
    pub faces: Vec<[usize; 3]>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct SubMeshExtra {
    pub x: i32,
    pub z: i32,
    rotation: i32
}

impl PartialOrd<Self> for SubMeshExtra {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some({
            let (x_cmp, y_cmp, rot_cmp) = (
                self.x.cmp(&other.x),
                self.z.cmp(&other.z),
                self.rotation.cmp(&other.rotation)
            );

            if x_cmp != Ordering::Equal {
                x_cmp
            } else if y_cmp != Ordering::Equal {
                y_cmp
            } else {
                rot_cmp
            }
        })
    }
}

impl Ord for SubMeshExtra {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl MeshData {
    pub fn parse_ply(content: &str) -> Self {
        let mut lines = content.lines();

        macro_rules! match_next(
            ($($l: literal),+) => {
                $(assert!(lines.next().unwrap().eq($l));)+
            }
        );

        match_next!(
            "ply",
            "format ascii 1.0"
        );

        let mut next_line = lines.next().unwrap();
        if next_line.starts_with("comment") {
            next_line = lines.next().unwrap();
        }

        assert!(next_line.starts_with("element vertex"));

        let vertex_count = usize::from_str(next_line.split_whitespace().last().unwrap()).unwrap();

        match_next!(
            "property float x",
            "property float y",
            "property float z",
            "property float nx",
            "property float ny",
            "property float nz",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "property uchar alpha"
        );

        next_line = lines.next().unwrap();

        assert!(next_line.starts_with("element face"));

        let face_count = usize::from_str(next_line.split_whitespace().last().unwrap()).unwrap();

        match_next!(
            "property list uchar uint vertex_indices",
            "end_header"
        );

        let mut vertices = Vec::with_capacity(vertex_count);
        let mut faces = Vec::with_capacity(face_count);

        for _ in 0..vertex_count {
            let mut vertex_data = lines.next().unwrap().split_whitespace().into_iter();
            let mut position = [0.0f32; 3];
            let mut normal = [0.0f32; 3];
            let mut color = [0.0f32; 3];
            let uvw = [0.0f32; 3];
            let ao = 1.0;
            for i in 0..3 {
                position[i] = f32::from_str(vertex_data.next().unwrap()).unwrap();
            }
            for i in 0..3 {
                normal[i] = f32::from_str(vertex_data.next().unwrap()).unwrap();
            }
            for i in 0..3 { // we skip alpha
                color[i] = (f32::from_str(vertex_data.next().unwrap()).unwrap() / 255.0).sqrt();
            }

            { // Since our file have y and z axes swapped, we reswap them here
                let (p, n) = (position[1], normal[1]);

                position[1] = position[2];
                position[2] = p;

                normal[1] = normal[2];
                normal[2] = n;
            }

            vertices.push(VertexData { position, normal, color, uvw, ao })
        }

        for _ in 0..face_count {
            let mut vertex_data = lines.next().unwrap().split_whitespace().into_iter();
            assert!(vertex_data.next().unwrap().eq("3"));
            let mut face = [Default::default(); 3];
            for i in 0..3 {
                face[i] = usize::from_str(vertex_data.next().unwrap()).unwrap();
            }
            faces.push(face)
        }

        Self {
            vertices,
            faces
        }
    }

    /// function for splitting the mesh into many smaller meshes <br><br>
    ///
    /// @step_size: the size of the step between pieces of a mesh along x and z axes <br><br>
    ///
    /// @width: the logical width of a piece. The final piece will be placed with a half offset
    /// for this width on x and z axes
    pub fn split(&self, step_size: f32, width: f32) -> Vec<(SubMeshExtra, Self)> {
        let partition = {
            // Step 1. Iterate over all faces and partition it over x and z coords of triangle centers
            let mut partition: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
            for face_id in 0..self.faces.len() {
                let verts = self
                    .faces[face_id].iter()
                    .map(|&id| self.vertices[id]);

                let half_position = verts
                    .fold([0.0f32; 3], |acc, next| {
                        let mut acc = acc;
                        acc[0] += next.position[0];
                        acc[1] += next.position[1];
                        acc[2] += next.position[2];
                        acc
                    }).map(|comp| comp / 3.0);

                let x_id = (half_position[0] / step_size) as i32;
                let z_id = (half_position[2] / step_size) as i32;

                if let Some(collection) = partition.get_mut(&(x_id, z_id)) {
                    collection.push(face_id);
                } else {
                    let collection = vec![face_id];
                    partition.insert((x_id, z_id), collection);
                }
            }
            partition
        };
        { // Step 2. Iterate over all partitioned data and extract a sub piece meshes along with their rotations

            // vertex_id_lookup is used to remap old vertex indices to the new,
            // used in an extracted mesh piece
            let mut vertex_id_lookup: HashMap<usize, usize> = HashMap::new();

            let mut result_meshes = Vec::new();
            for partitioned in partition.iter() {
                vertex_id_lookup.clear();

                let (x, z) = *partitioned.0;
                let sub_faces = partitioned.1;

                let mut vertices = Vec::new();
                for &sub_face_id in sub_faces.iter() {
                    for &vertex_idx in self.faces[sub_face_id].iter() {
                        if let Some(_) = vertex_id_lookup.get(&vertex_idx) {
                            continue;
                        }
                        vertex_id_lookup.insert(vertex_idx, vertices.len());

                        // shifting vertices so piece would be placed at center
                        let mut new_vert = self.vertices[vertex_idx];
                        new_vert.position[0] -= width / 2.0 + (x as f32 * step_size);
                        new_vert.position[1] -= width / 2.0;
                        new_vert.position[2] -= width / 2.0 + (z as f32 * step_size);

                        for component in new_vert.position.iter_mut() {
                            // remapping to [-1, 1] range:
                            *component *= 2.0 / width;
                            // posterizing values so we will not suffer from micro "seams":
                            *component = (*component * 100000.0).round() / 100000.0;
                        }

                        let normal_magnitude =
                            (new_vert.normal[0]*new_vert.normal[0] +
                            new_vert.normal[1]*new_vert.normal[1] +
                            new_vert.normal[2]*new_vert.normal[2]).sqrt();

                        for component in new_vert.normal.iter_mut() {
                            // normalize normals (just in case)
                            *component = *component / normal_magnitude;
                        }

                        vertices.push(new_vert);
                    }
                }

                let faces = sub_faces
                    .iter()
                    .map(|&idx| {
                        let mut new_indices: [usize; 3] = Default::default();
                        for i in 0..3 {
                            new_indices[i] = *vertex_id_lookup.get(&self.faces[idx][i]).unwrap();
                        }
                        new_indices
                    })
                    .collect::<Vec<_>>();

                let mut next_mesh = Self { vertices, faces };
                let mut rotation = 0;
                for _ in 0..3 { // we are filling our vector with additional rotations of each sub piece
                    result_meshes.push( (SubMeshExtra { x, z, rotation }, next_mesh.clone()) );
                    rotation += 90;
                    for vert in next_mesh.vertices.iter_mut() {
                        let (x, z, nx, nz) = (
                            vert.position[0], vert.position[2],
                            vert.normal[0], vert.normal[2]
                        );

                        vert.position[2] = x;
                        vert.position[0] = -z;

                        vert.normal[2] = nx;
                        vert.normal[0] = -nz;
                    }
                }
                result_meshes.push( (SubMeshExtra { x, z, rotation }, next_mesh) );
            }

            result_meshes.sort_by(|lhs, rhs| lhs.0.cmp(&rhs.0));

            result_meshes
        }
    }
}