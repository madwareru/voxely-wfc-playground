use std::collections::HashMap;
use glam::{Mat4};
use orom_miniquad::{CullFace, FilterMode, PipelineParams, TextureAccess, TextureFormat, TextureKind, TextureParams, TextureWrap};
use parry3d::bounding_volume::AABB;
use parry3d::math::{Point, Real, Vector};
use parry3d::query::{RayCast};
use parry3d::shape::Triangle;
use rayon::prelude::IntoParallelIterator;
use crate::geometry::VertexData;
use crate::tile_config::{CellType, TileConfigEntry};
use crate::utility::StopWatch;
use rayon::iter::ParallelIterator;

const NUM_SAMPLES: usize = 128;
const SUB_GROUP_SIZE: usize = 12;

struct AaBbEntry {
    aabb: AABB,
    start_vertex_id: usize,
    end_vertex_id: usize
}

#[derive(Copy, Clone)]
enum InternalState {
    Pending,
    GeneratingAO { subgroup_no: usize },
    Ready
}

pub struct VoxelMesh {
    width: usize,
    depth: usize,
    height: usize,
    intenal_state: InternalState,
    grid_data: Vec<Vec<Vec<super::tile_config::CellType>>>,
    pieces: HashMap<super::tile_config::TileConfigEntry, Vec<super::geometry::VertexData>>,
    pipeline: orom_miniquad::Pipeline,
    bindings: orom_miniquad::Bindings,
    is_dirty: bool,
    trivec: Vec<VertexData>,
    aabb_storage: Vec<AaBbEntry>
}

impl VoxelMesh {
    pub fn new(
        ctx: &mut orom_miniquad::Context,
        pieces: HashMap<super::tile_config::TileConfigEntry, Vec<super::geometry::VertexData>>,
        width: usize,
        depth: usize,
        height: usize
    ) -> Self {
        let grid_data = vec![vec![vec![CellType::Air; width + 1]; depth + 1]; height + 1];
        let indices = (0..100000u32)
            .flat_map(|it| [it * 3, it * 3 + 1, it * 3 + 2])
            .collect::<Vec<_>>();

        let ao_texture = orom_miniquad::Texture::new(
            ctx,
            TextureAccess::Static,
            None,
            TextureParams {
                format: TextureFormat::RGBA8,
                wrap: TextureWrap::Clamp,
                filter: FilterMode::Linear,
                width: width as u32,
                height: height as u32,
                depth: depth as u32
            },
            TextureKind::Texture3D
        );

        let bindings = orom_miniquad::Bindings {
            vertex_buffers: vec![
                orom_miniquad::Buffer::stream(
                    ctx,
                    orom_miniquad::BufferType::VertexBuffer,
                    900_000 * std::mem::size_of::<VertexData>(),
                )
            ],
            index_buffer: orom_miniquad::Buffer::immutable(
                ctx,
                orom_miniquad::BufferType::IndexBuffer,
                &indices
            ),
            images: vec![ao_texture],
        };

        let shader = orom_miniquad::Shader::new(
            ctx,
            shader::VERTEX,
            shader::FRAGMENT,
            shader::meta()
        ).unwrap();

        let pipeline = orom_miniquad::Pipeline::with_params(
            ctx,
            &[
                orom_miniquad::BufferLayout::default(),
            ],
            &[
                orom_miniquad::VertexAttribute::with_buffer("pos", orom_miniquad::VertexFormat::Float3, 0),
                orom_miniquad::VertexAttribute::with_buffer("normal0", orom_miniquad::VertexFormat::Float3, 0),
                orom_miniquad::VertexAttribute::with_buffer("color0", orom_miniquad::VertexFormat::Float3, 0),
                orom_miniquad::VertexAttribute::with_buffer("uvw0", orom_miniquad::VertexFormat::Float3, 0),
                orom_miniquad::VertexAttribute::with_buffer("ao0", orom_miniquad::VertexFormat::Float1, 0),
            ],
            shader,
            PipelineParams {
                cull_face: CullFace::Back,
                front_face_order: orom_miniquad::FrontFaceOrder::Clockwise,
                depth_test: orom_miniquad::Comparison::LessOrEqual,
                depth_write: true,
                depth_write_offset: None,
                color_blend: None,
                alpha_blend: None,
                stencil_test: None,
                color_write: (true, true, true, true),
                primitive_type: orom_miniquad::PrimitiveType::Triangles,
            }
        );

        let aabb_storage = Vec::with_capacity(width * height * depth);

        Self {
            intenal_state: InternalState::Pending,
            width, depth, height,
            grid_data,
            pieces,
            pipeline,
            bindings,
            aabb_storage,
            is_dirty: false,
            trivec: Vec::with_capacity(900_000)
        }
    }

    pub fn mark_pending(&mut self) {
        self.intenal_state = InternalState::Pending;
    }

    pub fn set_grid_vertex_unchecked(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        value: super::tile_config::CellType
    ) {
        self.is_dirty = true;
        self.grid_data[k][j][i] = value;
    }

    pub fn set_tile_unchecked(
        &mut self,
        i: usize,
        j: usize,
        k: usize,
        value: super::tile_config::TileConfigEntry
    ) {
        self.set_grid_vertex_unchecked(i, j, k, value.north_west.bottom);
        self.set_grid_vertex_unchecked(i, j, k + 1, value.north_west.up);

        self.set_grid_vertex_unchecked(i + 1, j, k, value.north_east.bottom);
        self.set_grid_vertex_unchecked(i + 1, j, k + 1, value.north_east.up);

        self.set_grid_vertex_unchecked(i, j + 1, k, value.south_west.bottom);
        self.set_grid_vertex_unchecked(i, j + 1, k + 1, value.south_west.up);

        self.set_grid_vertex_unchecked(i + 1, j + 1, k, value.south_east.bottom);
        self.set_grid_vertex_unchecked(i + 1, j + 1, k + 1, value.south_east.up);
    }

    fn get_tile_config_entry(&self, i: usize, j: usize, k: usize) -> super::tile_config::TileConfigEntry {
        TileConfigEntry {
            north_west: super::tile_config::TileConfigEntryColumn {
                bottom: self.grid_data[k][j][i],
                up: self.grid_data[k+1][j][i]
            },
            north_east: super::tile_config::TileConfigEntryColumn {
                bottom: self.grid_data[k][j][i+1],
                up: self.grid_data[k+1][j][i+1]
            },
            south_west: super::tile_config::TileConfigEntryColumn {
                bottom: self.grid_data[k][j+1][i],
                up: self.grid_data[k+1][j+1][i]
            },
            south_east: super::tile_config::TileConfigEntryColumn {
                bottom: self.grid_data[k][j+1][i+1],
                up: self.grid_data[k+1][j+1][i+1]
            },
        }
    }

    fn get_ao_baking_progress(&self) -> Option<f32> {
        match self.intenal_state {
            InternalState::GeneratingAO { subgroup_no } => {
                Some(
                    (
                        100.0 * (subgroup_no * SUB_GROUP_SIZE) as f32 / (self.trivec.len() as f32)
                    ).clamp(0.0, 1.0)
                )
            },
            _ => None
        }
    }

    pub fn update(&mut self, ctx: &mut orom_miniquad::Context) {
        match self.intenal_state {
            InternalState::Pending => {
                if !self.is_dirty { return; }
                self.is_dirty = false;

                self.trivec.clear();
                self.aabb_storage.clear();

                for k in 0..self.height {
                    for j in 0..self.depth {
                        for i in 0..self.width {
                            let entry = self.get_tile_config_entry(i, j, k);
                            if let Some(verts) = self.pieces.get(&entry) {
                                let x = 2.0 * (i) as f32 - self.width as f32;
                                let y = 2.0 * (k) as f32;
                                let z = -2.0 * (j) as f32 + self.depth as f32;

                                self.aabb_storage.push(AaBbEntry {
                                    aabb: AABB::new(
                                        [x - 1.0, y - 1.0, z - 1.0].into(),
                                        [x + 1.0, y + 1.0, z + 1.0].into()
                                    ),
                                    start_vertex_id: self.trivec.len(),
                                    end_vertex_id: self.trivec.len() + verts.len() - 1
                                });

                                for &v in verts.iter() {
                                    let mut uvw = [0.0f32; 3];
                                    uvw[0] = (i as f32) / (self.width as f32);
                                    uvw[0] += (v.position[0] + 1.0) / (self.width as f32 * 2.0);
                                    uvw[0] = (1.0 - uvw[0]).clamp(0.0, 1.0);

                                    uvw[1] = (k as f32) / (self.height as f32);
                                    uvw[1] += ((v.position[1] + 1.0) / (self.height as f32 * 2.0)).clamp(0.0, 1.0);

                                    uvw[2] = (j as f32) / (self.depth as f32);
                                    uvw[2] -= (v.position[2] + 1.0) / (self.depth as f32 * 2.0);
                                    uvw[2] = (1.0 - uvw[2]).clamp(0.0, 1.0);

                                    let v = VertexData {
                                        position: [
                                            v.position[0] + x,
                                            v.position[1] + y,
                                            v.position[2] + z
                                        ],
                                        uvw,
                                        ..v
                                    };
                                    self.trivec.push(v);
                                }
                            }
                        }
                    }
                }
                self.bindings.vertex_buffers[0].update(ctx, &self.trivec);
                self.intenal_state = InternalState::GeneratingAO { subgroup_no: 0 };
            }
            InternalState::GeneratingAO { subgroup_no } => {

                if self.is_dirty {
                    self.intenal_state = InternalState::Pending;
                    return;
                }

                {
                    let _sw = StopWatch::named("ambient occlusion generation step");
                    let current_offset = subgroup_no * SUB_GROUP_SIZE;
                    if current_offset >= self.trivec.len() {
                        self.bindings.vertex_buffers[0].update(ctx, &self.trivec);
                        self.intenal_state = InternalState::Ready;
                    } else {
                        let current_range = current_offset..(current_offset + SUB_GROUP_SIZE)
                            .min(self.trivec.len());
                        for idx in current_range {
                            if idx < self.trivec.len() {
                                let ao: f32 = (0..NUM_SAMPLES).into_par_iter()
                                    .map(|sample_no| self.integrate_vertex(idx, sample_no, NUM_SAMPLES))
                                    .sum();
                                self.trivec[idx].ao = ao / NUM_SAMPLES as f32;
                            }
                        }
                        self.intenal_state = InternalState::GeneratingAO { subgroup_no: subgroup_no + 1 };
                    }
                }

            }
            InternalState::Ready => {
                if self.is_dirty {
                    self.intenal_state = InternalState::Pending;
                }
            }
        }
    }

    fn integrate_vertex(&self, ix: usize, sample_no: usize, num_samples: usize) -> f32 {
        let origin: Point<Real> = self.trivec[ix].position.into();
        let dir: Vector<Real> = self.trivec[ix].normal.into();
        let dir = dir.normalize();

        let mut offset = super::utility::random_direction_on_a_unit_sphere();
        while offset.dot(&dir) < 0.25 {
            offset = super::utility::random_direction_on_a_unit_sphere();
        }
        let r0 = self.integrate_vertex_ext(
            sample_no,
            num_samples,
            origin + offset * 0.01,
            dir
        );

        offset = super::utility::random_direction_on_a_unit_sphere();
        while offset.dot(&dir) < 0.25 {
            offset = super::utility::random_direction_on_a_unit_sphere();
        }
        let r1 = self.integrate_vertex_ext(
            sample_no,
            num_samples,
            origin + offset * 0.01,
            dir
        );

        offset = super::utility::random_direction_on_a_unit_sphere();
        while offset.dot(&dir) < 0.25 {
            offset = super::utility::random_direction_on_a_unit_sphere();
        }
        let r2 = self.integrate_vertex_ext(
            sample_no,
            num_samples,
            origin + offset * 0.01,
            dir
        );

        offset = super::utility::random_direction_on_a_unit_sphere();
        while offset.dot(&dir) < 0.25 {
            offset = super::utility::random_direction_on_a_unit_sphere();
        }
        let r3 = self.integrate_vertex_ext(
            sample_no,
            num_samples,
            origin + offset * 0.01,
            dir
        );

        r0.min(r1).min(r2).min(r3)
    }

    fn integrate_vertex_ext(&self, sample_no: usize, num_samples: usize, origin: Point<Real>, dir: Vector<Real>) -> f32 {
        const RAY_LENGTH: f32 = 256.0;

        let next_ray = super::utility::sun_flower_ray_on_a_hemisphere(
            num_samples,
            sample_no,
            origin,
            dir
        );

        for aabb_idx in 0..self.aabb_storage.len() {
            if !self.aabb_storage[aabb_idx].aabb.intersects_local_ray(&next_ray, RAY_LENGTH) {
                continue;
            }

            let start = self.aabb_storage[aabb_idx].start_vertex_id;
            let end = self.aabb_storage[aabb_idx].end_vertex_id;

            for offset in (start..=end).step_by(3) {
                let next_triangle = Triangle::new(
                    self.trivec[offset].position.into(),
                    self.trivec[offset + 1].position.into(),
                    self.trivec[offset + 2].position.into()
                );

                let is_not_culled = -(next_triangle.normal().unwrap()).dot(&next_ray.dir) < 0.0;

                if is_not_culled && next_triangle.intersects_local_ray(&next_ray, RAY_LENGTH) {
                    return 0.0;
                }
            }
        }
        1.0
    }

    pub fn draw(&self, ctx: &mut orom_miniquad::Context, view_proj: Mat4, model: Mat4, time: f32) {
        if !self.trivec.is_empty() {
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&self.bindings);
            ctx.apply_uniforms(&shader::Uniforms { view_proj, model, time });
            ctx.draw(0, (self.trivec.len()) as i32, 1);
        }
    }
}

mod shader {
    use orom_miniquad::*;

    pub const VERTEX: &str = //language=glsl
        r#"#version 330
    layout(location = 0) in lowp vec3 pos;
    layout(location = 1) in lowp vec3 normal0;
    layout(location = 2) in lowp vec3 color0;
    layout(location = 3) in lowp vec3 uvw0;
    layout(location = 4) in lowp float ao0;

    uniform mat4 view_proj;
    uniform mat4 model;
    uniform lowp float time;

    out lowp vec3 normal;
    out lowp vec3 color;
    out highp vec3 view_dir;
    out highp vec3 light_direction;
    out highp float ao;

    void main() {
        vec4 new_pos = vec4(pos, 1.0);
        mat4 mvp = view_proj * model;
        gl_Position = mvp * new_pos;
        view_dir = normalize(-gl_Position.xyz);
        normal = (model * vec4(normal0, 0.0)).xyz;

        light_direction = (model * vec4(normalize(vec3(-0.3, -0.2, 0.25)), 0.0)).xyz;

        color = color0;
        ao = ao0;
    }
    "#;

    pub const FRAGMENT: &str = //language=glsl
        r#"#version 330
    in lowp vec3 normal;
    in lowp vec3 color;
    in highp vec3 view_dir;
    in highp float ao;
    in highp vec3 light_direction;

    layout(location = 0) out vec4 frag_color;

    void main() {
        float is_water = step(0.9, color.z);

        lowp vec3 ld_opposite = normalize(-light_direction);
        lowp float att = max(0.0, dot(normal, ld_opposite)) * 0.4;

        lowp float ambient = (ao * 1.5 + 0.5) * 0.6;

        lowp vec3 diffuse = //color *
        (
            ambient * vec3(0.6, 0.6, 1.0) +
            att * vec3(1.0, 0.65, 0.5)
        );

        lowp vec3 half_dir = (view_dir + ld_opposite) / 2.0;
        lowp float spec_angle = max(dot(half_dir, normal), 0.0);
        lowp float specular = pow(spec_angle, 48.0) * step(0.05, att) * is_water * 0.3;

        frag_color = vec4(
            diffuse + vec3(specular),
            1.0
        );
    }
    "#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec!["ao_texture".into()],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("view_proj", UniformType::Mat4),
                    UniformDesc::new("model", UniformType::Mat4),
                    UniformDesc::new("time", UniformType::Float1)
                ],
            },
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub view_proj: glam::Mat4,
        pub model: glam::Mat4,
        pub time: f32
    }
}