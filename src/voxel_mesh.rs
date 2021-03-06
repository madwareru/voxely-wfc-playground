use std::collections::HashMap;
use egui::CursorIcon::Default;
use glam::{Mat4};
use orom_miniquad::{Context, CullFace, FilterMode, PipelineParams, TextureAccess, TextureFormat, TextureKind, TextureParams, TextureWrap};
use parry3d::bounding_volume::AABB;
use parry3d::math::{Point, Real, Vector};
use parry3d::query::{Ray, RayCast, RayIntersection};
use parry3d::shape::{FeatureId, Triangle};
use rayon::prelude::IntoParallelIterator;
use crate::geometry::VertexData;
use crate::tile_config::{CellType, TileConfigEntry, TileConfigEntryColumn};
use crate::utility::{StopWatch};
use rayon::iter::ParallelIterator;
use rayon::iter::IndexedParallelIterator;

const NUM_SAMPLES: usize = 32;
const SUB_GROUP_SIZE: usize = 1500;
const AO_TICKS_PER_FRAME: usize = 1;

#[derive(Copy, Clone)]
struct AaBbEntry {
    aabb: AABB,
    start_vertex_id: usize,
    end_vertex_id: usize
}

#[derive(Copy, Clone)]
enum InternalState {
    Pending,
    GeneratingAO { subgroup_no: usize, sample_no: usize },
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
    trivec_tmp: Vec<VertexData>,
    aabb_storage: Vec<Vec<Vec<Option<AaBbEntry>>>>
}

struct HeightPlane {
    height: f32
}

impl RayCast for HeightPlane {
    fn cast_local_ray_and_get_normal(&self, ray: &Ray, max_toi: Real, _solid: bool) -> Option<RayIntersection> {
        if ray.dir.y == 0.0 { None } else { Some(ray.dir.y) }
            .and_then(|d| Some(-(ray.origin.y - self.height) / d))
            .and_then(|toi| {
                if toi < 0.0 || toi > max_toi {
                    None
                } else {
                    Some(RayIntersection { toi, normal: [0.0, 1.0, 0.0].into(), feature: FeatureId::Face(0) })
                }
            })
    }
}

impl VoxelMesh {
    pub(crate) fn get_cell_by_ray(
         &self,
         origin: Point<Real>,
         direction: Vector<Real>
    ) -> Option<[usize; 3]> {
        let mut min_dist = None;
        let mut result = None;

        let ray = Ray::new(origin, direction);

        for height in (0..self.height).map(|it| it as f32 * 2.0) {
            let plane = HeightPlane { height };
            if let Some(t) = plane.cast_local_ray(&ray, 250.0, true) {
                let mut point: Point<Real> = ray.origin + ray.dir * t;
                point.x = (point.x + 1.0).round();
                point.z = (point.z - 1.0).round();

                let (i, j, k) = (
                    ((point.x + self.width as f32) / 2.0) as isize,
                    (-((point.z - self.depth as f32) / 2.0)) as isize,
                    ((point.y) / 2.0) as usize
                );
                if i < 0 || j < 0 { continue; }

                let (i, j) = (i as usize, j as usize);

                let mut min_d = 0.0;
                let mut min_ijk = None;
                for ii in (if i > 0 { i - 1 } else { 0 })..=(if i >= self.width { self.width } else { i + 1 }) {
                    for jj in (if j > 0 { j - 1 } else { 0 })..=(if j >= self.depth { self.depth } else { j + 1 }) {
                        if ii > self.width || jj > self.depth { continue; }
                        if self.get_cell_data(ii, jj, k).eq(&CellType::Air) ||
                            !self.get_cell_data(ii, jj, k+1).eq(&CellType::Air) {
                            continue;
                        }
                        let x = 2.0 * (ii) as f32 - self.width as f32;
                        let y = 2.0 * (k) as f32;
                        let z = -2.0 * (jj) as f32 + self.depth as f32;
                        let pp: Point<Real> = [x, y, z].into();
                        let d = point - pp;
                        let d = d.dot(&d);

                        if d > 0.01 { continue; }

                        match min_ijk {
                            None => {
                                min_d = d;
                                min_ijk = Some([ii, jj, k]);
                            },
                            Some(_) => {
                                if d < min_d {
                                    min_d = d;
                                    min_ijk = Some([ii, jj, k]);
                                }
                            }
                        };
                    }
                }

                if let Some([i, j, k]) = min_ijk {
                    if min_dist.is_none() || min_dist.unwrap() > t {
                        min_dist = Some(t);
                        result = Some([i, j, k]);
                    }
                }
            }
        }

        result
    }
}

pub enum CheckedSetPolicy {
    ReplaceWaterWithGroundOnFailure,
    DontSetOnFailure
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
                front_face_order: orom_miniquad::FrontFaceOrder::CounterClockwise,
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

        let aabb_storage = vec![vec![vec![Option::None;width];depth];height];

        Self {
            intenal_state: InternalState::Pending,
            width, depth, height,
            grid_data,
            pieces,
            pipeline,
            bindings,
            aabb_storage,
            is_dirty: false,
            trivec: Vec::with_capacity(900_000),
            trivec_tmp: Vec::with_capacity(900_000)
        }
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

    pub fn set_column_unchecked(
        &mut self,
        i: usize,
        j: usize,
        value: super::tile_config::CellType
    ) {
        self.is_dirty = true;
        let mut k = 0;
        while self.grid_data[k][j][i] != CellType::Air && k < self.height {
            self.grid_data[k][j][i] = value;
            k += 1;
        }
    }

    pub fn set_column_checked(
        &mut self,
        i: usize,
        j: usize,
        value: super::tile_config::CellType,
        set_policy: CheckedSetPolicy
    ) {
        let mut k = 0;
        let mut fail = false;
        while self.grid_data[k][j][i] != CellType::Air && k < self.height {
            if i < self.width && j < self.height {
                let tile_data = self.get_tile_config_entry(i, j, k);
                let tile_data = TileConfigEntry {
                    south_west: TileConfigEntryColumn {
                        bottom: value,
                        up: if tile_data.south_west.up == CellType::Air {
                            CellType::Air
                        } else {
                            value
                        }
                    },
                    ..tile_data
                };
                if !self.pieces.contains_key(&tile_data) {
                    fail = true;
                    break;
                }
            }
            if i > 0 && j > 0 {
                let tile_data = self.get_tile_config_entry(i-1, j-1, k);
                let tile_data = TileConfigEntry {
                    north_east: TileConfigEntryColumn {
                        bottom: value,
                        up: if tile_data.north_east.up == CellType::Air {
                            CellType::Air
                        } else {
                            value
                        }
                    },
                    ..tile_data
                };
                if !self.pieces.contains_key(&tile_data) {
                    fail = true;
                    break;
                }
            }
            if i > 0 {
                let tile_data = self.get_tile_config_entry(i-1, j, k);
                let tile_data = TileConfigEntry {
                    south_east: TileConfigEntryColumn {
                        bottom: value,
                        up: if tile_data.south_east.up == CellType::Air {
                            CellType::Air
                        } else {
                            value
                        }
                    },
                    ..tile_data
                };
                if !self.pieces.contains_key(&tile_data) {
                    fail = true;
                    break;
                }
            }
            if j > 0 {
                let tile_data = self.get_tile_config_entry(i, j-1, k);
                let tile_data = TileConfigEntry {
                    north_west: TileConfigEntryColumn {
                        bottom: value,
                        up: if tile_data.north_west.up == CellType::Air {
                            CellType::Air
                        } else {
                            value
                        }
                    },
                    ..tile_data
                };
                if !self.pieces.contains_key(&tile_data) {
                    fail = true;
                    break;
                }
            }
            k += 1;
        }
        match (fail, set_policy) {
            (false, _) => self.set_column_unchecked(i, j, value),
            _ => {}
        }
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

    pub fn get_cell_data(&self, i: usize, j: usize, k: usize) -> CellType {
        self.grid_data[k][j][i]
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

    pub fn get_ao_baking_progress(&self) -> Option<f32> {
        match self.intenal_state {
            InternalState::GeneratingAO { sample_no, .. } => {
                Some(
                    (
                        100.0 * sample_no as f32 / (NUM_SAMPLES as f32)
                    ).clamp(0.0, 100.0)
                )
            },
            _ => None
        }
    }

    pub fn update(&mut self, ctx: &mut orom_miniquad::Context, ao_generation_is_on: bool) {
        puffin::profile_function!();
        if self.is_dirty {
            self.intenal_state = InternalState::Pending;
        }
        match self.intenal_state {
            InternalState::Pending => self.prepare_mesh(ctx),
            InternalState::GeneratingAO { .. } if ao_generation_is_on => {
                for _ in 0..AO_TICKS_PER_FRAME {
                    self.tick_ao_generation(ctx);
                }
            },
            _ => {}
        }
    }

    fn tick_ao_generation(&mut self, ctx: &mut Context) {
        if let InternalState::GeneratingAO { subgroup_no, sample_no } = self.intenal_state {
            if self.is_dirty {
                self.intenal_state = InternalState::Pending;
                return;
            }

            let dir = super::utility::sun_flower_direction_on_a_unit_sphere(NUM_SAMPLES, sample_no);
            let current_offset = subgroup_no * SUB_GROUP_SIZE;
            if current_offset < self.trivec.len() {
                let current_range = current_offset..(current_offset + SUB_GROUP_SIZE)
                    .min(self.trivec.len());

                let mut trivec_tmp = std::mem::take(&mut self.trivec_tmp);

                (&mut trivec_tmp[current_range]).into_par_iter().enumerate().for_each(|(idx, vx): (_, &mut VertexData)| {
                    let ao: f32 = self.integrate_vertex(idx + current_offset, dir);
                    let prev_ao = self.trivec[idx + current_offset].ao;
                    vx.ao += (ao - prev_ao) / (sample_no + 1) as f32;
                });

                self.trivec_tmp = trivec_tmp;

                {
                    self.trivec.clear();
                    self.trivec.extend_from_slice(&self.trivec_tmp);
                }
                self.bindings.vertex_buffers[0].update(ctx, &self.trivec);
                self.intenal_state = if current_offset + SUB_GROUP_SIZE >= self.trivec.len() {
                    if sample_no < (NUM_SAMPLES - 1) {
                        InternalState::GeneratingAO { subgroup_no: 0, sample_no: sample_no + 1 }
                    } else {
                        InternalState::Ready
                    }
                } else {
                    InternalState::GeneratingAO { subgroup_no: subgroup_no + 1, sample_no }
                };
            } else {
                if sample_no < (NUM_SAMPLES - 1) {
                    self.intenal_state = InternalState::GeneratingAO { subgroup_no: 0, sample_no: sample_no + 1 };
                } else {
                    self.bindings.vertex_buffers[0].update(ctx, &self.trivec);
                    self.intenal_state = InternalState::Ready;
                }
            }
        }
    }

    fn prepare_mesh(&mut self, ctx: &mut Context) {
        if !self.is_dirty { return; }
        self.is_dirty = false;

        self.trivec.clear();
        self.trivec_tmp.clear();

        for entry in self.aabb_storage.iter_mut().flat_map(|it| it.iter_mut()) {
            entry.fill(Option::None);
        }

        for k in 0..self.height {
            for j in 0..self.depth {
                for i in 0..self.width {
                    let entry = self.get_tile_config_entry(i, j, k);
                    if let Some(verts) = self.pieces.get(&entry) {
                        let x = 2.0 * (i) as f32 - self.width as f32;
                        let y = 2.0 * (k) as f32;
                        let z = -2.0 * (j) as f32 + self.depth as f32;

                        self.aabb_storage[k][j][i] =
                            Some(AaBbEntry {
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
                    } else if !(
                        entry.eq(&TileConfigEntry::all_of_type(CellType::Air)) ||
                        entry.eq(&TileConfigEntry::all_of_type(CellType::Ground)) ||
                        entry.eq(&TileConfigEntry::all_of_type(CellType::Water))
                    ) {
                        // println!("A matching tile was not found for {:?}", &entry);
                    }
                }
            }
        }
        self.trivec_tmp.clear();
        self.trivec_tmp.extend(self.trivec.iter().cloned());
        self.bindings.vertex_buffers[0].update(ctx, &self.trivec);
        self.intenal_state = InternalState::GeneratingAO { subgroup_no: 0, sample_no: 0 };
    }

    fn integrate_vertex(&self, ix: usize, ray_dir: Vector<Real>) -> f32 {
        let origin: Point<Real> = self.trivec[ix].position.into();
        let dir: Vector<Real> = self.trivec[ix].normal.into();
        let dir = dir.normalize();

        for _ in 0..4 {
            let mut offset = super::utility::random_direction_on_a_unit_sphere();
            while offset.dot(&dir) < 0.25 {
                offset = super::utility::random_direction_on_a_unit_sphere();
            }
            if self.integrate_vertex_impl(
                ray_dir,
                origin + offset * 0.01,
                dir
            ) {
                return 0.0;
            }
        }
        1.0
    }

    fn integrate_vertex_impl(&self, ray_dir: Vector<Real>, origin: Point<Real>, dir: Vector<Real>) -> bool {
        const RAY_LENGTH: f32 = 256.0;

        let next_ray = if dir.dot(&ray_dir) >= 0.0 {
            Ray::new(origin, ray_dir)
        } else {
            Ray::new(origin, -ray_dir)
        };

        let range_p: Point<Real> = [self.width as f32, self.height as f32 * 2.0, self.depth as f32].into();
        for j0 in 0..4 {
            for i0 in 0..4 {
                let aabb = AABB {
                    mins: [
                        -range_p.x + i0 as f32 * range_p.x / 2.0 - 1.0,
                        - 1.0,
                        range_p.z - (j0 + 1) as f32 * range_p.z / 2.0 - 1.0
                    ].into(),
                    maxs: [
                        -range_p.x + (i0 + 1) as f32 * range_p.x / 2.0 + 1.0,
                        range_p.y + 1.0,
                        range_p.z - j0 as f32 * range_p.z / 2.0 + 1.0
                    ].into()
                };
                if !aabb.intersects_local_ray(&next_ray, RAY_LENGTH) {
                    continue;
                }

                for k in 0..self.height {
                    for j in (j0 * self.depth/4)..((j0 + 1) * self.depth/4).min(self.depth) {
                        for i in (i0 * self.width/4)..((i0 + 1) * self.width/4).min(self.width) {
                            match self.aabb_storage[k][j][i] {
                                Some(aabb_entry)
                                if aabb_entry.aabb.intersects_local_ray(&next_ray, RAY_LENGTH) => {
                                    for offset in (aabb_entry.start_vertex_id..=aabb_entry.end_vertex_id).step_by(3) {
                                        {
                                            let tri_aabb = AABB {
                                                mins: [
                                                    self.trivec[offset].position[0]
                                                        .min(self.trivec[offset + 1].position[0])
                                                        .min(self.trivec[offset + 2].position[0]),
                                                    self.trivec[offset].position[1]
                                                        .min(self.trivec[offset + 1].position[1])
                                                        .min(self.trivec[offset + 2].position[1]),
                                                    self.trivec[offset].position[2]
                                                        .min(self.trivec[offset + 1].position[2])
                                                        .min(self.trivec[offset + 2].position[2])
                                                ].into(),
                                                maxs: [
                                                    self.trivec[offset].position[0]
                                                        .max(self.trivec[offset + 1].position[0])
                                                        .max(self.trivec[offset + 2].position[0]),
                                                    self.trivec[offset].position[1]
                                                        .max(self.trivec[offset + 1].position[1])
                                                        .max(self.trivec[offset + 2].position[1]),
                                                    self.trivec[offset].position[2]
                                                        .max(self.trivec[offset + 1].position[2])
                                                        .max(self.trivec[offset + 2].position[2])
                                                ].into()
                                            };

                                            if !tri_aabb.intersects_local_ray(&next_ray, RAY_LENGTH) {
                                                continue;
                                            }
                                        }

                                        let next_triangle = Triangle::new(
                                            self.trivec[offset].position.into(),
                                            self.trivec[offset + 1].position.into(),
                                            self.trivec[offset + 2].position.into()
                                        );

                                        let is_not_culled = -(next_triangle.normal().unwrap()).dot(&next_ray.dir) < 0.0;

                                        if is_not_culled && next_triangle.intersects_local_ray(&next_ray, RAY_LENGTH) {
                                            return true;
                                        }
                                    }
                                },
                                _ => {}
                            }
                        }
                    }
                }
            }
        }
        false
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

        light_direction = (model * vec4(normalize(vec3(0.3, -0.3, 0.25)), 0.0)).xyz;

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

        lowp float ambient = (ao * 1.5 + 0.6) * 0.6;

        lowp vec3 diffuse = color *
        (
            ambient * vec3(0.6, 0.6, 1.0) +
            att * vec3(1.0, 0.65, 0.5)
        );

        frag_color = vec4(diffuse, 1.0);
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
