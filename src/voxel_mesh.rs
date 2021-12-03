use std::collections::HashMap;
use glam::{Mat4};
use orom_miniquad::{CullFace, PipelineParams};
use crate::tile_config::{CellType, TileConfigEntry};

pub struct VoxelMesh {
    width: usize,
    depth: usize,
    height: usize,
    grid_data: Vec<Vec<Vec<super::tile_config::CellType>>>,
    pieces: HashMap<super::tile_config::TileConfigEntry, Vec<super::geometry::VertexData>>,
    pipeline: orom_miniquad::Pipeline,
    bindings: orom_miniquad::Bindings,
    is_dirty: bool,
    trivec: Vec<f32>,
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

        let bindings = orom_miniquad::Bindings {
            vertex_buffers: vec![
                orom_miniquad::Buffer::stream(
                    ctx,
                    orom_miniquad::BufferType::VertexBuffer,
                    300_000 * 9 * std::mem::size_of::<f32>(),
                )
            ],
            index_buffer: orom_miniquad::Buffer::immutable(
                ctx,
                orom_miniquad::BufferType::IndexBuffer,
                &indices
            ),
            images: Vec::with_capacity(1),
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

        Self {
            width, depth, height,
            grid_data,
            pieces,
            pipeline,
            bindings,
            is_dirty: false,
            trivec: Vec::with_capacity(300_000 * 9)
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

    pub fn update(&mut self, ctx: &mut orom_miniquad::Context) {
        if !self.is_dirty { return; }

        self.trivec.clear();

        for k in 0..self.height {
            for j in 0..self.depth {
                for i in 0..self.width {
                    let entry = self.get_tile_config_entry(i, j, k);
                    if let Some(verts) = self.pieces.get(&entry) {
                        let x = 2.0 * (i) as f32 - self.width as f32;
                        let y = 2.0 * (k) as f32;
                        let z = -2.0 * (j) as f32 + self.depth as f32;
                        for v in verts.iter() {
                            let position = [
                                v.position[0] + x,
                                v.position[1] + y,
                                v.position[2] + z
                            ];
                            self.trivec.extend_from_slice(&position);
                            self.trivec.extend_from_slice(&v.normal);
                            self.trivec.extend_from_slice(&v.color);
                        }
                    }
                }
            }
        }

        self.bindings.vertex_buffers[0].update(ctx, &self.trivec);
    }

    pub fn draw(&self, ctx: &mut orom_miniquad::Context, mvp: Mat4, time: f32) {
        if !self.trivec.is_empty() {
            ctx.apply_pipeline(&self.pipeline);
            ctx.apply_bindings(&self.bindings);
            ctx.apply_uniforms(&shader::Uniforms { mvp, time });
            ctx.draw(0, (self.trivec.len() / 9) as i32, 1);
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

    uniform mat4 mvp;

    out lowp vec3 normal_;
    out lowp vec3 color;
    out highp vec3 view_dir;

    void main() {
        vec4 new_pos = vec4(pos, 1.0);
        gl_Position = mvp * new_pos;
        view_dir = normalize(-gl_Position.xyz);
        normal_ = (mvp * vec4(normal0, 0.0)).xyz;
        color = color0;
    }
    "#;

    pub const FRAGMENT: &str = //language=glsl
        r#"#version 330
    in lowp vec3 normal_;
    in lowp vec3 color;
    in highp vec3 view_dir;

    uniform lowp float time;

    layout(location = 0) out vec4 frag_color;

    void main() {
        float is_water = step(0.9, color.z);
        lowp vec3 normal = normal_;

        lowp vec3 light_direction = normalize(vec3(1.3, -0.3, 1.0));
        lowp vec3 rim_light_direction = normalize(vec3(-1.0, -0.15, 0.0));

        lowp vec3 ld_opposite = -light_direction;
        lowp vec3 rim_ld_opposite = -rim_light_direction;
        lowp float att = max(0.0, dot(ld_opposite, normal)) * 0.6;
        lowp float rim_att = max(0.0, dot(rim_ld_opposite, normal)) * 0.3;

        lowp vec3 diffuse = color *
        (
            0.4 * vec3(0.8, 0.8, 1.0) +
            att * vec3(1.0, 0.9, 0.8) +
            rim_att * vec3(0.35, 0.25, 1.0)
        );

        lowp vec3 half_dir = (ld_opposite + view_dir) / 2.0;
        lowp float spec_angle = max(dot(half_dir, normal), 0.0);
        lowp float specular = pow(spec_angle, 12.0) * step(0.05, att) * is_water * 0.7;

        frag_color = vec4(
            diffuse + vec3(specular),
            1.0
        );
    }
    "#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec![],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("mvp", UniformType::Mat4),
                    UniformDesc::new("time", UniformType::Float1)
                ],
            },
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub mvp: glam::Mat4,
        pub time: f32
    }
}