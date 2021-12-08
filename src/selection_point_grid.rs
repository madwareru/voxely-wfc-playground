use egui::Key::P;
use glam::Mat4;
use orom_miniquad::CursorIcon::Default;
use orom_miniquad::{Context, CullFace, PipelineParams, VertexStep};
use crate::tile_config::CellType;
use crate::voxel_mesh::VoxelMesh;

#[derive(Copy, Clone)]
pub enum SelectionPointState {
    NonSelectable,
    Selectable,
    Selected
}
impl SelectionPointState {
    fn to_shader_weight(self) -> f32 {
        match self {
            SelectionPointState::NonSelectable => 0.0,
            SelectionPointState::Selectable => 0.5,
            SelectionPointState::Selected => 1.0
        }
    }
}

pub struct SelectionPointGrid {
    width: usize,
    depth: usize,
    height: usize,
    faces_length: i32,
    grid_data: Vec<Vec<Vec<SelectionPointState>>>,
    selections: Vec<f32>,
    pipeline: orom_miniquad::Pipeline,
    bindings: orom_miniquad::Bindings,
}

impl SelectionPointGrid {
    pub fn new(
        ctx: &mut orom_miniquad::Context,
        point_mesh: super::geometry::MeshData,
        width: usize,
        depth: usize,
        height: usize
    ) -> Self {
        let grid_data = vec![vec![vec![SelectionPointState::NonSelectable; width + 1]; depth + 1]; height];

        let mut vertices = Vec::with_capacity(point_mesh.vertices.len() * 3);
        for v in point_mesh.vertices.iter() {
            vertices.extend_from_slice(&v.position);
        }
        let mut indices = Vec::with_capacity(point_mesh.faces.len() * 3);
        for f in point_mesh.faces.iter() {
            for ix in f.iter() {
                indices.push(*ix as u16);
            }
        }
        let faces_length = indices.len() as i32;

        let mut positions = Vec::with_capacity((width+1) * (height+1) * (depth));
        let mut selections = Vec::with_capacity((width+1) * (height+1) * (depth));

        for k in 0..height {
            for j in 0..=depth {
                for i in 0..=width {
                    positions.push(-1.0 + 2.0 * (i) as f32 - width as f32); // x
                    positions.push(2.0 * (k) as f32); // y
                    positions.push(1.0 - 2.0 * (j) as f32 + depth as f32); // z
                    selections.push(SelectionPointState::NonSelectable.to_shader_weight());
                }
            }
        }

        let mut selections_buffer = orom_miniquad::Buffer::stream(
            ctx,
            orom_miniquad::BufferType::VertexBuffer,
            selections.len() * std::mem::size_of::<f32>(),
        );
        selections_buffer.update(ctx, &selections);

        let bindings = orom_miniquad::Bindings {
            vertex_buffers: vec![
                orom_miniquad::Buffer::immutable(
                    ctx,
                    orom_miniquad::BufferType::VertexBuffer,
                    &vertices,
                ),
                orom_miniquad::Buffer::immutable(
                    ctx,
                    orom_miniquad::BufferType::VertexBuffer,
                    &positions,
                ),
                selections_buffer
            ],
            index_buffer: orom_miniquad::Buffer::immutable(
                ctx,
                orom_miniquad::BufferType::IndexBuffer,
                &indices
            ),
            images: vec![],
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
                orom_miniquad::BufferLayout {
                    step_func: VertexStep::PerInstance,
                    ..std::default::Default::default()
                },
                orom_miniquad::BufferLayout {
                    step_func: VertexStep::PerInstance,
                    ..std::default::Default::default()
                }
            ],
            &[
                orom_miniquad::VertexAttribute::with_buffer("pos", orom_miniquad::VertexFormat::Float3, 0),
                orom_miniquad::VertexAttribute::with_buffer("offset", orom_miniquad::VertexFormat::Float3, 1),
                orom_miniquad::VertexAttribute::with_buffer("selection_weight", orom_miniquad::VertexFormat::Float1, 2),
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

        Self {
            width,
            depth,
            height,
            faces_length,
            grid_data,
            selections,
            pipeline,
            bindings
        }
    }

    pub fn draw(&self, ctx: &mut orom_miniquad::Context, view_proj: Mat4, model: Mat4) {
        ctx.apply_pipeline(&self.pipeline);
        ctx.apply_bindings(&self.bindings);
        ctx.apply_uniforms(&shader::Uniforms { view_proj, model });
        ctx.draw(0, self.faces_length, ((self.width+1) * (self.depth+1) * self.height) as i32);
    }

    pub fn update(
        &mut self,
        ctx: &mut orom_miniquad::Context,
        voxel_mesh: &VoxelMesh,
        selection: Option<[usize; 3]>
    ) {
        let mut offset = 0;
        for k in 0..self.height {
            for j in 0..=self.depth {
                for i in 0..=self.width {
                    self.selections[offset] = if let CellType::Air = voxel_mesh.get_cell_data(i, j, k) {
                        SelectionPointState::NonSelectable
                    } else {
                        match (voxel_mesh.get_cell_data(i, j, k + 1), selection) {
                            (CellType::Air, Some([ii, jj, kk])) => {
                                if ii == i && jj == j && kk == k {
                                    SelectionPointState::Selected
                                } else {
                                    SelectionPointState::Selectable
                                }
                            },
                            (CellType::Air, None) => SelectionPointState::Selectable,
                            _ => SelectionPointState::NonSelectable
                        }
                    }.to_shader_weight();
                    offset += 1;
                }
            }
        }
        self.bindings.vertex_buffers[2].update(ctx, &self.selections);
    }
}

mod shader {
    use orom_miniquad::*;

    pub const VERTEX: &str = //language=glsl
        r#"#version 330
    layout(location = 0) in lowp vec3 pos;
    layout(location = 1) in lowp vec3 offset;
    layout(location = 2) in lowp float selection_weight;

    uniform mat4 view_proj;
    uniform mat4 model;

    out lowp vec4 color;

    void main() {
        vec4 new_pos = vec4(pos * selection_weight + offset, 1.0);
        mat4 mvp = view_proj * model;
        gl_Position = mvp * new_pos;

        color = vec4(
            vec3(0.8, 0.6, 0.2) * step(selection_weight, 0.6) * step(0.4, selection_weight) +
            vec3(1.0, 0.6, 0.6) * step(0.9, selection_weight),
            step(0.2, selection_weight)
        );
    }
    "#;

    pub const FRAGMENT: &str = //language=glsl
        r#"#version 330
    in lowp vec4 color;

    layout(location = 0) out vec4 frag_color;

    void main() {
        if (color.w < 0.1) discard;
        frag_color = color;
    }
    "#;

    pub fn meta() -> ShaderMeta {
        ShaderMeta {
            images: vec![],
            uniforms: UniformBlockLayout {
                uniforms: vec![
                    UniformDesc::new("view_proj", UniformType::Mat4),
                    UniformDesc::new("model", UniformType::Mat4)
                ],
            },
        }
    }

    #[repr(C)]
    pub struct Uniforms {
        pub view_proj: glam::Mat4,
        pub model: glam::Mat4
    }
}