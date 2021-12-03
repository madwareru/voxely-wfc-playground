use std::collections::HashMap;
use std::time::Instant;
use orom_miniquad::{Context, KeyCode, KeyMods, MouseButton, PassAction};
use simple_tiled_wfc::errors::WfcError;
use simple_tiled_wfc::voxel_generation::{WfcContext, WfcModule};
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use ron::de::from_reader;
use crate::tile_config::{CellType, TileConfigEntry, TileConfigEntryColumn};
use crate::utility::StopWatch;
use crate::voxel_mesh::VoxelMesh;

type CustomBitSet = [u8; 32];

const TILES_PLY_CONTENT: &str = include_str!("../assets/wfc_parts.ply");
const CONFIG_BYTES: &[u8] = include_bytes!("../assets/tile_config.ron");

pub struct CollapseDemoStage {
    instant: Instant,
    x_size: usize,
    y_size: usize,
    z_size: usize,
    modules: Vec<WfcModule<CustomBitSet>>,
    water_plane_module_id: usize,
    air_module_id: usize,
    piece_tile_configs: Vec<TileConfigEntry>,
    voxel_mesh: VoxelMesh,
    view_proj: Mat4,
    inv_view_proj: Mat4,
    mouse_x: f32,
    mouse_y: f32,
    grab_start_mouse_x: Option<f32>,
    grab_start_mouse_y: Option<f32>,
    can_grab: bool,
    ry: f32,
    time: f32,
    grab_start_ry: Option<f32>
}

impl CollapseDemoStage {
    pub fn new(ctx: &mut orom_miniquad::Context, sizes: [usize; 3]) -> Self {
        let tile_config: super::tile_config::TileConfig = from_reader(CONFIG_BYTES).unwrap();

        let [x_size, y_size, z_size] = sizes;
        let mesh = super::geometry::MeshData::parse_ply(TILES_PLY_CONTENT);
        let pieces = mesh.split(tile_config.tile_step, tile_config.tile_width);

        let mut piece_tile_configs = Vec::new();
        for piece_entry in pieces.iter().step_by(4) {
            let idx = tile_config.stride * (piece_entry.0.x as usize) + piece_entry.0.z as usize;
            let mapping = tile_config.mappings[&idx];
            piece_tile_configs.push(mapping);
            let mapping = mapping.rotate();
            piece_tile_configs.push(mapping);
            let mapping = mapping.rotate();
            piece_tile_configs.push(mapping);
            let mapping = mapping.rotate();
            piece_tile_configs.push(mapping);
        }
        piece_tile_configs.push(TileConfigEntry::all_of_type(CellType::Air));
        piece_tile_configs.push(TileConfigEntry::all_of_type(CellType::Ground));
        piece_tile_configs.push(TileConfigEntry::all_of_type(CellType::Water));
        for spec_water_tile in [
            TileConfigEntry{
                north_east: TileConfigEntryColumn { bottom: CellType::Ground, up: CellType::Ground },
                north_west: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Water },
                south_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Water },
                south_west: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Water }
            },
            TileConfigEntry{
                north_east: TileConfigEntryColumn { bottom: CellType::Ground, up: CellType::Ground },
                north_west: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Water },
                south_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Water },
                south_west: TileConfigEntryColumn { bottom: CellType::Ground, up: CellType::Ground }
            },
            TileConfigEntry{
                north_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Water },
                north_west: TileConfigEntryColumn { bottom: CellType::Ground, up: CellType::Ground },
                south_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Water },
                south_west: TileConfigEntryColumn { bottom: CellType::Ground, up: CellType::Ground }
            },
            TileConfigEntry{
                north_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Water },
                north_west: TileConfigEntryColumn { bottom: CellType::Ground, up: CellType::Ground },
                south_east: TileConfigEntryColumn { bottom: CellType::Ground, up: CellType::Ground },
                south_west: TileConfigEntryColumn { bottom: CellType::Ground, up: CellType::Ground }
            }
        ] {
            piece_tile_configs.push(spec_water_tile);
            let spec_water_tile = spec_water_tile.rotate();
            piece_tile_configs.push(spec_water_tile);
            let spec_water_tile = spec_water_tile.rotate();
            piece_tile_configs.push(spec_water_tile);
            let spec_water_tile = spec_water_tile.rotate();
            piece_tile_configs.push(spec_water_tile);
        }

        let water_plane_module_id = piece_tile_configs
            .iter()
            .enumerate()
            .find(|(_, it)| {
                let it = *it;
                let it = *it;
                it.eq(&TileConfigEntry{
                    north_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air },
                    north_west: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air },
                    south_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air },
                    south_west: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air }
                })
            })
            .map(|it| it.0)
            .unwrap();

        let mut modules: Vec<WfcModule<CustomBitSet>> = vec![WfcModule::new(); piece_tile_configs.len()];

        let air_module_id = pieces.len();

        for i in 0..piece_tile_configs.len() {
            let ei = &piece_tile_configs[i];
            for j in 0..piece_tile_configs.len() {
                let other_ei = &piece_tile_configs[j];

                if ei.is_east_neighbour(other_ei) {
                    modules[i].add_east_neighbour(j);
                }
                if ei.is_west_neighbour(other_ei) {
                    modules[i].add_west_neighbour(j);
                }

                if ei.is_north_neighbour(other_ei) {
                    modules[i].add_north_neighbour(j);
                }
                if ei.is_south_neighbour(other_ei) {
                    modules[i].add_south_neighbour(j);
                }

                if ei.is_up_neighbour(other_ei) {
                    modules[i].add_upper_neighbour(j);
                }
                if ei.is_bottom_neighbour(other_ei) {
                    modules[i].add_bottom_neighbour(j);
                }
            }
        }

        let mut pieces_for_voxel_mesh = HashMap::new();
        for i in 0..pieces.len() {
            let mut vec = Vec::with_capacity(pieces[i].1.faces.len() * 3);
            for vertex_id in pieces[i].1.faces.iter().flat_map(|it| *it) {
                vec.push(pieces[i].1.vertices[vertex_id]);
            }
            pieces_for_voxel_mesh.insert(piece_tile_configs[i], vec);
        }

        let mut voxel_mesh = VoxelMesh::new(
            ctx,
            pieces_for_voxel_mesh,
            x_size,
            z_size,
            y_size
        );

        for i in 0..x_size {
            for j in 0..z_size {
                voxel_mesh.set_tile_unchecked(i, j, 0, TileConfigEntry {
                    north_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air },
                    north_west: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air },
                    south_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air },
                    south_west: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air }
                })
            }
        }

        let view_proj = pitch_yaw_camera_view_mat(
            [0.0, 25.0, -25.0].into(), 45.0f32.to_radians(), 0.0,
            60.0f32.to_radians(), ctx.screen_size().0 / ctx.screen_size().1, 0.01, 250.0
        );
        let inv_view_proj = view_proj.inverse();

        Self {
            instant: Instant::now(),
            x_size,
            y_size,
            z_size,
            piece_tile_configs,
            modules,
            water_plane_module_id,
            air_module_id,
            voxel_mesh,
            view_proj,
            inv_view_proj,
            mouse_x: 0.0,
            mouse_y: 0.0,
            can_grab: false,
            grab_start_mouse_x: None,
            grab_start_mouse_y: None,
            time: 0.0,
            ry: 0.0,
            grab_start_ry: None
        }
    }

    pub fn generate_map(&mut self) {
        let _sw = StopWatch::named("generating a map");
        let collapse_result = self.collapse();
        if let Ok(result_indices) = collapse_result {
            let row_size = self.x_size;
            let stride = row_size * self.z_size;
            for idx in 0..result_indices.len() {
                let i = (idx % stride) % row_size;
                let j = (idx % stride) / row_size;
                let k = idx / stride;
                self.voxel_mesh.set_tile_unchecked(i, j, k, self.piece_tile_configs[result_indices[idx]]);
            }
        }
    }

    fn collapse(&mut self) -> Result<Vec<usize>, WfcError> {
        let mut wfc_context = WfcContext::new(&self.modules, self.x_size, self.z_size, self.y_size);
        let centre_z = self.z_size as f32 / 2.0;
        let centre_x = self.x_size as f32 / 2.0;
        for j in 0..self.z_size {
            for i in 0..self.x_size {
                let dz = j as f32 - centre_z;
                let dx = i as f32 - centre_x;
                if dz*dz + dx*dx > 75.0 {
                    for k in 0..self.y_size - 1 {
                        wfc_context.set_module(
                            k, i, j,
                            if k == 0 { self.water_plane_module_id } else { self.air_module_id }
                        );
                    }
                }
                wfc_context.set_module(self.y_size - 1, i, j, self.air_module_id);
            }
        }
        wfc_context.collapse(10000)
    }
}

impl orom_miniquad::EventHandler for CollapseDemoStage {
    fn update(&mut self, ctx: &mut Context) {
        let next_instant = Instant::now();
        let elapsed = next_instant - self.instant;
        self.time += elapsed.as_secs_f32();
        self.instant = next_instant;

        if let (Some(grab_start_x), Some(_grab_start_y), Some(grab_start_ry)) =
            (self.grab_start_mouse_x, self.grab_start_mouse_y, self.grab_start_ry) {
            self.ry = grab_start_ry + (self.mouse_x - grab_start_x) / 100.0
        }

        self.voxel_mesh.update(ctx);

        self.view_proj = pitch_yaw_camera_view_mat(
            [0.0, 25.0, -25.0].into(), 45.0f32.to_radians(), 0.0,
            60.0f32.to_radians(), ctx.screen_size().0 / ctx.screen_size().1, 0.01, 250.0
        );
        self.inv_view_proj = self.view_proj.inverse();
    }

    fn draw(&mut self, ctx: &mut Context) {
        let model = Mat4::from_rotation_y(self.ry);

        let mvp = self.view_proj * model;

        ctx.begin_default_pass(PassAction::Clear {
            color: Some((0.10, 0.06, 0.18, 1.0)),
            depth: Some(1.0),
            stencil: None,
        });

        self.voxel_mesh.draw(ctx, mvp, self.time);

        ctx.end_render_pass();
        ctx.commit_frame();
    }

    fn mouse_motion_event(&mut self, _ctx: &mut Context, x: f32, y: f32) {
        self.mouse_x = x;
        self.mouse_y = y;
    }

    fn mouse_button_down_event(&mut self, _ctx: &mut Context, button: MouseButton, _x: f32, _y: f32) {
        if let( true, MouseButton::Left ) = (self.can_grab, button) {
            self.grab_start_mouse_x = Some(self.mouse_x);
            self.grab_start_mouse_y = Some(self.mouse_y);
            self.grab_start_ry = Some(self.ry);
        }
    }

    fn mouse_button_up_event(&mut self, _ctx: &mut Context, button: MouseButton, _x: f32, _y: f32) {
        if let( true, MouseButton::Left ) = (self.can_grab, button) {
            self.grab_start_mouse_x = None;
            self.grab_start_mouse_y = None;
            self.grab_start_ry = None;
        }
    }

    fn key_down_event(&mut self, ctx: &mut Context, keycode: KeyCode, _keymods: KeyMods, _repeat: bool,) {
        match keycode {
            KeyCode::Escape => ctx.quit(),
            KeyCode::Space => {
                self.can_grab = true;
            }
            _ => {}
        }
    }

    fn key_up_event(&mut self, _ctx: &mut Context, keycode: KeyCode, _keymods: KeyMods) {
        match keycode {
            KeyCode::Enter => { self.generate_map(); },
            KeyCode::Space => {
                self.can_grab = false;
                self.grab_start_mouse_x = None;
                self.grab_start_mouse_y = None;
                self.grab_start_ry = None;
            }
            _ => {}
        }
    }
}

fn pitch_yaw_camera_view_mat(
    position: Vec3,
    pitch: f32,
    yaw: f32,
    fov: f32,
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32
) -> Mat4 {
    let up_vector = [0.0, 1.0, 0.0].into();
    let view_vector : Vec4 = [0.0, 0.0, 1.0, 0.0].into();
    let view_vector = Mat4::from_rotation_y(yaw) * Mat4::from_rotation_x(pitch) * view_vector;

    let center = position + view_vector.xyz();

    let lookat_mat = glam::Mat4::look_at_rh(
        position,
        center,
        up_vector
    );
    let proj = Mat4::perspective_rh_gl(fov, aspect_ratio, z_near, z_far);
    proj * lookat_mat
}