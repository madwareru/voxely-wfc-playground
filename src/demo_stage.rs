use std::collections::HashMap;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::thread;
use std::time::Instant;
use orom_miniquad::{Context, KeyCode, KeyMods, MouseButton, PassAction};
use simple_tiled_wfc::errors::WfcError;
use simple_tiled_wfc::voxel_generation::{WfcContext, WfcModule};
use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use orom_miniquad::egui_integration::EguiMq;
use ron::de::from_reader;
use egui::{Color32, TextStyle, FontDefinitions, FontFamily, Align2, Order};
use parry3d::math::{Point, Real, Vector};
use crate::selection_point_grid::SelectionPointGrid;
use crate::tile_config::{CellType, TileConfigEntry, TileConfigEntryColumn};
use crate::utility::StopWatch;
use crate::voxel_mesh::VoxelMesh;

type CustomBitSet = [u8; 32];

const TILES_PLY_CONTENT: &str = include_str!("../assets/wfc_parts.ply");
const SELECTION_POINT_PLY_CONTENT: &str = include_str!("../assets/selection_point.ply");
const CONFIG_BYTES: &[u8] = include_bytes!("../assets/tile_config.ron");
pub const JETBRAINS_MONO_FONT: &[u8] = include_bytes!("../assets/JetBrainsMono-Medium.ttf");

const NEAR_PLANE: f32 = 0.01;
const FAR_PLANE: f32 = 250.0;
const MAX_X_ROT: f32 = std::f32::consts::PI / 4.0;
const MIN_X_ROT: f32 = -std::f32::consts::PI / 6.0;

trait Lerp : Copy {
    fn lerp(self, other: &Self, t: f32) -> Self;
    fn strange_lerp(self, other: &Self, t: f32, strangeness: f32) -> Self {
        let strangeness = strangeness.clamp(0.0, 1.0);
        let usual_lerp = self.lerp(other, t);
        let t = (1.0 + (std::f32::consts::PI * (1.0 + t)).cos()) / 2.0;
        usual_lerp.lerp(&self.lerp(other, t), strangeness)
    }
}

impl Lerp for f32 {
    fn lerp(self, other: &Self, t: f32) -> Self {
        let t = t.clamp(0.0, 1.0);
        self * (1.0 - t) + *other * t
    }
}

pub struct CollapseDemoStage {
    egui: EguiMq,
    instant: Instant,
    x_size: usize,
    y_size: usize,
    z_size: usize,
    modules: Vec<WfcModule<CustomBitSet>>,
    water_plane_module_id: usize,
    air_module_id: usize,
    piece_tile_configs: Vec<TileConfigEntry>,
    voxel_mesh: VoxelMesh,
    selection_grid: SelectionPointGrid,
    view_proj: Mat4,
    mouse_x: f32,
    mouse_y: f32,
    grab_start_mouse_x: Option<f32>,
    grab_start_mouse_y: Option<f32>,
    compound_results_receiver: Receiver<Result<Vec<usize>, WfcError>>,
    compound_results_transmitter: Sender<Result<Vec<usize>, WfcError>>,
    can_grab: bool,
    ry: f32,
    rx: f32,
    time: f32,
    grab_start_ry: Option<f32>,
    grab_start_rx: Option<f32>
}

impl CollapseDemoStage {
    pub fn new(ctx: &mut orom_miniquad::Context, sizes: [usize; 3]) -> Self {
        let egui = EguiMq::new(ctx);
        let mut fonts = FontDefinitions::default();
        fonts.font_data
            .insert("JetBrains Mono".to_owned(), std::borrow::Cow::Borrowed(JETBRAINS_MONO_FONT));

        fonts.fonts_for_family
            .get_mut(&FontFamily::Proportional)
            .unwrap()
            .insert(0, "JetBrains Mono".to_owned());

        fonts.fonts_for_family
            .get_mut(&FontFamily::Monospace)
            .unwrap()
            .insert(0, "JetBrains Mono".to_owned());

        if let Some(setting) = fonts.family_and_size.get_mut(&TextStyle::Button) {
            *setting = (FontFamily::Monospace, 24.0);
        }
        if let Some(setting) = fonts.family_and_size.get_mut(&TextStyle::Heading) {
            *setting = (FontFamily::Monospace, 32.0);
        }
        if let Some(setting) = fonts.family_and_size.get_mut(&TextStyle::Monospace) {
            *setting = (FontFamily::Monospace, 24.0);
        }
        if let Some(setting) = fonts.family_and_size.get_mut(&TextStyle::Body) {
            *setting = (FontFamily::Monospace, 24.0);
        }
        if let Some(setting) = fonts.family_and_size.get_mut(&TextStyle::Small) {
            *setting = (FontFamily::Monospace, 24.0);
        }

        egui.egui_ctx().set_fonts(fonts);

        let tile_config: super::tile_config::TileConfig = from_reader(CONFIG_BYTES).unwrap();

        let [x_size, y_size, z_size] = sizes;

        let selection_point_mesh =  super::geometry::MeshData::parse_ply(SELECTION_POINT_PLY_CONTENT);
        let selection_grid = SelectionPointGrid::new(ctx, selection_point_mesh, x_size, z_size, y_size);

        let mesh = super::geometry::MeshData::parse_ply(TILES_PLY_CONTENT);
        let pieces = mesh.split(tile_config.tile_step, tile_config.tile_width);

        let (compound_results_transmitter, compound_results_receiver) = channel();

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
            .find_map(|(id, it)| {
                let it = *it;
                if it.eq(&TileConfigEntry{
                    north_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air },
                    north_west: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air },
                    south_east: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air },
                    south_west: TileConfigEntryColumn { bottom: CellType::Water, up: CellType::Air }
                }) {
                    Some(id)
                } else {
                    None
                }
            })
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

        let voxel_mesh = VoxelMesh::new(
            ctx,
            pieces_for_voxel_mesh,
            x_size,
            z_size,
            y_size
        );

        let view_proj = pitch_yaw_camera_view_mat(
            [0.0, 25.0, -25.0].into(), 45.0f32.to_radians(), 0.0,
            60.0f32.to_radians(), ctx.screen_size().0 / ctx.screen_size().1, 0.01, 250.0
        );

        let mut res = Self {
            egui,
            instant: Instant::now(),
            x_size,
            y_size,
            z_size,
            compound_results_transmitter,
            compound_results_receiver,
            piece_tile_configs,
            modules,
            water_plane_module_id,
            air_module_id,
            voxel_mesh,
            selection_grid,
            view_proj,
            mouse_x: 0.0,
            mouse_y: 0.0,
            can_grab: false,
            grab_start_mouse_x: None,
            grab_start_mouse_y: None,
            time: 0.0,
            ry: 0.0,
            rx: 0.3,
            grab_start_ry: None,
            grab_start_rx: None
        };

        res.generate_map();

        res
    }

    pub fn generate_map(&mut self) {
        let _sw = StopWatch::named("generating a map");
        self.collapse();
    }

    fn ui(&mut self) {
        let egui_ctx = self.egui.egui_ctx().clone();

        if let Some(progress) = self.voxel_mesh.get_ao_baking_progress() {
            egui::Area::new("ao")
                .anchor(Align2::CENTER_BOTTOM, [0.0, 0.0])
                .order(Order::Background)
                .interactable(false)
                .movable(false)
                .show(&egui_ctx, |ui| {
                    ui.add(egui::Label::new(format!("Generating ambient occlusion: {}%", progress.round()))
                        .heading()
                        .background_color(Color32::from_rgba_unmultiplied(255, 255, 255, 5))
                        .text_color(Color32::from_rgb(0, 0, 0))
                        .strong()
                    );
                });
        }

        egui::Window::new("menu")
            .anchor(Align2::CENTER_TOP, [0.0, 0.0])
            .show(&egui_ctx, |ui| {
                ui.vertical_centered_justified(|ui| {
                    if ui.button("Generate").clicked() {
                        self.generate_map();
                    }

                    ui.separator();

                    if ui.button("Quit (esc)").clicked() {
                        std::process::exit(0);
                    }
                });
            });
    }

    fn collapse(&mut self) {
        let tx = self.compound_results_transmitter.clone();
        let modules = self.modules.clone();

        let (x_size, z_size, y_size) = (self.x_size, self.z_size, self.y_size);
        let (water_plane_module_id, air_module_id) = (self.water_plane_module_id, self.air_module_id);

        thread::spawn(move || {
            let tx = tx;
            let modules = modules;
            let mut wfc_context = WfcContext::new(&modules, x_size, z_size, y_size);
            let centre_z = z_size as f32 / 2.0;
            let centre_x = x_size as f32 / 2.0;
            for j in 0..z_size {
                for i in 0..x_size {
                    let dz = j as f32 - centre_z;
                    let dx = i as f32 - centre_x;
                    if dz*dz + dx*dx > 75.0 {
                        for k in 0..y_size - 1 {
                            wfc_context.set_module(
                                k, i, j,
                                if k == 0 { water_plane_module_id } else { air_module_id }
                            );
                        }
                    }
                    wfc_context.set_module(y_size - 1, i, j, air_module_id);
                }
            }
            let result = wfc_context.collapse(10000);
            tx.send(result).unwrap();
        });
    }

    fn get_model_matrix(&self) -> Mat4 {
        let t = 0.0.strange_lerp(&1.0, self.rx, 0.3);
        let x_a = MIN_X_ROT.strange_lerp(&MAX_X_ROT, t, 0.3);

        Mat4::from_rotation_x(-x_a) * Mat4::from_rotation_y(-self.ry)
    }
}

impl orom_miniquad::EventHandler for CollapseDemoStage {
    fn update(&mut self, ctx: &mut Context) {
        let next_instant = Instant::now();
        let elapsed = next_instant - self.instant;
        self.time += elapsed.as_secs_f32();
        self.instant = next_instant;

        if let (Some(grab_start_x), Some(grab_start_y), Some(grab_start_ry), Some(grab_start_rx)) =
            (self.grab_start_mouse_x, self.grab_start_mouse_y, self.grab_start_ry, self.grab_start_rx) {
            self.ry = grab_start_ry + (self.mouse_x - grab_start_x) / 100.0;
            self.rx = (grab_start_rx + (self.mouse_y - grab_start_y) / 300.0).clamp(0.0, 1.0);
        }

        match self.compound_results_receiver.try_recv() {
            Ok(Ok(result_indices)) => {
                let row_size = self.x_size;
                let stride = row_size * self.z_size;
                for idx in 0..result_indices.len() {
                    let i = (idx % stride) % row_size;
                    let j = (idx % stride) / row_size;
                    let k = idx / stride;
                    self.voxel_mesh.set_tile_unchecked(i, j, k, self.piece_tile_configs[result_indices[idx]]);
                }
            }
            Ok(Err(_)) => {
                println!("Error during generation!")
            }
            _ => {}
        }

        self.voxel_mesh.update(ctx);

        self.view_proj = pitch_yaw_camera_view_mat(
            [0.0, 25.0, -25.0].into(), 45.0f32.to_radians(), 0.0,
            60.0f32.to_radians(), ctx.screen_size().0 / ctx.screen_size().1, NEAR_PLANE, FAR_PLANE
        );
        let remapped_mouse_coords = (
            (self.mouse_x / ctx.screen_size().0 - 0.5) * 2.0,
            -(self.mouse_y / ctx.screen_size().1 - 0.5) * 2.0
        );

        let v_near = unproj_vec(
            [remapped_mouse_coords.0, remapped_mouse_coords.1, -0.2].into(),
            self.view_proj * self.get_model_matrix()
        );
        let v_far = unproj_vec(
            [remapped_mouse_coords.0, remapped_mouse_coords.1, 0.2].into(),
            self.view_proj * self.get_model_matrix()
        );

        let origin: Point<Real> = [v_near.x, v_near.y, v_near.z].into();

        let dir = v_far - v_near;
        let dir: Vector<Real> = [dir.x, dir.y, dir.z].into();
        let dir = dir.normalize();

        let selected_tile = self.voxel_mesh.get_cell_by_ray(origin, dir);
        self.selection_grid.update(ctx, &self.voxel_mesh, selected_tile.map(|it| it.1));
    }

    fn draw(&mut self, ctx: &mut Context) {
        self.egui.begin_frame(ctx);
        self.ui();
        self.egui.end_frame(ctx);

        // Draw things behind egui here

        {
            let model = self.get_model_matrix();

            ctx.begin_default_pass(PassAction::Clear {
                color: Some((0.10, 0.06, 0.18, 1.0)),
                depth: Some(1.0),
                stencil: None,
            });

            self.voxel_mesh.draw(ctx, self.view_proj, model, self.time);
            self.selection_grid.draw(ctx, self.view_proj, model);

            ctx.end_render_pass();
        }

        self.egui.draw(ctx);

        // Draw things in front of egui here

        ctx.commit_frame();
    }

    fn mouse_motion_event(&mut self, ctx: &mut Context, x: f32, y: f32) {
        self.mouse_x = x;
        self.mouse_y = y;
        self.egui.mouse_motion_event(ctx, x, y);
        if self.egui.egui_ctx().is_pointer_over_area() && self.can_grab {
            self.grab_start_mouse_x = None;
            self.grab_start_mouse_y = None;
            self.grab_start_ry = None;
            self.grab_start_rx = None;
        }
    }

    fn mouse_wheel_event(&mut self, ctx: &mut Context, dx: f32, dy: f32) {
        self.egui.mouse_wheel_event(ctx, dx, dy);
    }

    fn mouse_button_down_event(&mut self, ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        self.egui.mouse_button_down_event(ctx, button, x, y);
        if !self.egui.egui_ctx().is_pointer_over_area() {
            if let( true, MouseButton::Left ) = (self.can_grab, button) {
                self.grab_start_mouse_x = Some(self.mouse_x);
                self.grab_start_mouse_y = Some(self.mouse_y);
                self.grab_start_ry = Some(self.ry);
                self.grab_start_rx = Some(self.rx);
            }
        }
    }

    fn mouse_button_up_event(&mut self, ctx: &mut Context, button: MouseButton, x: f32, y: f32) {
        self.egui.mouse_button_up_event(ctx, button, x, y);
        if !self.egui.egui_ctx().is_pointer_over_area() {
            if let( true, MouseButton::Left ) = (self.can_grab, button) {
                self.grab_start_mouse_x = None;
                self.grab_start_mouse_y = None;
                self.grab_start_ry = None;
                self.grab_start_rx = None;
            }
        }
    }

    fn char_event(&mut self, _ctx: &mut Context, character: char, _keymods: KeyMods, repeat: bool) {
        self.egui.char_event(character);
        if self.egui.egui_ctx().wants_keyboard_input() { return; }
        if character == '~' && !repeat {
            //self.show_ui = !self.show_ui;
        }
    }

    fn key_down_event(&mut self, ctx: &mut Context, keycode: KeyCode, keymods: KeyMods, _repeat: bool,) {
        self.egui.key_down_event(ctx, keycode, keymods);
        match keycode {
            KeyCode::Escape => {
                ctx.quit()
            },
            KeyCode::Space => {
                self.can_grab = true;
            }
            _ => {}
        }
    }

    fn key_up_event(&mut self, _ctx: &mut Context, keycode: KeyCode, keymods: KeyMods) {
        self.egui.key_up_event(keycode, keymods);
        match keycode {
            KeyCode::Enter => { self.generate_map(); },
            KeyCode::Space => {
                self.can_grab = false;
                self.grab_start_mouse_x = None;
                self.grab_start_mouse_y = None;
                self.grab_start_ry = None;
                self.grab_start_rx = None;
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

    let lookat_mat = glam::Mat4::look_at_lh(
        position,
        center,
        up_vector
    );
    let proj = Mat4::perspective_lh(fov, aspect_ratio, z_near, z_far);
    proj * lookat_mat
}

fn unproj_vec(v: Vec3, mat: Mat4) -> Vec3 {
    let v: Vec4 = [v.x, v.y, v.z, 1.0].into();
    let v = mat.inverse() * v;
    [v.x / v.w, v.y / v.w, v.z / v.w].into()
}

fn proj_vec(v: Vec3, mat: Mat4) -> Vec3 {
    let v: Vec4 = [v.x, v.y, v.z, 1.0].into();
    let v = mat * v;
    [v.x / v.w, v.y / v.w, v.z / v.w].into()
}