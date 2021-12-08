pub mod geometry;
pub mod demo_stage;
pub mod tile_config;
pub mod utility;
pub mod voxel_mesh;
pub mod selection_point_grid;

fn main() {
    orom_miniquad::start(
        orom_miniquad::conf::Conf {
            sample_count: 8,
            ..Default::default()
        },
        |mut ctx| {
        orom_miniquad::UserData::owning(
            demo_stage::CollapseDemoStage::new(&mut ctx, [20, 5, 20]),
            ctx
        )
    });
}
