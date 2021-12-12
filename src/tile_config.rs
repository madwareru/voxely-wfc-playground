use std::collections::HashMap;

#[derive(PartialEq, Eq, Hash, Copy, Clone, serde::Deserialize, Debug)]
pub enum CellType {
    Ground,
    Water,
    Air
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, serde::Deserialize, Debug)]
pub struct TileConfigEntryColumn {
    pub bottom: CellType,
    pub up: CellType
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, serde::Deserialize, Debug)]
pub struct TileConfigEntry {
    pub north_east: TileConfigEntryColumn,
    pub north_west: TileConfigEntryColumn,
    pub south_east: TileConfigEntryColumn,
    pub south_west: TileConfigEntryColumn
}

impl TileConfigEntry {
    pub fn all_of_type(cell_type: CellType) -> Self {
        Self {
            north_east: TileConfigEntryColumn { bottom: cell_type, up: cell_type },
            north_west: TileConfigEntryColumn { bottom: cell_type, up: cell_type },
            south_east: TileConfigEntryColumn { bottom: cell_type, up: cell_type },
            south_west: TileConfigEntryColumn { bottom: cell_type, up: cell_type }
        }
    }

    pub fn rotate(&self) -> Self {
        Self {
            north_east: self.south_east,
            south_east: self.south_west,
            south_west: self.north_west,
            north_west: self.north_east
        }
    }

    pub fn is_east_neighbour(&self, other: &Self) -> bool {
        self.north_east.eq(&other.north_west) && self.south_east.eq(&other.south_west)
    }

    pub fn is_west_neighbour(&self, other: &Self) -> bool {
        self.north_west.eq(&other.north_east) && self.south_west.eq(&other.south_east)
    }

    pub fn is_north_neighbour(&self, other: &Self) -> bool {
        self.north_west.eq(&other.south_west) && self.north_east.eq(&other.south_east)
    }

    pub fn is_south_neighbour(&self, other: &Self) -> bool {
        self.south_west.eq(&other.north_west) && self.south_east.eq(&other.north_east)
    }

    pub fn is_bottom_neighbour(&self, other: &Self) -> bool {
        self.south_east.bottom.eq(&other.south_east.up) &&
            self.south_west.bottom.eq(&other.south_west.up) &&
            self.north_east.bottom.eq(&other.north_east.up) &&
            self.north_west.bottom.eq(&other.north_west.up)
    }

    pub fn is_up_neighbour(&self, other: &Self) -> bool {
        self.south_east.up.eq(&other.south_east.bottom) &&
            self.south_west.up.eq(&other.south_west.bottom) &&
            self.north_east.up.eq(&other.north_east.bottom) &&
            self.north_west.up.eq(&other.north_west.bottom)
    }
}

#[derive(Clone, serde::Deserialize)]
pub struct TileConfig {
    pub stride: usize,
    pub tile_step: f32,
    pub tile_width: f32,
    pub mappings: HashMap<usize, TileConfigEntry>
}