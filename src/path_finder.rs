/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use anyhow::Result;
use std::path::PathBuf;

// Builds the paths needed for storing data.
pub(crate) struct PathFinder {
    base_path: PathBuf,
}
impl PathFinder {
    pub(crate) fn new(base_path: &str) -> Result<Self> {
        let finder = Self {
            base_path: PathBuf::from(base_path),
        };

        // Create Db dir if it doesn't exists
        std::fs::create_dir_all(finder.get_meta_db_path())?;
        Ok(finder)
    }

    fn get_meta_db_path(&self) -> PathBuf {
        let mut path = self.base_path.clone();
        path.push(".smugmug_db");
        path
    }

    pub(crate) fn does_album_and_node_data_file_exists(&self) -> Result<bool> {
        Ok(std::fs::exists(self.get_album_and_node_data_file())?)
    }

    pub(crate) fn get_album_and_node_data_file(&self) -> PathBuf {
        let mut path = self.get_meta_db_path();
        path.push("node_and_album_tree.db");
        path
    }

    pub(crate) fn does_album_image_map_file_exists(&self) -> Result<bool> {
        Ok(std::fs::exists(self.get_album_image_map_file())?)
    }

    #[allow(dead_code)]
    pub(crate) fn get_album_image_map_file(&self) -> PathBuf {
        let mut path = self.get_meta_db_path();
        path.push("album_image_map.db");
        path
    }
}
