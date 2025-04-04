/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use std::{fs::File, path::PathBuf};

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use smugmug::v2::{Client, Node};
use std::io::BufReader;

use crate::{
    PathFinder,
    smugmug_folder::{SmugMugFolder, SmugMugFolderStats},
};

/// Manages the local folder
pub struct LocalFolder {
    smugmug_folder: SmugMugFolder,
    path_finder: PathFinder,
}

impl LocalFolder {
    /// Opens the given local directory and reads in the smugmug metadata
    pub fn get(path: &str) -> Result<Self> {
        let path_finder = PathFinder::new(path)?;

        let smugmug_folder = SmugMugFolder::populate_from_file(
            path_finder.get_album_and_node_data_file(),
            path_finder.get_album_image_map_file_if_exists(),
        )?;

        Ok(Self {
            smugmug_folder,
            path_finder,
        })
    }

    /// Opens the given local directory and reads in the smugmug metadata if available
    /// otherwise it will generating it.
    ///
    /// TODO: Some basic sanity check on if the node matches the data
    pub async fn get_or_create(
        path: &str,
        client: Client,
        node: Node,
        do_download_artifact_info: bool,
        force_update_if_exists: bool,
    ) -> Result<Self> {
        let path_finder = PathFinder::new(path)?;
        let mut props = Props {
            ..Default::default()
        };

        let mut smugmug_folder =
            if force_update_if_exists || !path_finder.does_album_and_node_data_file_exists()? {
                log::debug!("downloading album and node data");
                props.album_node_last_sync_ts = Some(Utc::now());
                SmugMugFolder::populate_from_node(client.clone(), node).await?
            } else {
                // Retrieve all from file cache
                let album_map_file_opt = if path_finder.does_album_image_map_file_exists()? {
                    Some(path_finder.get_album_image_map_file())
                } else {
                    None
                };
                log::debug!("loading folder data from file");
                SmugMugFolder::populate_from_file(
                    path_finder.get_album_and_node_data_file(),
                    album_map_file_opt,
                )?
            };

        // Download the Image/Video meta data
        if do_download_artifact_info
            && (force_update_if_exists || !smugmug_folder.are_images_populated())
        {
            log::debug!("downloading image data");
            props.image_map_last_sync_ts = Some(Utc::now());
            smugmug_folder = smugmug_folder.populate_image_map(client).await?
        }

        // Save generated tree to file
        log::debug!("Saving folder metadata");
        smugmug_folder.save_meta_data(
            path_finder.get_album_and_node_data_file(),
            path_finder.get_album_image_map_file(),
        )?;

        // Save generic properties if they changed
        if !props.are_all_none() {
            log::debug!("Writing out props");
            let props_path = path_finder.get_props_file();
            let mut new_props = Self::read_props_file(&props_path).unwrap_or_default();
            if props.album_node_last_sync_ts.is_some() {
                new_props.album_node_last_sync_ts = props.album_node_last_sync_ts
            }
            if props.image_map_last_sync_ts.is_some() {
                new_props.image_map_last_sync_ts = props.image_map_last_sync_ts
            }
            Self::write_props_file(&props_path, new_props)?;
        }

        Ok(Self {
            smugmug_folder,
            path_finder,
        })
    }

    // Read props file
    fn read_props_file(path: &PathBuf) -> Option<Props> {
        File::open(path).ok().and_then(|f| {
            let rdr = BufReader::new(f);
            serde_json::from_reader(rdr).ok()
        })
    }

    // Write props file
    fn write_props_file(path: &PathBuf, props: Props) -> Result<()> {
        let props_str = serde_json::to_string_pretty(&props)?;
        Ok(std::fs::write(path, props_str)?)
    }

    /// Calculate smugmug disk space
    pub fn get_smugmug_folder_stats(&self) -> SmugMugFolderStats {
        self.smugmug_folder.get_folder_stats()
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
struct Props {
    album_node_last_sync_ts: Option<DateTime<Utc>>,
    image_map_last_sync_ts: Option<DateTime<Utc>>,
}

impl Props {
    fn are_all_none(&self) -> bool {
        self.album_node_last_sync_ts.is_none() && self.image_map_last_sync_ts.is_none()
    }
}

impl std::fmt::Display for LocalFolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.smugmug_folder)
    }
}
