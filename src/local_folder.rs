/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use std::{
    collections::HashMap,
    fs::File,
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, Utc};
use polars::prelude::*;
use semver::Version;
use serde::{Deserialize, Serialize};
use smugmug::v2::{Client, Node};
use std::io::BufReader;
use tokio::sync::mpsc;

use crate::{
    PathFinder,
    face_detector::{FaceDetection, find_faces},
    label_maker::LabelMaker,
    parquet_store::{get_dataframe, save_dataframe},
    smugmug_folder::{SmugMugFolder, SmugMugFolderStats},
};

/// Manages the local folder
///
///
#[derive(Debug, Serialize)]
pub struct LocalFolder {
    smugmug_folder: SmugMugFolder,
    #[serde(skip)]
    path_finder: PathFinder,
    #[serde(skip)]
    client: Option<Client>,
}

impl LocalFolder {
    /// Opens the given local directory and reads in the smugmug metadata
    pub fn get(path: &str) -> Result<Self> {
        let path_finder = PathFinder::new(path)?;
        let props_path = path_finder.get_props_file();
        let (props, version_changed) = Self::read_and_check_props(&props_path)?;

        let smugmug_folder = SmugMugFolder::populate_from_file(
            path_finder.get_album_and_node_data_file(),
            path_finder.get_album_image_map_file_if_exists(),
        )?;

        if version_changed {
            Self::write_props_file(&props_path, props)?;
        }

        Ok(Self {
            smugmug_folder,
            path_finder,
            client: None,
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

        // Read the saved props before anything is downloaded or rewritten, so a props file that
        // can't be parsed stops the sync without changing any data
        let props_path = path_finder.get_props_file();
        let (mut new_props, version_changed) = Self::read_and_check_props(&props_path)?;

        let mut smugmug_folder =
            if force_update_if_exists || !path_finder.does_album_and_node_data_file_exists()? {
                log::debug!("downloading album and node data");
                props.album_node_last_sync_ts = Some(Utc::now());
                SmugMugFolder::populate_from_node(client.to_owned(), node).await?
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
            smugmug_folder = smugmug_folder.populate_image_map(client.to_owned()).await?
        }

        // Save generic properties if they changed
        if !props.are_all_none() {
            // Save generated tree to file
            log::debug!("Saving folder metadata");
            smugmug_folder.save_meta_data(
                path_finder.get_album_and_node_data_file(),
                path_finder.get_album_image_map_file(),
            )?;

            log::debug!("Writing out props");
            if props.album_node_last_sync_ts.is_some() {
                new_props.album_node_last_sync_ts = props.album_node_last_sync_ts
            }
            if props.image_map_last_sync_ts.is_some() {
                new_props.image_map_last_sync_ts = props.image_map_last_sync_ts
            }
        }
        if !props.are_all_none() || version_changed {
            Self::write_props_file(&props_path, new_props)?;
        }

        Ok(Self {
            smugmug_folder,
            path_finder,
            client: Some(client.to_owned()),
        })
    }

    /// Syncs Image/Video Data
    ///
    ///
    pub async fn sync_artifacts(&self) -> Result<()> {
        let client = self
            .client
            .clone()
            .expect("Client wasn't found for sync artifacts operation");

        // Get and create folder to sync artifacts to if not already created
        let artifacts_folder = self.path_finder.get_artifacts_folder();
        std::fs::create_dir_all(&artifacts_folder)?;

        log::debug!(
            "Syncing artifacts to: {}",
            artifacts_folder.to_string_lossy()
        );
        self.smugmug_folder
            .sync_artifacts(
                &artifacts_folder,
                &self.path_finder.get_video_info_dir(),
                client.clone(),
            )
            .await?;

        Ok(())
    }

    /// Syncs the EXIF/IPTC metadata SmugMug holds for each Image/Video
    ///
    ///
    pub async fn sync_image_metadata(&self) -> Result<()> {
        let client = self
            .client
            .clone()
            .expect("Client wasn't found for sync image metadata operation");

        self.smugmug_folder
            .sync_image_metadata(self.path_finder.get_image_metadata_dir(), client)
            .await
    }

    /// Read props file.  Returns None if it doesn't exist yet and an error if it can't be parsed,
    /// so a bad file is never overwritten with defaults.
    ///
    fn read_props_file(path: &PathBuf) -> Result<Option<Props>> {
        match File::open(path) {
            Ok(f) => Ok(Some(serde_json::from_reader(BufReader::new(f))?)),
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    /// Read the props file (defaults if it doesn't exist yet) and check its `db_version` is one
    /// this build can work with.  Returns the props with `db_version` raised to [`DB_VERSION`]
    /// and whether that changed them, in which case the caller writes them back.
    ///
    fn read_and_check_props(path: &PathBuf) -> Result<(Props, bool)> {
        let mut props = Self::read_props_file(path)?.unwrap_or_default();
        let changed = props
            .stamp_db_version()
            .with_context(|| format!("Checking {}", path.display()))?;
        Ok((props, changed))
    }

    /// Write props file
    ///
    ///
    fn write_props_file(path: &PathBuf, props: Props) -> Result<()> {
        let props_str = serde_json::to_string_pretty(&props)?;
        Ok(std::fs::write(path, props_str)?)
    }

    /// Calculate smugmug disk space
    ///
    ///
    pub fn get_smugmug_folder_stats(&self) -> SmugMugFolderStats {
        self.smugmug_folder.get_folder_stats()
    }

    /// Calculate smugmug disk space
    ///
    ///
    pub async fn check_for_invalid_artifacts(&self) -> Result<Vec<String>> {
        let artifacts_folder = self.path_finder.get_artifacts_folder();
        self.smugmug_folder
            .artifacts_verifier(artifacts_folder, self.path_finder.get_video_info_dir())
            .await
    }

    /// Removes the upload keys based on the given parameters.  Returns number of albums
    /// upload keys were removed from
    ///
    ///
    pub async fn remove_albums_upload_keys(
        &self,
        days_since_created: u16,
        days_since_last_updated: u16,
    ) -> Result<usize> {
        let client = self.client.clone().expect("Client wasn't found");

        let num_removed = self
            .smugmug_folder
            .remove_albums_upload_keys(client, days_since_created, days_since_last_updated)
            .await?;
        Ok(num_removed)
    }

    /// Generates thumbnails and embeddings for the images in the artifacts folder
    ///
    pub async fn generate_thumbnails_and_embeddings(&self) -> Result<()> {
        let facial_embeddings_dir = self.path_finder.get_facial_embeddings_dir();
        let artifacts_dir = self.path_finder.get_artifacts_folder();

        let (tx, mut rx) = mpsc::unbounded_channel();

        // worker that generates the embedding
        let face_detector_worker =
            move |image_path: PathBuf, thumbnail_dir: PathBuf, tx: mpsc::UnboundedSender<_>| {
                log::trace!("Processing image: {}", image_path.display());
                let detections = match find_faces(
                    image_path.to_str().unwrap(),
                    Some(thumbnail_dir.to_str().unwrap()),
                ) {
                    Ok(detections) => detections,
                    Err(e) => {
                        log::error!(
                            "Error processing image {}: {}",
                            image_path.file_stem().unwrap().to_str().unwrap(),
                            e
                        );
                        return;
                    }
                };

                log::debug!(
                    "Finished processing image: {} found {} faces",
                    image_path.file_stem().unwrap().to_str().unwrap(),
                    detections.iter().filter(|v| v.face_id.is_some()).count()
                );

                if let Err(e) = tx.send(detections) {
                    log::error!(
                        "Failed sending detection results for image {}: {}",
                        image_path.file_stem().unwrap().to_str().unwrap(),
                        e
                    );
                }
            };

        // Read in parquet lazy data frames from the facial embeddings
        // directory and find all the existing facial embeddings image_ids
        // to not process them again
        let processed_image_ids: Vec<String> =
            get_dataframe(&facial_embeddings_dir, [col("image_id")])?
                .unique(None, UniqueKeepStrategy::Any)
                .collect()?
                .column("image_id")?
                .str()?
                .iter()
                .filter_map(|v| v.map(|s| s.to_string()))
                .collect();
        log::debug!(
            "Found {} processed images in the facial embeddings directory",
            processed_image_ids.len()
        );

        // create workers to process the images if they have not already been processed
        self.smugmug_folder
            .get_unique_artifacts_metadata()
            .into_iter()
            .filter(|v| {
                !v.is_video
                    && !processed_image_ids.contains(&v.image_key)
                    && artifacts_dir.join(&v.image_key).exists()
            })
            .for_each(|v| {
                let facial_thumbnail_dir = self.path_finder.get_facial_thumbnails_dir();
                let image_path = artifacts_dir.join(&v.image_key);
                let tx = tx.clone();
                tokio::spawn(async move {
                    face_detector_worker(image_path, facial_thumbnail_dir, tx);
                });
            });

        drop(tx); // Close the channel so it knowns to stop getting messages

        // save the dataframe to parquet
        fn checkpoint_save_builder(
            builder: &mut DetectionDataFrameBuilder,
            path: &Path,
        ) -> Result<()> {
            let mut df = builder.build()?;
            save_dataframe(&mut df, path, "embeddings")?;
            Ok(())
        }

        // wait for the workers to send facial detections and finish
        let mut face_detector_df_builder = DetectionDataFrameBuilder::new();
        while let Some(detections) = rx.recv().await {
            // Add the detections to the DataFrame builder
            for detection in detections {
                face_detector_df_builder.add_detection(detection);
            }

            // Check if the detections has reached a certain size
            if face_detector_df_builder.len() > 1000 {
                checkpoint_save_builder(&mut face_detector_df_builder, &facial_embeddings_dir)?;

                // Reset the builder
                face_detector_df_builder = DetectionDataFrameBuilder::new();
            }
        }

        // Save the remaining DataFrame
        log::debug!("Finished processing images and saving remaining detections");
        if face_detector_df_builder.len() > 0 {
            checkpoint_save_builder(&mut face_detector_df_builder, &facial_embeddings_dir)?;
            // For safety
            drop(face_detector_df_builder);
        }

        Ok(())
    }

    /// Generates labels based on the images found in the pre-sorted thumbnails directory and comparing
    /// the embeddings for those to the unsorted images in the facial thumbnails directory
    ///
    /// This will:
    /// - Read in the thumbnails in the pre-sorted directory
    /// - Load in the embeddings associated with the images
    /// - Compare them to the unsorted images in the facial thumbnails directory
    /// - Generate the labels (using the directory names in the pre-sorted directory) and the compaison
    ///
    pub async fn generate_labels_from_presorted_dir(
        &self,
        presorted_thumbnail_dir: &str,
    ) -> Result<()> {
        let facial_embeddings_dir = self.path_finder.get_facial_embeddings_dir();
        let detection_df = get_dataframe(
            &facial_embeddings_dir,
            [col("face_id"), col("image_id"), col("embeddings")],
        )?;
        let label_maker = LabelMaker::new(&self.path_finder, presorted_thumbnail_dir, detection_df);
        label_maker.sort_images_using_presorted_dir().await?;
        log::debug!("Generating labels files");
        label_maker.create_labels_file().await
    }

    /// Updates the labels based on the facial tags json file
    pub async fn update_smugmug_images_with_facial_tags(&self) -> Result<bool> {
        let labels_file = self.path_finder.get_facial_tags_file();
        let file = File::open(&labels_file)?;
        let labels_to_imgs_map: HashMap<String, Vec<String>> = serde_json::from_reader(file)?;
        let client = self.client.clone().expect("Client wasn't found");
        self.smugmug_folder
            .update_labels_from_json_tag(client, labels_to_imgs_map)
            .await
    }
}

/// Version of the `.smugmug_db` layout this build reads and writes, stored as `db_version` in
/// `props.json` so readers such as smugmug_syncer_viewer can tell whether they understand it.
///
/// It follows semver and is separate from the crate version:
/// - major: a breaking change (a file, directory or column renamed, removed or retyped), which
///   readers of the old layout can't handle
/// - minor: an additive change (a new file, directory or optional column), which readers of the
///   old layout can ignore
/// - patch: a fix that doesn't change the layout
///
/// Archives written before the version was recorded have the 1.0.0 layout.
pub(crate) const DB_VERSION: &str = "1.0.0";

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
#[serde(default)]
struct Props {
    album_node_last_sync_ts: Option<DateTime<Utc>>,
    image_map_last_sync_ts: Option<DateTime<Utc>>,
    /// Layout version of the `.smugmug_db` directory, see [`DB_VERSION`]
    db_version: Option<String>,

    // Keys this version doesn't know about, kept so rewriting the file doesn't drop them
    #[serde(flatten)]
    other: serde_json::Map<String, serde_json::Value>,
}

impl Props {
    fn are_all_none(&self) -> bool {
        self.album_node_last_sync_ts.is_none() && self.image_map_last_sync_ts.is_none()
    }

    /// Checks that this build can work with the stored `db_version` and raises it to
    /// [`DB_VERSION`] if it is older or missing.  Returns whether it changed, i.e. whether the
    /// props need writing.  A stored version with a different major version is an error, so a
    /// layout this build doesn't understand is never written to.  A newer minor version is
    /// kept rather than lowered, since that layout is still compatible.
    fn stamp_db_version(&mut self) -> Result<bool> {
        let ours = Version::parse(DB_VERSION).expect("DB_VERSION is valid semver");
        let Some(stored_str) = self.db_version.as_deref() else {
            self.db_version = Some(DB_VERSION.to_string());
            return Ok(true);
        };
        let stored = Version::parse(stored_str)
            .with_context(|| format!("props.json has an invalid db_version '{stored_str}'"))?;
        if stored.major > ours.major {
            bail!(
                "The archive's database is version {stored}, written by a newer smugmug_syncer. \
                 This build supports {}.x; please upgrade smugmug_syncer.",
                ours.major
            );
        }
        if stored.major < ours.major {
            bail!(
                "The archive's database is version {stored}, which this build ({ours}) can't \
                 migrate."
            );
        }
        if stored < ours {
            self.db_version = Some(DB_VERSION.to_string());
            return Ok(true);
        }
        Ok(false)
    }
}

impl std::fmt::Display for LocalFolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.smugmug_folder)
    }
}

/// Builder for the Face Detection DataFrame
///
#[derive(Default)]
struct DetectionDataFrameBuilder {
    image_ids: Vec<String>,
    face_ids: Vec<Option<String>>,
    embeddings: Vec<Series>,
    rect_x: Vec<Option<i32>>,
    rect_y: Vec<Option<i32>>,
    rect_w: Vec<Option<i32>>,
    rect_h: Vec<Option<i32>>,
}
impl DetectionDataFrameBuilder {
    fn new() -> Self {
        Self::default()
    }
    fn add_detection(&mut self, detection: FaceDetection) {
        self.image_ids.push(detection.image_id);
        self.face_ids.push(detection.face_id);
        let series_embedding: Series = detection
            .embeddings
            .unwrap_or_default()
            .into_iter()
            .collect();
        self.embeddings.push(series_embedding);
        if let Some(rect) = detection.rect_in_image {
            self.rect_x.push(Some(rect.x));
            self.rect_y.push(Some(rect.y));
            self.rect_w.push(Some(rect.w));
            self.rect_h.push(Some(rect.h));
        } else {
            self.rect_x.push(None);
            self.rect_y.push(None);
            self.rect_w.push(None);
            self.rect_h.push(None);
        }
    }
    fn len(&self) -> usize {
        self.image_ids.len()
    }

    fn build(&mut self) -> Result<polars::prelude::DataFrame> {
        let df = df![
            "image_id" => self.image_ids.to_owned(),
            "face_id" => self.face_ids.to_owned(),
            "x" => self.rect_x.to_owned(),
            "y" => self.rect_y.to_owned(),
            "w" => self.rect_w.to_owned(),
            "h" => self.rect_h.to_owned(),
            "embeddings" => self.embeddings.to_owned(),
        ]?;
        Ok(df)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props_with(version: Option<&str>) -> Props {
        Props {
            db_version: version.map(str::to_string),
            ..Default::default()
        }
    }

    #[test]
    fn missing_db_version_is_stamped() {
        let mut props = props_with(None);
        assert!(props.stamp_db_version().unwrap());
        assert_eq!(props.db_version.as_deref(), Some(DB_VERSION));
    }

    #[test]
    fn current_db_version_is_unchanged() {
        let mut props = props_with(Some(DB_VERSION));
        assert!(!props.stamp_db_version().unwrap());
    }

    #[test]
    fn newer_minor_db_version_is_kept() {
        let ours = Version::parse(DB_VERSION).unwrap();
        let newer = format!("{}.{}.0", ours.major, ours.minor + 1);
        let mut props = props_with(Some(&newer));
        assert!(!props.stamp_db_version().unwrap());
        assert_eq!(props.db_version.as_deref(), Some(newer.as_str()));
    }

    #[test]
    fn other_major_db_version_is_rejected() {
        let ours = Version::parse(DB_VERSION).unwrap();
        let newer = format!("{}.0.0", ours.major + 1);
        assert!(props_with(Some(&newer)).stamp_db_version().is_err());
        assert!(
            props_with(Some("not a version"))
                .stamp_db_version()
                .is_err()
        );
    }

    #[test]
    fn props_keep_unknown_keys() {
        let json = r#"{"album_node_last_sync_ts":null,"future_key":42}"#;
        let mut props: Props = serde_json::from_str(json).unwrap();
        props.stamp_db_version().unwrap();
        let out = serde_json::to_value(&props).unwrap();
        assert_eq!(out["future_key"], 42);
        assert_eq!(out["db_version"], DB_VERSION);
    }
}
