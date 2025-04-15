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

use anyhow::Result;
use chrono::{DateTime, Utc};
use polars::prelude::*;
use serde::{Deserialize, Serialize};
use smugmug::v2::{Client, Node};
use std::io::BufReader;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::{
    PathFinder,
    face_detector::{FaceDetection, find_faces},
    label_maker::LabelMaker,
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

        let smugmug_folder = SmugMugFolder::populate_from_file(
            path_finder.get_album_and_node_data_file(),
            path_finder.get_album_image_map_file_if_exists(),
        )?;

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
            .sync_artifacts(&artifacts_folder, client.clone())
            .await?;

        Ok(())
    }

    /// Read props file
    ///
    ///
    fn read_props_file(path: &PathBuf) -> Option<Props> {
        File::open(path).ok().and_then(|f| {
            let rdr = BufReader::new(f);
            serde_json::from_reader(rdr).ok()
        })
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
            .artifacts_verifier(artifacts_folder)
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
                    &detections.iter().filter(|v| v.face_id.is_some()).count()
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
                .into_iter()
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

            let file_id = Uuid::new_v4().simple().to_string();
            let file_path = path.join(format!("embeddings_{}.parquet", file_id));

            log::debug!(
                "Doing checkpoint save for: {} at: {}",
                &df,
                file_path.display()
            );

            let file = File::create(file_path)?;
            ParquetWriter::new(file)
                .with_compression(ParquetCompression::Zstd(None))
                .finish(&mut df)?;

            Ok(())
        }

        // Create facial embeddings directory if it doesn't exist
        match std::fs::create_dir_all(&facial_embeddings_dir) {
            Ok(_) => {}
            Err(ref error) if error.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(error) => {
                return Err(error.into()); // Return the error
            }
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

// Reads the parquet files in the given directory and returns a DataFrame
pub(crate) fn get_dataframe<E: AsRef<[Expr]>>(root_path: &Path, exprs: E) -> Result<LazyFrame> {
    let parquet_glob = root_path.join("*.parquet");
    let lf = LazyFrame::scan_parquet(parquet_glob, Default::default())?;
    Ok(lf.select(exprs))
}
