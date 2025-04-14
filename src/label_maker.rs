/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::BufWriter,
    path::{Path, PathBuf},
};

use anyhow::Result;
use futures::future::TryJoinAll;
use polars::prelude::*;
use tokio::task::JoinHandle;

use crate::{PathFinder, face_detector::FaceDetection};

pub struct LabelMaker {
    path_finder: PathFinder,
    presorted_thumbnails_dir: PathBuf,
    detection_df: LazyFrame,
}

impl LabelMaker {
    pub fn new(path_finder: &PathFinder, presorted_thumbnails_dir: &str, df: LazyFrame) -> Self {
        Self {
            path_finder: path_finder.clone(),
            presorted_thumbnails_dir: PathBuf::from(presorted_thumbnails_dir),
            detection_df: df,
        }
    }

    /// Generates labels based on the images found in the pre-sorted thumbnails directory and comparing
    /// the embeddings for those to the unsorted images in the facial thumbnails directory
    ///
    /// This will:
    /// - Read in the thumbnails in the pre-sorted directory
    /// - Load in the embeddings associated with the images
    /// - Compare them to the unsorted images in the facial thumbnails directory
    /// - Generate the labels (using the directory names in the pre-sorted directory) and the comparison
    ///
    pub async fn sort_images_using_presorted_dir(&self) -> Result<()> {
        let label_dirs = get_label_dirs(&self.presorted_thumbnails_dir)?;
        log::debug!("Labels and their dirs: {:?}", label_dirs);

        let presorted_embeddings =
            find_embeddings_in_label_dirs(label_dirs, self.detection_df.clone()).await?;
        log::debug!(
            "Pre-sorted embeddings size: {}",
            &presorted_embeddings.len()
        );
        // Compare the embeddings to the unsorted images in the facial thumbnails directory
        let unsorted_embeddings = embeddings_collector_for_dir(
            self.path_finder.get_facial_thumbnails_dir(),
            self.detection_df.clone(),
        )?;

        // For each of the unsorted images, find the label it belongs to
        let mut sorted_faces: HashMap<&str, Vec<String>> = HashMap::new();
        unsorted_embeddings.into_iter().for_each(|unknown| {
            for (label, known_detections) in &presorted_embeddings {
                let num_matches = known_detections
                    .iter()
                    .filter(|known| known.is_same_face(&unknown))
                    .count();
                if num_matches as f32 / known_detections.len() as f32 > 0.9 {
                    sorted_faces
                        .entry(label.as_str())
                        .or_default()
                        .push(unknown.face_id.as_ref().unwrap().clone());
                }
            }
        });

        log::debug!("Sorted images size: {}", &sorted_faces.len());
        let sorted_thumbnail_dir = self.path_finder.get_sorted_facial_thumbnails_dir();
        let unsorted_thumbnail_dir = self.path_finder.get_facial_thumbnails_dir();
        sort_thumbnails_into_dirs(sorted_faces, sorted_thumbnail_dir, unsorted_thumbnail_dir)?;
        Ok(())
    }

    /// Generates a labels file based on the images found in the pre-sorted thumbnails directory
    /// and the sorted images in the facial thumbnails directory
    pub async fn create_labels_file(&self) -> Result<()> {
        // get the labels from the presorted thumbnails directory

        let get_label_to_image_ids_map =
            async |dir: &Path| -> Result<HashMap<String, HashSet<String>>> {
                let label_dirs = get_label_dirs(dir)?;
                let images: HashMap<String, HashSet<String>> =
                    find_embeddings_in_label_dirs(label_dirs, self.detection_df.clone())
                        .await?
                        .into_iter()
                        .map(|(label, detections)| {
                            let image_ids: HashSet<String> =
                                detections.into_iter().map(|v| v.image_id).collect();
                            (label, image_ids)
                        })
                        .collect();
                Ok(images)
            };

        // combine the sorted and presorted labels -> images to generate the label file.
        let label_to_image_map = {
            // get the labels from the presorted thumbnails directory
            let mut label_to_image_map =
                get_label_to_image_ids_map(&self.presorted_thumbnails_dir).await?;
            let sorted_label_to_image_map =
                get_label_to_image_ids_map(&self.path_finder.get_sorted_facial_thumbnails_dir())
                    .await?;
            for (label, image_ids) in sorted_label_to_image_map {
                label_to_image_map
                    .entry(label)
                    .or_default()
                    .extend(image_ids);
            }
            label_to_image_map
        };

        let labels_file = self.path_finder.get_facial_tags_file();
        let file = File::create(&labels_file)?;
        let writer = BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &label_to_image_map)?;
        Ok(())
    }
}

// Moves the unsorted facial thumbnails into directories based on the labels
fn sort_thumbnails_into_dirs(
    sorted_faces: HashMap<&str, Vec<String>>,
    sorted_thumbnail_dir: PathBuf,
    unsorted_thumbnail_dir: PathBuf,
) -> Result<()> {
    std::fs::create_dir_all(&sorted_thumbnail_dir)?;

    // For each of the labels, create a directory and move the images into it
    for (label, face_ids) in sorted_faces {
        let label_dir = sorted_thumbnail_dir.join(label);
        std::fs::create_dir_all(&label_dir)?;
        face_ids.iter().for_each(|face_id| {
            let thumbnail_img = format!("{}.jpg", face_id);
            let src_image_path = unsorted_thumbnail_dir.join(&thumbnail_img);
            let new_image_path = label_dir.join(&thumbnail_img);
            if !new_image_path.exists() && src_image_path.exists() {
                log::debug!(
                    "Moving image {} to {}",
                    src_image_path.display(),
                    new_image_path.display()
                );
                std::fs::rename(src_image_path, new_image_path).unwrap();
            }
        });
    }
    Ok(())
}

// Retrieves the list of directories in the pre-sorted thumbnails directory
// whose root path will be used for labels
fn get_label_dirs(presorted_thumbnail_dir: &Path) -> Result<HashMap<String, PathBuf>> {
    let label_dirs: HashMap<String, PathBuf> = std::fs::read_dir(presorted_thumbnail_dir)?
        .filter_map(|entry| {
            entry.ok().and_then(|entry| {
                let path = entry.path();
                if path.is_dir() {
                    let label = path.file_name().unwrap().to_str().unwrap().to_string();
                    Some((label, path))
                } else {
                    None
                }
            })
        })
        .collect();
    Ok(label_dirs)
}

// Retrieves the embeddings from presorted thumbnails directory
// and returns a map of label to the embeddings
async fn find_embeddings_in_label_dirs(
    label_dirs: HashMap<String, PathBuf>,
    detection_df: LazyFrame,
) -> Result<HashMap<String, Vec<FaceDetection>>> {
    // Collect embeddings for the found facial thumbnails in these folders
    let presorted_embeddings: HashMap<String, Vec<FaceDetection>> = label_dirs
        .into_iter()
        .map(|(label, label_dir)| {
            let df = detection_df.clone();
            tokio::spawn(async move {
                log::debug!("Processing label {} dir: {}", &label, label_dir.display());
                let embeddings = embeddings_collector_for_dir(label_dir.clone(), df)?;
                log::trace!(
                    "Finished processing label {} dir: {}",
                    &label,
                    label_dir.display()
                );

                Ok::<(String, Vec<FaceDetection>), anyhow::Error>((label, embeddings))
            })
        })
        .collect::<TryJoinAll<JoinHandle<Result<_, _>>>>()
        .await?
        .into_iter()
        .filter_map(|v: Result<(String, Vec<FaceDetection>), _>| {
            if let Ok((label, detections)) = v {
                Some((label, detections))
            } else {
                None
            }
        })
        .collect();
    Ok(presorted_embeddings)
}

// Extracts the detection from the DataFrame for a given face_id
fn extract_detections_for_face_ids(
    face_ids: HashSet<String>,
    df: LazyFrame,
) -> Result<Vec<FaceDetection>> {
    log::trace!("Extracting detections for: {}", &face_ids.len());
    let face_id_filter_values: Vec<&str> = face_ids.iter().map(String::as_str).collect();
    let filter_series = Series::new("face_id_filter".into(), face_id_filter_values);

    let filtered = df
        .filter(col("face_id").is_in(lit(filter_series)))
        .collect()?;

    let face_ids = filtered.column("face_id")?.str()?;
    let image_ids = filtered.column("image_id")?.str()?;
    let embeddings_col = filtered.column("embeddings")?.list()?;

    let mut results = Vec::new();

    for i in 0..filtered.height() {
        let face_id = face_ids.get(i).map(|s| s.to_string());
        let image_id = image_ids.get(i).unwrap_or("").to_string();

        let embedding_series_opt = embeddings_col.get_as_series(i);

        let embedding = embedding_series_opt.and_then(|series| {
            series
                .f64()
                .ok()
                .map(|chunked| chunked.into_no_null_iter().collect::<Vec<f64>>())
        });

        results.push(FaceDetection {
            embeddings: embedding,
            image_id,
            face_id,
            ..Default::default()
        });
    }
    Ok(results)
}

// Reads the given directory and returns the facial embeddings based on the images it finds
fn embeddings_collector_for_dir(
    thumbnail_dir: PathBuf,
    df: LazyFrame,
) -> Result<Vec<FaceDetection>> {
    // scan the given dir for the facial thumbnails and extract their image id
    let image_ids_to_get_embedding: HashSet<String> = std::fs::read_dir(&thumbnail_dir)?
        .filter_map(|entry| {
            entry.ok().and_then(|entry| {
                let path = entry.path();
                if path.is_file() {
                    Some(path.file_stem().unwrap().to_str().unwrap().to_string())
                } else {
                    None
                }
            })
        })
        .collect();

    // extract the FaceDetection for the found image ids
    let found_detections: Vec<FaceDetection> =
        extract_detections_for_face_ids(image_ids_to_get_embedding, df.clone())?;

    log::debug!(
        "For {} found these detections {}",
        thumbnail_dir.display(),
        &found_detections.len()
    );

    Ok(found_detections)
}
