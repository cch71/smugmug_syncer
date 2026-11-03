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
    io::BufReader,
    path::{Path, PathBuf},
    sync::Arc,
};

use anyhow::{Result, anyhow};
use chrono::Utc;
use futures::{
    StreamExt,
    future::{TryJoinAll, try_join_all},
    pin_mut,
};
use image::ImageReader;
use serde::Serialize;
use serde_json::json;
use smugmug::v2::{
    Album, Client, Image, ImageMetadata, Node, NodeType, NodeTypeFilters, SortDirection, SortMethod,
};
use tokio::sync::mpsc;

use tree_ds::prelude as TreeDs;

use crate::{
    file_io::{bytes_md5, file_md5, remove_partial_files, write_file_atomically},
    parquet_store::save_in_batches,
    retry::with_retries,
    smugmug_image_metadata::ImageMetadataRecord,
    smugmug_video::VideoInfo,
};

type DsTree = TreeDs::Tree<Arc<Node>, Arc<Album>>;
type DsNode = TreeDs::Node<Arc<Node>, Arc<Album>>;

/// Represents a folder in SmugMug
#[derive(Debug, Serialize)]
pub struct SmugMugFolder {
    tree: DsTree,
    album_image_map: HashMap<String, Vec<Arc<Image>>>,
    #[serde(skip)]
    client_req_limiter: Arc<tokio::sync::Semaphore>,
}

impl SmugMugFolder {
    // gets num concurrent requests allowed
    fn get_num_concurrent_requests() -> usize {
        std::env::var("SMUGMUG_SYNC_WORKERS")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(1) as usize
    }

    /// Populates the tree starting from the given node
    pub async fn populate_from_node(client: Client, node: Node) -> Result<Self> {
        // Wrap the node in an Arc to make it cheaper to access in ds_tree
        let node = Arc::new(node);

        // Create tree using given node name
        let mut tree = DsTree::new(Some(&node.name));

        // Create a ds_tree Node.  Internally they are using an Arc as well
        let ds_node = DsNode::new(node, None);

        // Add given node to tree and parent.
        tree.add_node(ds_node.clone(), None)?;

        // Build tree of nodes first.
        Self::build_node_tree(&mut tree, ds_node.clone()).await?;

        log::debug!("Discovered {} Nodes", tree.get_nodes().len());

        // Get the Albums next.  Doing it this way allows the use of the multi-get call for Albums which
        // we can't do with nodes since we have to discover them.
        Self::populate_albums(client, &tree).await?;

        Ok(Self {
            tree,
            album_image_map: HashMap::new(),
            client_req_limiter: Arc::new(tokio::sync::Semaphore::new(
                Self::get_num_concurrent_requests(),
            )),
        })
    }

    /// Go through the Albums and get the images information
    pub async fn populate_image_map(&self, client: Client) -> Result<Self> {
        // Collects the images for an album and returns a tuple of album key and image collection
        let image_child_collector = async |album: Arc<Album>| -> Result<(String, Vec<Arc<Image>>)> {
            log::debug!("Collecting image metadata for album: {}", &album);
            let req_limiter = self.client_req_limiter.acquire().await.unwrap();
            let images = album.images_with_client(client.clone())?;
            drop(req_limiter);

            let mut image_info_list = Vec::new();
            pin_mut!(images);
            while let Some(image_result) = images.next().await {
                let image_info = image_result?;
                image_info_list.push(Arc::new(image_info));
            }
            log::debug!(
                "Collected metadata for {} images for album: {}",
                image_info_list.len(),
                &album
            );
            if log::log_enabled!(log::Level::Debug) {
                let num_req_remaining = client
                    .get_last_rate_limit_window_update()
                    .and_then(|v| v.num_remaining_requests())
                    .map_or("??????".to_string(), |v| v.to_string());
                log::debug!("Requests remaining: {}", num_req_remaining);
            }
            Ok((album.album_key.clone(), image_info_list))
        };

        let albums: Vec<Arc<Album>> = self
            .tree
            .get_nodes()
            .iter()
            .filter_map(|ds_node| ds_node.get_value().ok().flatten())
            .collect();

        // Parallel way of doing it but harder on SmugMug API and debugging so disabling for now
        let album_image_map: HashMap<String, Vec<Arc<Image>>> = albums
            .into_iter()
            .map(image_child_collector)
            .collect::<TryJoinAll<_>>()
            .await?
            .into_iter()
            .collect();

        log::debug!("Finished collecting image metadata");

        Ok(Self {
            tree: self.tree.clone(),
            album_image_map,
            client_req_limiter: Arc::new(tokio::sync::Semaphore::new(
                Self::get_num_concurrent_requests(),
            )),
        })
    }

    /// Populates the Folder meta data from the provide file
    pub fn populate_from_file<P: AsRef<Path>>(
        album_node_file: P,
        image_map_file_opt: Option<P>,
    ) -> Result<Self> {
        fn get_data<P: AsRef<Path>>(path: &P) -> Result<Vec<u8>> {
            log::trace!("Opening file: {}", path.as_ref().display());
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            Ok(zstd::decode_all(reader)?)
        }

        let tree: DsTree = serde_json::from_slice(&get_data(&album_node_file)?)?;

        let album_image_map: HashMap<String, Vec<Arc<Image>>> = match image_map_file_opt {
            Some(image_map) => serde_json::from_slice(&get_data(&image_map)?)?,
            None => HashMap::new(),
        };

        Ok(Self {
            tree,
            album_image_map,
            client_req_limiter: Arc::new(tokio::sync::Semaphore::new(
                Self::get_num_concurrent_requests(),
            )),
        })
    }

    // Goes through the nodes and builds up the node tree
    async fn populate_albums(client: Client, tree: &DsTree) -> Result<()> {
        let mut album_to_node_map: HashMap<String, DsNode> = tree
            .get_nodes()
            .iter()
            .map(|ds_node| Ok((ds_node.get_node_id()?, ds_node)))
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .filter(|(node, _)| matches!(node.node_type, NodeType::Album))
            .map(|(node, ds_node)| {
                let album_id = node.album_id().unwrap();
                (album_id, ds_node.clone())
            })
            .collect();

        // Retrieve the Albums
        if !album_to_node_map.is_empty() {
            log::debug!("Getting {} Albums", album_to_node_map.len());
            let albums = Album::from_id_slice(
                client,
                album_to_node_map
                    .keys()
                    .map(|v| v.as_str())
                    .collect::<Vec<&str>>()
                    .as_slice(),
            )
            .await?;

            // Associate the albums with the nodes
            for album in albums {
                let ds_node = album_to_node_map.remove(&album.album_key).unwrap();
                ds_node.set_value(Some(Arc::new(album)))?;
            }
        }

        Ok(())
    }

    // Goes through the nodes and builds up the node tree
    async fn build_node_tree(tree: &mut DsTree, ds_parent_node: DsNode) -> Result<()> {
        // function to get child nodes
        let mut get_children = async |ds_parent_node: DsNode| -> Result<Vec<DsNode>> {
            let node = ds_parent_node.get_node_id()?;

            // Retrieve the Albums under the root node
            let child_nodes = node.children(
                NodeTypeFilters::Any,
                SortDirection::Ascending,
                SortMethod::Organizer,
            )?;

            let mut to_further_process = vec![];
            // Iterate over the node children
            pin_mut!(child_nodes);
            // Just as a note this hides
            while let Some(child_node_result) = child_nodes.next().await {
                let child_node = child_node_result?;

                // log::debug!("Found node: {:?}", child_node,);
                let child_node = Arc::new(child_node);

                // Create ds_tree node
                let ds_node = DsNode::new(child_node.clone(), None);

                // Associate ds_tree child node with parent
                tree.add_node(ds_node.clone(), Some(&node))?;

                if child_node.has_children {
                    to_further_process.push(ds_node);
                }
            }
            Ok(to_further_process)
        };

        // Nodes to process children
        let mut nodes_to_process = vec![ds_parent_node];

        // While we have nodes with children keep going.
        while let Some(ds_node) = nodes_to_process.pop() {
            let mut more_nodes_to_process = get_children(ds_node).await?;
            nodes_to_process.append(&mut more_nodes_to_process);
        }
        Ok(())
    }

    /// Saves the tree and image information to the given local location
    pub fn save_meta_data<P: AsRef<Path>>(
        &self,
        node_and_album_path: P,
        image_map_path: P,
    ) -> Result<()> {
        let write_data = |contents: Vec<u8>, path| -> Result<()> {
            let contents = zstd::encode_all(contents.as_slice(), 22)?;
            Ok(std::fs::write(path, contents)?)
        };

        write_data(serde_json::to_vec(&self.tree)?, node_and_album_path)?;
        if !self.album_image_map.is_empty() {
            write_data(serde_json::to_vec(&self.album_image_map)?, image_map_path)?;
        }

        Ok(())
    }

    /// Return true if the image map is populated.
    pub fn are_images_populated(&self) -> bool {
        !self.album_image_map.is_empty()
    }

    /// Verifies artifacts are not corrupted
    pub async fn artifacts_verifier<P: AsRef<Path>>(
        &self,
        path: P,
        video_info_dir: P,
    ) -> Result<Vec<String>> {
        if !self.are_images_populated() {
            return Err(anyhow!("artifacts metadata must be populated"));
        }

        let video_infos = Arc::new(VideoInfo::load_all(video_info_dir.as_ref())?);

        let verifier = async |mut path: PathBuf,
                              image: Arc<Image>,
                              video_infos: Arc<HashMap<String, VideoInfo>>|
               -> Result<Option<String>> {
            let artifacts_dir = path.clone();
            path.push(&image.image_key);

            // SmugMug serves a "PROCESSING" placeholder instead of the original while this is set
            if image.is_processing {
                return Ok(Some(format!(
                    "Artifact {}: SmugMug reports image still processing (uploaded {}), original not available",
                    image, image.date_created
                )));
            }

            if !std::fs::exists(&path).unwrap_or_default() {
                return Ok(Some(format!("Artifact {}: not found", image)));
            }

            if std::fs::metadata(&path)?.len() == 0 {
                return Ok(Some(format!("Artifact {}: is empty", image)));
            }

            let mut reports = check_image_file(&path, &image)?;

            // For videos the artifact above is only the poster frame so check the video itself too
            if image.is_video {
                match video_infos.get(&image.image_key) {
                    None => reports.push("Video not downloaded".to_string()),
                    Some(info) => {
                        match check_video_file(
                            &info.video_path(&artifacts_dir, &image.image_key),
                            info,
                        ) {
                            Err(err) => reports.push(format!("Video not readable: {:?}", err)),
                            Ok(Some(problem)) => reports.push(problem),
                            Ok(None) => {}
                        }
                    }
                }
            }

            let final_report = if reports.is_empty() {
                None
            } else {
                Some(format!("Artifact {}: {}", image, reports.join(",")))
            };

            Ok(final_report)
        };

        let results = self
            .get_unique_artifacts_metadata()
            .into_iter()
            .filter(|v| v.archived_uri.is_some())
            .map(|v| {
                let path = PathBuf::from(path.as_ref());
                let video_infos = video_infos.clone();
                tokio::spawn(async move { verifier(path, v, video_infos).await })
            })
            .collect::<TryJoinAll<_>>()
            .await?
            .into_iter()
            .collect::<Result<Vec<Option<String>>>>()?
            .into_iter()
            .flatten()
            .collect();

        Ok(results)
    }

    /// Download artifacts from SmugMug to the given directory
    pub async fn sync_artifacts<P: AsRef<Path>>(
        &self,
        path: P,
        video_info_dir: P,
        client: Client,
    ) -> Result<()> {
        if !self.are_images_populated() {
            return Err(anyhow!("artifacts metadata must be populated"));
        }

        remove_partial_files(path.as_ref())?;
        remove_partial_files(video_info_dir.as_ref())?;
        let known_videos = VideoInfo::load_all(video_info_dir.as_ref())?;

        // Downloads the largest video rendition for a video artifact and sends its info to be saved
        let video_downloader = async |image: Arc<Image>,
                                      video_info_tx: mpsc::UnboundedSender<(String, VideoInfo)>|
               -> Result<()> {
            let artifacts_dir = path.as_ref();

            if let Some(info) = known_videos.get(&image.image_key) {
                let video_path = info.video_path(artifacts_dir, &image.image_key);
                if std::fs::metadata(&video_path).is_ok_and(|v| v.len() > 0) {
                    return Ok(());
                }
            }

            let result = with_retries(&format!("Downloading video: {}", image), || async {
                let video = {
                    let _req_limiter = self.client_req_limiter.acquire().await.unwrap();
                    image.largest_video_with_client(client.clone()).await?
                };
                let Some(video) = video else {
                    return Ok(None);
                };
                let info = VideoInfo::from(&video);
                let video_path = info.video_path(artifacts_dir, &image.image_key);

                // A video already on disk but not recorded (e.g. from an interrupted sync)
                // is kept if it matches what SmugMug reports.  Hashing it is blocking work.
                let already_downloaded = std::fs::metadata(&video_path).is_ok_and(|v| v.len() > 0)
                    && {
                        let (video_path, info) = (video_path.clone(), info.clone());
                        matches!(
                            tokio::task::spawn_blocking(move || check_video_file(
                                &video_path,
                                &info
                            ))
                            .await?,
                            Ok(None)
                        )
                    };

                let problem = if already_downloaded {
                    None
                } else {
                    log::debug!(
                        "Downloading video: {} to: {}",
                        &image,
                        video_path.as_os_str().to_string_lossy()
                    );
                    let data = {
                        let _req_limiter = self.client_req_limiter.acquire().await.unwrap();
                        video.download_with_client(client.clone()).await?
                    };
                    // Write and check the data while it's in memory, off the download task
                    let (video_path, info) = (video_path.clone(), info.clone());
                    tokio::task::spawn_blocking(move || {
                        write_file_atomically(&video_path, &data)?;
                        Ok::<_, anyhow::Error>(video_mismatch(
                            &info,
                            &bytes_md5(&data),
                            data.len() as u64,
                        ))
                    })
                    .await??
                };
                Ok(Some((info, video_path, already_downloaded, problem)))
            })
            .await?;

            match result {
                Some((info, video_path, already_downloaded, problem)) => {
                    if already_downloaded {
                        log::info!(
                            "Recorded existing Video: {} at: {}",
                            &image,
                            video_path.as_os_str().to_string_lossy()
                        );
                    } else {
                        if let Some(problem) = problem {
                            log::warn!("Video: {}: {}", &image, problem);
                        }
                        log::info!(
                            "Downloaded Video: {} to: {}",
                            &image,
                            video_path.as_os_str().to_string_lossy()
                        );
                    }
                    video_info_tx.send((image.image_key.clone(), info))?;
                }
                None => log::warn!("Video: {} has no video available on SmugMug", &image),
            }
            Ok(())
        };

        // Downloads the artifact
        let artifact_downloader = async |image: Arc<Image>,
                                         video_info_tx: mpsc::UnboundedSender<(
            String,
            VideoInfo,
        )>|
               -> Result<()> {
            let mut path = PathBuf::from(path.as_ref());
            path.push(&image.image_key);

            // The ArchivedUri returns a placeholder image while SmugMug is processing, so skip it
            if image.is_processing {
                log::warn!(
                    "Artifact: {} is still processing on SmugMug (uploaded {}). Skipping download",
                    &image,
                    image.date_created
                );
                return Ok(());
            }

            // For videos the ArchivedUri is only the poster frame, so the video is fetched separately
            if image.is_video {
                video_downloader(image.clone(), video_info_tx).await?;
            }

            if std::fs::exists(&path).unwrap_or_default() {
                if std::fs::metadata(&path)?.len() > 0 {
                    return Ok(());
                }

                log::warn!(
                    "Artifact: {} exists but is empty size. Retrying download",
                    &image
                );
            }

            log::debug!(
                "Downloading image: {} to: {}",
                &image,
                path.as_os_str().to_string_lossy()
            );

            let image_data = with_retries(
                &format!(
                    "Downloading image: {} to: {}",
                    image,
                    path.as_os_str().to_string_lossy()
                ),
                || async {
                    let _req_limiter = self.client_req_limiter.acquire().await.unwrap();
                    Ok(image.get_archive_with_client(client.clone()).await?)
                },
            )
            .await?;

            if !image_data.is_empty() {
                // Writing, hashing and decoding are blocking work, so keep them off the task
                // that drives every download
                let checks = {
                    let (path, image) = (path.clone(), image.clone());
                    tokio::task::spawn_blocking(move || {
                        write_file_atomically(&path, &image_data)?;
                        Ok::<_, anyhow::Error>(check_image_file(&path, &image))
                    })
                    .await??
                };
                log::info!(
                    "Downloaded Artifact: {} to: {}",
                    &image,
                    path.as_os_str().to_string_lossy()
                );
                // A failed check is reported, the same as by the verifier, rather than stopping the sync
                match checks {
                    Ok(problems) => {
                        for problem in problems {
                            log::warn!("Artifact: {}: {}", &image, problem);
                        }
                    }
                    Err(err) => log::warn!("Artifact: {} couldn't be checked: {:?}", &image, err),
                }
            } else {
                log::error!("Artifact: {} had 0 data", &image);
            }

            Ok(())
        };

        let (video_info_tx, video_info_rx) = mpsc::unbounded_channel();
        let downloads = async {
            let result = self
                .get_unique_artifacts_metadata()
                .into_iter()
                .filter(|v| v.archived_uri.is_some())
                .map(|v| artifact_downloader(v, video_info_tx.clone()))
                .collect::<TryJoinAll<_>>()
                .await;
            drop(video_info_tx); // Close the channel so the saver knows to stop
            result
        };

        // Saves video info in batches so an interrupted sync keeps what it already recorded
        let video_info_saver = save_in_batches(video_info_rx, 100, |batch| {
            VideoInfo::save_all(video_info_dir.as_ref(), batch)
        });

        let (downloaded, saved) = tokio::join!(downloads, video_info_saver);
        saved?;
        downloaded?;

        Ok(())
    }

    /// Downloads the EXIF/IPTC metadata SmugMug holds for each artifact into the given directory.
    /// Albums whose images all have metadata saved already are skipped.
    pub async fn sync_image_metadata<P: AsRef<Path>>(
        &self,
        image_metadata_dir: P,
        client: Client,
    ) -> Result<()> {
        if !self.are_images_populated() {
            return Err(anyhow!("artifacts metadata must be populated"));
        }

        // Images already saved, plus those saved during this run since images can be in
        // more than one album
        remove_partial_files(image_metadata_dir.as_ref())?;
        let recorded = std::sync::Mutex::new(ImageMetadataRecord::load_recorded_keys(
            image_metadata_dir.as_ref(),
        )?);

        let albums: Vec<Arc<Album>> = {
            let recorded = recorded.lock().unwrap();
            self.tree
                .get_nodes()
                .iter()
                .filter_map(|ds_node| ds_node.get_value().ok().flatten())
                .filter(|album| {
                    self.album_image_map
                        .get(&album.album_key)
                        .is_some_and(|images| {
                            images.iter().any(|v| !recorded.contains(&v.image_key))
                        })
                })
                .collect()
        };
        log::info!("Getting image metadata for {} albums", albums.len());

        // Gets the metadata for an album's images and sends the ones not yet saved
        let album_collector = async |album: Arc<Album>,
                                     tx: mpsc::UnboundedSender<ImageMetadataRecord>|
               -> Result<()> {
            let items = with_retries(
                &format!("Getting image metadata for album: {}", album),
                || async {
                    let _req_limiter = self.client_req_limiter.acquire().await.unwrap();
                    let images = album.images_with_metadata_with_client(client.clone())?;
                    pin_mut!(images);
                    let mut items = Vec::new();
                    while let Some(item) = images.next().await {
                        items.push(item?);
                    }
                    Ok(items)
                },
            )
            .await?;

            // Saves a record and marks the image as done.  Images SmugMug has no metadata for
            // get a record without metadata so their album isn't fetched again on every run.
            let save = |image: &Image, metadata: Option<&ImageMetadata>| -> Result<bool> {
                if recorded.lock().unwrap().contains(&image.image_key) {
                    return Ok(false);
                }
                tx.send(ImageMetadataRecord::new(image, metadata)?)?;
                recorded.lock().unwrap().insert(image.image_key.clone());
                Ok(true)
            };

            let mut num_saved = 0;
            let mut returned_keys = HashSet::new();
            for item in &items {
                returned_keys.insert(item.image.image_key.as_str());
                if item.metadata.is_none() {
                    log::warn!("Artifact: {} has no metadata on SmugMug", item.image);
                }
                num_saved += save(&item.image, item.metadata.as_ref())? as usize;
            }

            // Images synced locally that SmugMug no longer lists in this album
            for image in self
                .album_image_map
                .get(&album.album_key)
                .into_iter()
                .flatten()
            {
                if !returned_keys.contains(image.image_key.as_str()) && save(image, None)? {
                    log::warn!(
                        "Artifact: {} isn't listed in album: {} on SmugMug, so has no metadata",
                        image,
                        album
                    );
                }
            }

            log::info!("Got metadata for {} images in album: {}", num_saved, album);
            Ok(())
        };

        let (tx, rx) = mpsc::unbounded_channel();
        let collectors = async {
            let result = albums
                .into_iter()
                .map(|v| album_collector(v, tx.clone()))
                .collect::<TryJoinAll<_>>()
                .await;
            drop(tx); // Close the channel so the saver knows to stop
            result
        };
        let saver = save_in_batches(rx, 1000, |batch| {
            ImageMetadataRecord::save_all(image_metadata_dir.as_ref(), batch)
        });

        let (collected, saved) = tokio::join!(collectors, saver);
        saved?;
        collected?;

        Ok(())
    }

    /// Retrieves a unique list of images
    pub fn get_unique_artifacts_metadata(&self) -> Vec<Arc<Image>> {
        self.album_image_map
            .values()
            .fold(HashSet::new(), |mut acc, v| {
                acc.extend(v.iter().map(Arc::clone));
                acc
            })
            .into_iter()
            .collect()
    }

    /// Retrieves statistics about this folder
    pub fn get_folder_stats(&self) -> SmugMugFolderStats {
        let mut acc = self
            .tree
            .get_nodes()
            .iter()
            .filter_map(|ds_node| ds_node.get_value().ok().flatten())
            .fold(SmugMugFolderStats::new(), |mut acc, album| {
                acc.total_data_usage_in_bytes += album.total_sizes.unwrap_or_default() as usize;
                acc.original_data_usage_in_bytes +=
                    album.original_sizes.unwrap_or_default() as usize;
                acc.total_num_images += album.image_count as usize;
                acc
            });

        // If reading from a read only side the album data doesn't contain size so get it from the images
        if self.are_images_populated() {
            let images = self.get_unique_artifacts_metadata();

            let (mut formats, total_unique_image_size) =
                images.iter().fold((HashSet::new(), 0), |mut acc, v| {
                    acc.0.insert(v.format.to_uppercase());
                    acc.1 += v.archived_size.unwrap_or_default();
                    acc
                });

            acc.total_num_unique_images = images.len();
            acc.unique_image_data_usage_in_bytes = total_unique_image_size as usize;
            acc.artifact_formats_found = Vec::from_iter(formats.drain());
        }

        acc
    }

    /// Removes the upload keys based on the given parameters.  Returns number of albums
    /// upload keys were removed from
    pub async fn remove_albums_upload_keys(
        &self,
        client: Client,
        days_since_created: u16,
        days_since_last_updated: u16,
    ) -> Result<usize> {
        let cutoff_from_date_created_dt =
            Utc::now() - chrono::Duration::days(days_since_created as i64);
        let last_updated_cutoff_dt =
            Utc::now() - chrono::Duration::days(days_since_last_updated as i64);

        let albums_with_upload_keys: Vec<Arc<Album>> = self
            .tree
            .get_nodes()
            .iter()
            .filter_map(|ds_node| ds_node.get_value().ok().flatten())
            .filter(|album| {
                album.upload_key.is_some()
                    && (cutoff_from_date_created_dt > album.date_created.unwrap_or(Utc::now())
                        || last_updated_cutoff_dt > album.last_updated)
            })
            .collect();
        let num_albums_to_remove = albums_with_upload_keys.len();
        for album in albums_with_upload_keys {
            log::info!("Removing Upload Key From: {}", album);
            album.clear_upload_key_with_client(client.clone()).await?;
        }
        Ok(num_albums_to_remove)
    }

    /// Updates the image labels from the given JSON tag
    pub async fn update_labels_from_json_tag(
        &self,
        client: Client,
        labels_to_imgs_map: HashMap<String, Vec<String>>,
    ) -> Result<bool> {
        let updated_smugmug_image = async move |label: String, image: Arc<Image>| -> Result<bool> {
            // to update we need the image id that is on the end of the image uri
            let image_real_id = image
                .uri
                .split('/')
                .next_back()
                .ok_or_else(|| anyhow!("Failed to get image id from uri"))?;
            // log::info!("Getting image: {}", image_id);
            // let image = Image::from_id(client.clone(), &image_id).await?;

            let mut new_keywords: HashSet<&String> = image.keywords.iter().collect();
            if new_keywords.contains(&label) {
                log::info!("Image: {} already has label: {}", image_real_id, label);
                return Ok(false);
            }
            new_keywords.insert(&label);
            let data = serde_json::to_vec(&json!({"KeywordArray": new_keywords}))?;
            let req_limiter = self.client_req_limiter.acquire().await.unwrap();
            let _ =
                Image::update_image_data_with_client_from_id(client.clone(), data, image_real_id)
                    .await?;
            drop(req_limiter);
            log::info!("Updated image: {} with label: {}", image_real_id, label);
            Ok(true)
        };

        // Create an image id->Image map
        let img_map: HashMap<String, Arc<Image>> = self
            .album_image_map
            .values()
            .fold(HashSet::new(), |mut acc, v| {
                acc.extend(v.iter().map(Arc::clone));
                acc
            })
            .into_iter()
            .map(|v| (v.image_key.clone(), v))
            .collect();

        let mut workers = Vec::new();
        for (label, images) in labels_to_imgs_map.into_iter() {
            for image_id in images.into_iter() {
                if let Some(image) = img_map.get(&image_id) {
                    let label = label.clone();
                    let image = Arc::clone(image);
                    let jh = updated_smugmug_image(label, image);
                    workers.push(jh);
                }
            }
        }
        let was_updated = try_join_all(workers).await?.iter().all(|v| *v);

        Ok(was_updated)
    }
}

/// Checks a downloaded image artifact against SmugMug's metadata and that it decodes.
/// The file is streamed for the md5sum rather than read into memory.  Returns each problem found.
fn check_image_file(path: &Path, image: &Image) -> Result<Vec<String>> {
    let (digest, size) = file_md5(path)?;
    let archived_size = image
        .archived_size
        .map(|v| v.to_string())
        .unwrap_or("?".to_string());

    let mut problems = Vec::new();
    match image.archived_md5.as_ref() {
        Some(md5sum) if md5sum == &digest => {}
        Some(md5sum) => problems.push(format!(
            "Metadata doesn't match.  md5: {}:{}, size {},{}",
            md5sum, digest, archived_size, size
        )),
        None => problems.push(format!(
            "Metadata doesn't match.  md5: missing, size {},{}",
            archived_size, size
        )),
    }

    if let Err(err) = ImageReader::open(path)
        .map_err(anyhow::Error::from)
        .and_then(|v| v.with_guessed_format().map_err(anyhow::Error::from))
        .and_then(|v| v.decode().map_err(anyhow::Error::from))
    {
        problems.push(format!("Media detection failed: {:?}", err));
    }

    Ok(problems)
}

/// Checks a downloaded video against the info SmugMug reported for it.  Returns the problem if any.
fn check_video_file(path: &Path, info: &VideoInfo) -> Result<Option<String>> {
    let (digest, size) = file_md5(path)?;
    Ok(video_mismatch(info, &digest, size))
}

/// Describes how a video's md5sum and size differ from what SmugMug reported, if they do
fn video_mismatch(info: &VideoInfo, digest: &str, size: u64) -> Option<String> {
    (!info.matches(digest, size)).then(|| {
        format!(
            "Video metadata doesn't match.  md5: {}:{}, size {},{}",
            info.md5.as_deref().unwrap_or("missing"),
            digest,
            info.size.map(|v| v.to_string()).unwrap_or("?".to_string()),
            size
        )
    })
}

impl std::fmt::Display for SmugMugFolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.tree)
    }
}
#[derive(Debug, Default, Clone, Serialize)]
pub struct SmugMugFolderStats {
    pub total_data_usage_in_bytes: usize,
    pub original_data_usage_in_bytes: usize,
    pub unique_image_data_usage_in_bytes: usize,
    pub total_num_images: usize,
    pub total_num_unique_images: usize,
    pub artifact_formats_found: Vec<String>,
}
impl SmugMugFolderStats {
    fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
}
