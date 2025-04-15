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
    io::{BufReader, Cursor},
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
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
    Album, Client, Image, Node, NodeType, NodeTypeFilters, SortDirection, SortMethod,
};

use tree_ds::prelude as TreeDs;

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
            .flat_map(|ds_node| ds_node.get_value())
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
            .filter(|ds_node| matches!(ds_node.get_node_id().node_type, NodeType::Album))
            .map(|ds_node| {
                let node = ds_node.get_node_id();
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
            albums.into_iter().for_each(|album| {
                let ds_node = album_to_node_map.remove(&album.album_key).unwrap();
                ds_node.set_value(Some(Arc::new(album)));
            });
        }

        Ok(())
    }

    // Goes through the nodes and builds up the node tree
    async fn build_node_tree(tree: &mut DsTree, ds_parent_node: DsNode) -> Result<()> {
        // function to get child nodes
        let mut get_children = async |ds_parent_node: DsNode| -> Result<Vec<DsNode>> {
            let node = ds_parent_node.get_node_id();

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
                tree.add_node(ds_node.clone(), Some(&ds_parent_node.get_node_id()))?;

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
    pub async fn artifacts_verifier<P: AsRef<Path>>(&self, path: P) -> Result<Vec<String>> {
        if !self.are_images_populated() {
            return Err(anyhow!("artifacts metadata must be populated"));
        }

        let verifier = async |mut path: PathBuf, image: Arc<Image>| -> Result<Option<String>> {
            path.push(&image.image_key);

            if !std::fs::exists(&path).unwrap_or_default() {
                return Ok(Some(format!("Artifact {}: not found", image)));
            }

            let data = std::fs::read(&path)?;
            if data.is_empty() {
                return Ok(Some(format!("Artifact {}: is empty", image)));
            }

            let metadata_report = if let Some(md5sum) = image.archived_md5.as_ref() {
                let digest = format!("{:x}", md5::compute(&data));
                if &digest != md5sum {
                    format!(
                        "Metadata doesn't match.  md5: {}:{}, size {},{}",
                        md5sum,
                        digest,
                        image
                            .archived_size
                            .map(|v| v.to_string())
                            .unwrap_or("?".to_string()),
                        data.len()
                    )
                } else {
                    String::new()
                }
            } else {
                format!(
                    "Metadata doesn't match.  md5: missing, size {},{}",
                    image
                        .archived_size
                        .map(|v| v.to_string())
                        .unwrap_or("?".to_string()),
                    data.len()
                )
            };

            let image_detection_report = match ImageReader::new(Cursor::new(data))
                .with_guessed_format()
                .map_err(anyhow::Error::from)
                .and_then(|v| v.decode().map_err(anyhow::Error::from))
            {
                Err(err) => format!("Media detection failed: {:?}", err),
                _ => String::new(),
            };

            let final_report = if metadata_report.is_empty() && image_detection_report.is_empty() {
                None
            } else {
                let final_report = format!(
                    "Artifact {}: {}",
                    image,
                    [metadata_report, image_detection_report].join(",")
                );
                Some(final_report)
            };

            Ok(final_report)
        };

        let results = self
            .get_unique_artifacts_metadata()
            .into_iter()
            .filter(|v| v.archived_uri.is_some())
            .map(|v| {
                let path = PathBuf::from(path.as_ref());
                tokio::spawn(async move { verifier(path, v).await })
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
    pub async fn sync_artifacts<P: AsRef<Path>>(&self, path: P, client: Client) -> Result<()> {
        if !self.are_images_populated() {
            return Err(anyhow!("artifacts metadata must be populated"));
        }

        // verifies the binary data
        let binary_verifier = async |prefix_msg: &str, image: Arc<Image>, data: Vec<u8>| {
            match image.archived_md5.as_ref() {
                Some(md5sum) => {
                    let digest = format!("{:x}", md5::compute(&data));
                    if &digest != md5sum {
                        log::warn!(
                            "{}Artifact: {} md5sum didn't match: {} calculated:{}, size: {:?}, downloaded size: {}",
                            prefix_msg,
                            &image,
                            md5sum,
                            digest,
                            image.archived_size,
                            data.len()
                        );
                    }
                }
                None => {
                    log::warn!(
                        "{}Artifact: {} due to already exists. NOTE!!! The smugmug md5sum doesn't exist",
                        prefix_msg,
                        &image
                    );
                }
            };

            if let Err(err) = ImageReader::new(Cursor::new(data))
                .with_guessed_format()
                .map_err(anyhow::Error::from)
                .and_then(|v| v.decode().map_err(anyhow::Error::from))
            {
                log::warn!(
                    "Artifact: {} binary type failed to be detected: {:?}",
                    image,
                    err
                );
            }
        };

        // Downloads the artifact
        let artifact_downloader = async |image: Arc<Image>| -> Result<()> {
            let mut path = PathBuf::from(path.as_ref());
            path.push(&image.image_key);

            if std::fs::exists(&path).unwrap_or_default() {
                let existing_data = std::fs::read(&path)?;
                if !existing_data.is_empty() {
                    // tokio::spawn(async move {
                    //     binary_verifier("Skipping download of ", image, existing_data).await;
                    // });
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

            let mut retries = 0;
            let image_data = loop {
                let req_limiter = self.client_req_limiter.acquire().await.unwrap();
                let image_get_result = image.get_archive_with_client(client.clone()).await;
                drop(req_limiter);
                match image_get_result {
                    Ok(data) => break data,
                    Err(err) => {
                        retries += 1;
                        if retries >= 3 {
                            return Err(anyhow::Error::from(err));
                        } else {
                            log::warn!(
                                "Downloading image: {} to: {} retry #: {}",
                                &image,
                                path.as_os_str().to_string_lossy(),
                                retries,
                            );
                        }
                    }
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            };

            if !image_data.is_empty() {
                std::fs::write(&path, &image_data)?;
                log::info!(
                    "Downloaded Artifact: {} to: {}",
                    &image,
                    path.as_os_str().to_string_lossy()
                );
                tokio::spawn(async move {
                    binary_verifier("", image, image_data.to_vec()).await;
                });
            } else {
                log::error!("Artifact: {} had 0 data", &image);
            }

            Ok(())
        };

        self.get_unique_artifacts_metadata()
            .into_iter()
            .filter(|v| v.archived_uri.is_some())
            .map(artifact_downloader)
            .collect::<TryJoinAll<_>>()
            .await?;

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
            .flat_map(|ds_node| ds_node.get_value())
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
            .flat_map(|ds_node| ds_node.get_value())
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
