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
use futures::{StreamExt, future::TryJoinAll, pin_mut};
use smugmug::v2::{
    Album, Client, Image, Node, NodeType, NodeTypeFilters, SortDirection, SortMethod,
};

use tree_ds::prelude as TreeDs;

type DsTree = TreeDs::Tree<Arc<Node>, Arc<Album>>;
type DsNode = TreeDs::Node<Arc<Node>, Arc<Album>>;

/// Represents a folder in SmugMug
#[derive(Debug)]
pub struct SmugMugFolder {
    tree: DsTree,
    album_image_map: HashMap<String, Vec<Arc<Image>>>,
}

impl SmugMugFolder {
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
        })
    }

    /// Go through the Albums and get the images information
    pub async fn populate_image_map(&self, client: Client) -> Result<Self> {
        // Collects the images for an album and returns a tuple of album key and image collection
        let image_child_collector = async |album: Arc<Album>| -> Result<(String, Vec<Arc<Image>>)> {
            log::debug!("Collecting image metadata for album: {}", &album);
            let images = album.images_with_client(client.clone())?;

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

        // Serial way to get image information
        // let mut album_image_map = HashMap::new();
        // for album in albums {
        //     log::debug!("Collecting images for: {}", &album);
        //     let (id, image_collection) = image_child_collector(album).await?;
        //     album_image_map.insert(id, image_collection);
        // }

        log::debug!("Finished collecting image metadata");

        Ok(Self {
            tree: self.tree.clone(),
            album_image_map,
        })
    }

    /// Populates the Folder meta data from the provide file
    pub fn populate_from_file<P: AsRef<Path>>(
        album_node_file: P,
        image_map_file_opt: Option<P>,
    ) -> Result<Self> {
        let get_data = |path| -> Result<Vec<u8>> {
            let file = File::open(path)?;
            let reader = BufReader::new(file);
            Ok(zstd::decode_all(reader)?)
        };

        // let tree: DsTree = flexbuffers::from_slice(&reader)?;
        let tree: DsTree = serde_json::from_slice(&get_data(album_node_file)?)?;

        let album_image_map: HashMap<String, Vec<Arc<Image>>> = match image_map_file_opt {
            Some(image_map) => serde_json::from_slice(&get_data(image_map)?)?,
            None => HashMap::new(),
        };

        Ok(Self {
            tree,
            album_image_map,
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
            //let contents = flexbuffers::to_vec(&self.tree)?;
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

    /// Download artifacts from SmugMug to the given directory
    pub async fn sync_artifacts<P: AsRef<Path>>(&self, path: P, client: Client) -> Result<()> {
        if !self.are_images_populated() {
            return Err(anyhow!("artifacts metadata must be populated"));
        }

        let artifact_downloader = async |image: Arc<Image>| -> Result<()> {
            let client = client.clone();
            let mut path = PathBuf::from(path.as_ref());
            path.push(&image.image_key);

            if let Some(md5sum) = image
                .archived_md5
                .as_ref()
                .filter(|_| std::fs::exists(&path).unwrap_or_default())
            {
                let existing_data = std::fs::read(&path)?;
                let digest = format!("{:x}", md5::compute(existing_data));
                if &digest == md5sum {
                    log::debug!(
                        "Skipping download of artifact: {} due to already exists",
                        &image
                    );
                    return Ok(());
                }
            }

            log::debug!(
                "Downloading image: {} to: {}",
                &image,
                path.as_os_str().to_string_lossy()
            );

            let image_data = image.get_archive_with_client(client).await?;

            match image.archived_md5.as_ref() {
                Some(md5sum) => {
                    let digest = format!("{:x}", md5::compute(&image_data));
                    if &digest != md5sum {
                        log::error!(
                            "Image {} md5sum didn't match: {} calculated:{}, size: {:?}, downloaded size: {}",
                            &image,
                            md5sum,
                            digest,
                            image.archived_size,
                            image_data.len()
                        );
                    }
                }
                None => log::error!("Image {} did not contain an md5sum", &image),
            };

            if !image_data.is_empty() {
                std::fs::write(&path, image_data)?;
                log::debug!(
                    "Finished downloading image: {} to: {}",
                    &image,
                    path.as_os_str().to_string_lossy()
                );
            } else {
                log::error!("Image: {} had 0 data", &image);
            }

            Ok(())
        };

        self.get_unique_images()
            .into_iter()
            .filter(|v| v.archived_uri.is_some())
            .map(artifact_downloader)
            .collect::<TryJoinAll<_>>()
            .await?;

        Ok(())
    }

    /// Retrieves a unique list of images
    pub fn get_unique_images(&self) -> Vec<Arc<Image>> {
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
            let images = self.get_unique_images();

            let total_unique_image_size = images
                .iter()
                .map(|v| v.archived_size.unwrap_or_default())
                .sum::<u64>();

            acc.total_num_unique_images = images.len();
            acc.unique_image_data_usage_in_bytes = total_unique_image_size as usize;
        }

        acc
    }
}

impl std::fmt::Display for SmugMugFolder {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.tree)
    }
}
#[derive(Debug, Default, Clone)]
pub struct SmugMugFolderStats {
    pub total_data_usage_in_bytes: usize,
    pub original_data_usage_in_bytes: usize,
    pub unique_image_data_usage_in_bytes: usize,
    pub total_num_images: usize,
    pub total_num_unique_images: usize,
}
impl SmugMugFolderStats {
    fn new() -> Self {
        Self {
            ..Default::default()
        }
    }
}
