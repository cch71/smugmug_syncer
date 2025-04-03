/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use smugmug::v2::{Client, User};

use crate::PathFinder;
use crate::smugmug_folder::SmugMugFolder;
use crate::tokens::get_full_auth_tokens;
use anyhow::Result;

use crate::SyncArgs;

static CLIENT: tokio::sync::OnceCell<Client> = tokio::sync::OnceCell::const_new();

async fn get_smugmug_client() -> Client {
    CLIENT
        .get_or_init(|| async {
            let creds = get_full_auth_tokens()
                .await
                .expect("Failed to get auth tokens");
            Client::new(creds)
        })
        .await
        .clone()
}

async fn download_node_and_album_info(args: &SyncArgs) -> Result<SmugMugFolder> {
    let client = get_smugmug_client().await;
    let node = match args.smugmug_path.nickname.as_ref() {
        None => {
            User::authenticated_user_info(client.clone())
                .await?
                .node()
                .await?
        }
        Some(nickname) => {
            User::from_id(client.clone(), nickname)
                .await?
                .node()
                .await?
        }
    };

    //TODO: use args.smugmug_path.path

    SmugMugFolder::populate_from_node(client.clone(), node).await
}

// Handles the syrchronizaition cli request
pub(crate) async fn handle_synchronization_req(
    path_finder: &PathFinder,
    args: SyncArgs,
) -> Result<()> {
    log::debug!("Sync args {:#?}", args);
    let mut folder = if args.force || !path_finder.does_album_and_node_data_file_exists()? {
        download_node_and_album_info(&args).await?
    } else {
        // Retrieve all from file cache
        let album_map_file_opt = if path_finder.does_album_image_map_file_exists()? {
            Some(path_finder.get_album_image_map_file())
        } else {
            None
        };
        SmugMugFolder::populate_from_file(
            path_finder.get_album_and_node_data_file(),
            album_map_file_opt,
        )?
    };

    // Download the Image/Video meta data
    if args.download_artifact_info && (args.force || !folder.are_images_populated()) {
        let client = get_smugmug_client().await;
        folder = folder.populate_image_map(client).await?
    }

    // Save generated tree to file
    folder.save_meta_data(
        path_finder.get_album_and_node_data_file(),
        path_finder.get_album_image_map_file(),
    )?;

    // let end_limit = client
    //     .get_last_rate_limit_window_update()
    //     .unwrap()
    //     .num_remaining_requests()
    //     .unwrap();

    // log::debug!(
    //     "Request limit left: {}.  Used {} requests.",
    //     end_limit,
    //     start_limit - end_limit
    // );

    Ok(())
}
