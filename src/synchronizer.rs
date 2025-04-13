/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use smugmug::v2::{Client, Node, User};

use crate::tokens::get_full_auth_tokens;
use crate::{SmugMugPathArgs, local_folder::LocalFolder};
use anyhow::Result;

use crate::{ClearUploadKeysArgs, SyncArgs};

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

async fn get_node(client: Client, smugmug_path: &SmugMugPathArgs) -> Result<Node> {
    let node = match smugmug_path.nickname.as_ref() {
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
    if let Some(rate_limits) = client.get_last_rate_limit_window_update() {
        log::debug!(
            "Request rate limits: Remaining: {:?} retry after(s): {:?} window: {:?}",
            rate_limits.num_remaining_requests(),
            rate_limits.retry_after_seconds(),
            rate_limits.window_reset_datetime(),
        );
    }
    //TODO: use args.smugmug_path.path
    Ok(node)
}

// Handles the synchronization cli request
pub(crate) async fn handle_synchronization_req(path: &str, args: SyncArgs) -> Result<()> {
    log::debug!("Sync args {:#?}", args);

    let client = get_smugmug_client().await;

    let node = get_node(client.clone(), &args.smugmug_path).await?;

    let local_folder = LocalFolder::get_or_create(
        path,
        client.clone(),
        node,
        args.download_artifact_info || args.download_artifacts,
        args.force,
    )
    .await?;

    if args.download_artifacts {
        local_folder.sync_artifacts().await?;
    }

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

    log::debug!("Finished syncing");
    Ok(())
}

pub(crate) async fn handle_clear_key_req(path: &str, args: ClearUploadKeysArgs) -> Result<()> {
    log::trace!("Clearing upload keys {:#?}", args);
    let client = get_smugmug_client().await;
    let node = get_node(client.clone(), &args.smugmug_path).await?;

    let local_folder = LocalFolder::get_or_create(path, client.clone(), node, false, false).await?;
    let num_removed = local_folder
        .remove_albums_upload_keys(
            args.days_since_album_was_created.unwrap_or(60),
            args.days_since_last_album_update.unwrap_or(45),
        )
        .await?;

    println!("Upload keys removed from {num_removed} albums");
    Ok(())
}

pub(crate) async fn handle_updating_tags_req(_smugmug_path: &SmugMugPathArgs) -> Result<()> {
    todo!()
}
