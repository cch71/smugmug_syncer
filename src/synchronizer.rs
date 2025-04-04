/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use smugmug::v2::{Client, User};

use crate::local_folder::LocalFolder;
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

// Handles the synchronization cli request
pub(crate) async fn handle_synchronization_req(path: &str, args: SyncArgs) -> Result<()> {
    log::debug!("Sync args {:#?}", args);

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
