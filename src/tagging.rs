/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use crate::{TaggingArgs, local_folder, synchronizer::handle_updating_tags_req};
use anyhow::Result;

// Handles the tagging cli request
pub(crate) async fn handle_tagging_req(path: &str, args: TaggingArgs) -> Result<()> {
    log::trace!("Tagging args {:#?}", args);
    if args.update_smugmug_tags {
        return handle_updating_tags_req(&args.smugmug_path).await;
    }

    let local_folder = local_folder::LocalFolder::get(path)?;

    if args.gen_thumbnails_and_embeddings {
        local_folder.generate_thumbnails_and_embeddings().await?;
    } else if args.gen_labels {
        match args.presorted_thumbnails_dir {
            None => {
                return Err(anyhow::anyhow!(
                    "--gen-labels requires --presorted-thumbnails-dir"
                ));
            }
            Some(ref presorted_thumbnails_dir) => {
                local_folder
                    .generate_labels_from_presorted_dir(presorted_thumbnails_dir)
                    .await?;
            }
        }
    } else {
        return Err(anyhow::anyhow!("No valid tagging option provided"));
    }

    Ok(())
}
