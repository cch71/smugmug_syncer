/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */
mod local_folder;
mod path_finder;
mod querier;
mod smugmug_folder;
mod synchronizer;
mod tagging;
mod tokens;
mod face_detector;

use anyhow::Result;
use clap::{Args, CommandFactory, Parser, Subcommand, error::ErrorKind};
use dotenvy::dotenv;
use path_finder::PathFinder;
use querier::handle_query_req;
use synchronizer::{handle_clear_key_req, handle_synchronization_req};
use tagging::handle_tagging_req;

macro_rules! fill_smugmug_path_args_from_env {
    ($a:expr) => {
        if $a.smugmug_path.nickname.is_none() {
            $a.smugmug_path.nickname = std::env::var("SMUGMUG_NICKNAME").ok();
        }
        // if $a.smugmug_path.path.is_none() {
        //     $a.smugmug_path.path = std::env::var("SMUGMUG_PATH").ok();
        // }
    };
}

// CLI Definitions
static LONG_ABOUT: &str = r#"
This is a utility for backing up and getting information about SmugMug accounts.

Expected/Required environment variables:
SMUGMUG_API_KEY - This is required for accessing public SmugMug Data;

SMUGMUG_API_SECRET - Required for full access to SmugMug Data
SMUGMUG_AUTH_CACHE - Local path to cache authentication tokens

SMUGMUG_NICKNAME - Can be used instead of CLI args to set the user nickname to use
SMUGMUG_SYNC_LOCATION - Can be used instead of the --syncto arg
SMUGMUG_SYNC_WORKERS - Number of parallel smugmug req to make default is 1
A .env file can be created in the working directory that contains these as well.
"#;

#[derive(Debug, Parser)] // requires `derive` feature
#[command(name = "smugmug-syncer")]
#[command(about = "Utility for managing local backup of SmugMug data", long_about = Some(LONG_ABOUT))]
struct Cli {
    /// Directory where synchronized data will be stored.
    // #[arg(required = true)]
    #[arg(long)]
    syncto: Option<String>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Commands {
    /// Synchronizes the Folder/Album and optionally Image data from SmugMug.  
    ///   Does not overwrite if data already exists in the --syncto dir.
    #[command(arg_required_else_help = true)]
    Sync(SyncArgs),

    /// Performs analysis of synced data
    #[command(arg_required_else_help = true)]
    Query(QueryArgs),

    /// Clears the upload keys from the albums in a folder
    #[command(arg_required_else_help = false)]
    ClearKeys(ClearUploadKeysArgs),

    /// Tagging functionality
    #[command(arg_required_else_help = true)]
    Tags(TaggingArgs),
}

#[derive(Debug, Args)]
pub(crate) struct SyncArgs {
    /// Will synchronize the Image/Video meta information (Not the image/videos themselves).
    #[arg(long)]
    pub(crate) download_artifact_info: bool,

    /// Will download Image/Videos not already downloaded. Implies --download-artifact-info
    #[arg(long)]
    pub(crate) download_artifacts: bool,

    #[command(flatten)]
    pub(crate) smugmug_path: SmugMugPathArgs,

    /// Forces resync of data
    #[arg(long)]
    pub(crate) force: bool,
}

#[derive(Debug, Args)]
pub(crate) struct ClearUploadKeysArgs {
    /// Days since album was last updated.  Default to 45
    #[arg(long)]
    pub(crate) days_since_last_album_update: Option<u16>,

    /// Days since album was created.  Defaults to 60
    #[arg(long)]
    pub(crate) days_since_album_was_created: Option<u16>,

    #[command(flatten)]
    pub(crate) smugmug_path: SmugMugPathArgs,
}

#[derive(Debug, Args)]
pub(crate) struct SmugMugPathArgs {
    /// Nickname of user for account to sync with. (Defaults to API user)
    #[arg(short = 'n', long)]
    pub(crate) nickname: Option<String>,
    //
    // /// Path name to SmugMug location. Default to root node [/].
    // #[arg(short = 'p', long)]
    // pub(crate) path: Option<String>,
}

#[derive(Debug, Args)]
pub(crate) struct QueryArgs {
    /// Prints the Folder/Album Tree
    #[arg(short, long)]
    pub(crate) print_tree: bool,

    /// Show images for Album
    #[arg(short, long)]
    pub(crate) list_images_from_album_id: Option<String>,

    /// Calculates stats about the synced metadata.
    #[arg(short, long)]
    pub(crate) stats: bool,

    /// Checks for invalid files
    #[arg(long)]
    pub(crate) check_for_invalid_artifacts: bool,

    /// Outputs the results as JSON
    #[arg(short = 'j', long)]
    pub(crate) use_json_output: bool,
}

#[derive(Debug, Args)]
pub(crate) struct TaggingArgs {
    /// Generates thumbnails and embeddings from the images
    #[arg(long)]
    pub(crate) gen_thumbnails_and_embeddings: bool,

    /// Generates labels based on images found in the pre-sorted thumbnails directory
    #[arg(long)]
    pub(crate) gen_labels: bool,

    /// Directory where thumbnails have been manually sorted. REQUIRED if --gen-labels is set
    #[arg(short = 'd', long)]
    pub(crate) presorted_thumbnails_dir: Option<String>,

    /// Update smugmug image tags
    #[arg(long)]
    pub(crate) update_smugmug_tags: bool,

    #[command(flatten)]
    pub(crate) smugmug_path: SmugMugPathArgs,
}

async fn handle_cli_arg(args: Cli) -> Result<()> {
    let local_path = args
        .syncto
        .or(std::env::var("SMUGMUG_SYNC_LOCATION").ok())
        .unwrap_or_else(|| {
            let mut cmd = Cli::command();
            cmd.error(
                ErrorKind::MissingRequiredArgument,
                "--syncto directory must be set",
            )
            .exit()
        });

    log::debug!("Using Local Path: {}", &local_path);

    match args.command {
        Commands::Sync(mut sync_args) => {
            fill_smugmug_path_args_from_env!(sync_args);
            handle_synchronization_req(&local_path, sync_args).await?;
        }
        Commands::Query(query_args) => {
            handle_query_req(&local_path, query_args).await?;
        }
        Commands::ClearKeys(mut clear_key_args) => {
            fill_smugmug_path_args_from_env!(clear_key_args);
            handle_clear_key_req(&local_path, clear_key_args).await?;
        }
        Commands::Tags(mut tagging_args) => {
            fill_smugmug_path_args_from_env!(tagging_args);
            handle_tagging_req(&local_path, tagging_args).await?;
        }
    };
    Ok(())
}

// CLI tool for synchronizing with smugmug accounts
#[tokio::main]
async fn main() -> Result<()> {
    dotenv().ok();
    env_logger::init();
    log::debug!("+++ syncer");
    let args = Cli::parse();

    handle_cli_arg(args).await?;

    // log::debug!("Folder: {}", folder);
    log::debug!("--- syncer");

    Ok(())
}
