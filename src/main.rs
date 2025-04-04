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
mod tokens;

use anyhow::Result;
use clap::{Args, CommandFactory, Parser, Subcommand, error::ErrorKind};
use dotenvy::dotenv;
use path_finder::PathFinder;
use querier::handle_query_req;
use synchronizer::handle_synchronization_req;

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

    /// Calculates size synchronized data would take up.
    #[arg(short, long)]
    pub(crate) calc_size: bool,
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
            if sync_args.smugmug_path.nickname.is_none() {
                sync_args.smugmug_path.nickname = std::env::var("SMUGMUG_NICKNAME").ok();
            }
            // if sync_args.smugmug_path.path.is_none() {
            //     sync_args.smugmug_path.path = std::env::var("SMUGMUG_PATH").ok();
            // }
            handle_synchronization_req(&local_path, sync_args).await?;
        }
        Commands::Query(query_args) => {
            handle_query_req(&local_path, query_args).await?;
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
