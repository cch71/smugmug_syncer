# SmugMug Synchronizer

This tool provides some basic functionality for manipulating SmugMug accounts.
It can:

- Backup the contents of a SmugMug account. It does this by downloading the
  metadata for Nodes, Albums, and Images. It will also download the unique
  images and store them by their image key.
  - Reproducing the albums via symlinks is possible however most cloud storage
    systems do not support this and this approach reduces duplicate images.
- Prints out the tree to enable matching the image id with the Album/Image name.
- Removes upload keys for an account based on time period.
- Generates stats about the SmugMug account.
- Supports the ability to tag images via AI analysis.
  - This is a multi step process that will be explained in the
    [Image Labeling instructions](HOWTO_LABEL_IMAGES.md)

## Building

It should build/run on MacOS, Linux, and Windows (via WSL)

### Docker/Podman build instructions

```console
➜  git clone https://github.com/cch71/smugmug_syncer syncer

➜  cd syncer

➜  docker build . -t smugmug-syncer
   ...

➜ docker run \
  --env-file .env \
  -v <SMUGMUG_SYNC_LOCATION>:<SMUGMUG_SYNC_LOCATION> \
  -v <SMUGMUG_AUTH_CACHE>:<SMUGMUG_AUTH_CACHE> \
  -v <SMUGMUG_SYNC_MODELS_DIR>:<SMUGMUG_SYNC_MODELS_DIR>:ro \
  -it --rm \
  smugmug-syncer query -p
```

Replace the `-v` bracket entries `< >` with actual paths

### Manual build instructions for MacOS/Linux/WSL(Ubuntu)

- Install rust tools: [How to install Rust](https://www.rust-lang.org/tools/install)
- Install AI dependencies: dlib, nlohmann-json, and opencv
  - Recommend using `brew` to install on MacOS
  - For Linux refer to the Dockerfile for exact deps on Ubuntu
- Clone the repo
- cd cloned repo
- run cargo build --release
- run it

```console
➜  git clone https://github.com/cch71/smugmug_syncer syncer

➜  cd syncer

➜  cargo build --release
   Compiling proc-macro2 v1.0.94
   ...

➜  target/release/smugmug-syncer --help
```

## Getting started

Note: _If building from source (Binary isn't provided yet) then you need to get
an API key from SmugMug_

### To synchronize metadata

```console
smugmug-syncer --syncto ~/Pictures/SmugMugArchive sync --download-artifact-info --nickname apidemo
```

- Create a folder if it doesn't exist at ~/Pictures/SmugMugArchive
- Use the account `apidemo` which equates to [https://apidemo.smugmug.com](https://apidemo.smugmug.com)
- Download all Folder/Album and, with the --download-artifact-info, download the
  images for the given account.
  - this does not download the images/videos themselves just the metadata

#### Equivalent with Docker/Podman

create an .env file that looks like

```file
SMUGMUG_API_KEY=<API_KEY>
SMUGMUG_API_SECRET=<API_SECRET>
SMUGMUG_AUTH_CACHE=<PATH TO STORE CACHED AUTH CREDS>
SMUGMUG_SYNC_LOCATION=<PATH TO SYNC SMUGMUG ACCOUNT TO (--syncto)>
SMUGMUG_SYNC_MODELS_DIR=<PATH TO AI MODELS DIR>
SMUGMUG_NICKNAME=<SMUGMUG ACCOUNT NICKNAME>
SMUGMUG_SYNC_WORKERS=2
```

```console
➜ docker run \
  --env-file .env \
  -v <SMUGMUG_SYNC_LOCATION>:<SMUGMUG_SYNC_LOCATION> \
  -v <SMUGMUG_AUTH_CACHE>:<SMUGMUG_AUTH_CACHE> \
  -v <SMUGMUG_SYNC_MODELS_DIR>:<SMUGMUG_SYNC_MODELS_DIR>:ro \
  -it --rm \
  smugmug-syncer query sync --download-artifact-info
```

### To download images/videos

```console
smugmug-syncer --syncto ~/Pictures/SmugMugArchive sync --download-artifacts --nickname apidemo
```

This will download the images video into an `artifacts` folder at the given location.
The images/videos will be named by their SmugMug image key. _Future versions of this
utility may create a symlink representation of the Album/Image/Folder relationship._

### Print out tree representation of the metadata

```console
smugmug-syncer --syncto ~/Pictures/Troop27SmugMugArchive query -p -c
```

- `-p` causes the tree to be printed out
- `-s` will cause statistics about metadata to be calculated

Unless otherwise stated, the queries flags do not require the images/video
artifacts to be downloaded.

## CLI Usage

### General help

```console
Usage: smugmug-syncer [OPTIONS] <COMMAND>

Commands:
  sync   Synchronizes the Folder/Album and optionally Image data from SmugMug.
             Does not overwrite if already exist at the --syncto dir
  query  Performs analysis of synced data
  help   Print this message or the help of the given subcommand(s)

Options:
      --syncto <SYNCTO>
          Directory where synchronized data will be stored

  -h, --help
          Print help (see a summary with '-h')
```

### Sync help

```console
Synchronizes the Folder/Album and optionally Image data from SmugMug.
    Does not overwrite if already exist at the --syncto dir

Usage: smugmug-syncer sync [OPTIONS]

Options:
      --download-artifact-info  Will synchronize the Image/Video meta information (Not the image/videos themselves)
      --download-artifacts      Will download Image/Videos not already downloaded. Implies --download-artifact-info
  -n, --nickname <NICKNAME>     Nickname of user for account to sync with. (Defaults to API user)
      --force                   Forces resync of data
  -h, --help                    Print help
```

### Query help

```console
Performs analysis of synced data

Usage: smugmug-syncer query [OPTIONS]

Options:
  -p, --print-tree
          Prints the Folder/Album Tree
  -l, --list-images-from-album-id <LIST_IMAGES_FROM_ALBUM_ID>
          Show images for Album
  -s, --stats
          Calculates stats about the synced metadata
  -h, --help
          Print help
```

### Tag/Labeling Help

For detaild Tag/Labeling support see: [Image Labeling instructions](HOWTO_LABEL_IMAGES.md)

```console
Usage: smugmug-syncer tags [OPTIONS]

Options:
      --gen-thumbnails-and-embeddings
          Generates thumbnails and embeddings from the images
      --gen-labels
          Generates labels based on images found in the pre-sorted thumbnails directory
  -d, --presorted-thumbnails-dir <PRESORTED_THUMBNAILS_DIR>
          Directory where thumbnails have been manually sorted. REQUIRED if --gen-labels is set
      --update-smugmug-tags
          Update smugmug image tags
  -n, --nickname <NICKNAME>
          Nickname of user for account to sync with. (Defaults to API user)
  -h, --help
          Print help
```

## Expected/Required environment variables

Required for working with any online SmugMug folders:

`SMUGMUG_API_KEY` - This is required for accessing public SmugMug albums.

Required for modifying/creating on the online SmugMug platform.

`SMUGMUG_API_SECRET` - Required for full access to SmugMug Data
`SMUGMUG_AUTH_CACHE` - Local path to cache authentication tokens

Useful for not having to specify it on the command line repeatedly.

`SMUGMUG_NICKNAME` - Can be used instead of CLI args to set the user nickname to use
`SMUGMUG_SYNC_LOCATION` - Can be used instead of the --syncto arg
`SMUGMUG_SYNC_WORKERS` - Number of parallel SmugMug req to make default is 1

`SMUGMUG_SYNC_MODELS_DIR` - Directory that holds the data models for the facial
recognition/tagging functionality

_A .env file can be created in the working directory that contains these as well._

## License

Licensed under either of
<a href="LICENSE-APACHE">Apache License Version 2.0</a> or <a href="LICENSE-MIT">MIT license</a>
at your option.

## Contributions

Contributions are welcome.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
