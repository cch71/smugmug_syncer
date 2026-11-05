# SmugMug Synchronizer

[![CI](https://github.com/cch71/smugmug_syncer/actions/workflows/ci.yml/badge.svg)](https://github.com/cch71/smugmug_syncer/actions/workflows/ci.yml)
[![CodeQL](https://github.com/cch71/smugmug_syncer/actions/workflows/codeql.yml/badge.svg)](https://github.com/cch71/smugmug_syncer/actions/workflows/codeql.yml)
[![Security](https://github.com/cch71/smugmug_syncer/actions/workflows/security.yml/badge.svg)](https://github.com/cch71/smugmug_syncer/actions/workflows/security.yml)

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

## Installing a release

Prebuilt binaries are on the [Releases page](https://github.com/cch71/smugmug_syncer/releases).
Each archive holds the `smugmug-syncer` binary and a `DEPENDENCIES.txt` listing the system
libraries it needs. `SHA256SUMS.txt` lists a checksum for each archive.

| Platform | Archive | Libraries |
| --- | --- | --- |
| macOS, Apple silicon | `smugmug-syncer_<version>_macos-aarch64.tar.gz` | `brew install dlib opencv` |
| Linux x86_64 (Ubuntu 24.04+, Debian 13+) | `smugmug-syncer_<version>_linux-x86_64.tar.gz` | the apt packages in `DEPENDENCIES.txt` |

```console
➜  tar -xzf smugmug-syncer_<version>_<platform>.tar.gz
➜  cd smugmug-syncer_<version>_<platform>
➜  ./smugmug-syncer --version
```

On macOS, a downloaded binary is quarantined; clear it with
`xattr -d com.apple.quarantine smugmug-syncer`. The Linux binary is built against Ubuntu
24.04's dlib and OpenCV 4.6 packages, so it needs glibc 2.39 or newer. Other platforms (Intel
macOS, Windows via WSL with an older Ubuntu) can build from source as below, or use Docker.

### Archive database version

The archive's `.smugmug_db/props.json` records `db_version`, the
[semver](https://semver.org/) version of the database layout (`smugmug-syncer --version`
shows the one a build uses). A change to the major version means older readers, such as an
older smugmug_syncer_viewer, can't read the archive; a minor version adds data older readers
ignore. The syncer refuses to write to an archive with a different major version.

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

For videos SmugMug's archived file is only a still frame, so the largest video SmugMug has is
also downloaded as `artifacts/<image key>.<ext>` (e.g. `.mp4`).

### To download image metadata (EXIF/IPTC)

```console
smugmug-syncer --syncto ~/Pictures/SmugMugArchive sync --download-image-metadata --nickname apidemo
```

This saves the metadata SmugMug holds for each image/video (camera, date taken, exposure,
location, ...) as parquet files in `.smugmug_db/image_metadata`. SmugMug keeps this even when
the file it serves for download has had its EXIF removed.

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
      --download-artifact-info   Will synchronize the Image/Video meta information (Not the image/videos themselves)
      --download-artifacts       Will download Image/Videos not already downloaded. Implies --download-artifact-info
      --download-image-metadata  Will download the EXIF/IPTC metadata SmugMug holds for each Image/Video (camera, date taken, location, ...) not already downloaded. Implies --download-artifact-info
  -n, --nickname <NICKNAME>      Nickname of user for account to sync with. (Defaults to API user)
      --force                    Forces resync of data
  -h, --help                     Print help
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

## Development

```console
cargo fmt --all --check
cargo clippy --release --all-targets -- -D warnings
cargo test --release
cargo deny check          # advisories, licenses, sources
```

## Releasing

Releases are made with [cargo-release](https://github.com/crate-ci/cargo-release) (configured
in [release.toml](release.toml)). Nothing is published to crates.io.

1. Add the changes to the `## [Unreleased]` section of [CHANGELOG.md](CHANGELOG.md) and
   commit them. If the `.smugmug_db` layout changed, bump `DB_VERSION` in
   `src/local_folder.rs` too (see [AGENTS.md](AGENTS.md#database-version)).
2. Do a dry run, then the release. Use `patch`, `minor` or `major` to choose the version, or
   leave it out to release the version already in `Cargo.toml`:

   ```bash
   cargo release patch
   cargo release patch --execute
   ```

   This bumps the version and dates the changelog. It then commits, creates the tag `vX.Y.Z`,
   and pushes both.
3. The pushed tag starts [the release workflow](.github/workflows/release.yml). It builds
   `.tar.gz` archives for Linux x86_64 and macOS Apple silicon, then creates the GitHub
   release with notes from the changelog (plus the database version), attaches the archives
   and `SHA256SUMS.txt`, and adds GitHub's generated notes.

A release can also be started by creating a `vX.Y.Z` tag from GitHub's **Draft a new
release** page, or by re-running the workflow manually with an existing tag. The tag has to
match the `Cargo.toml` version, or the build stops.

If only some platforms fail, fix the workflow on `main`, then rebuild just those platforms
without retagging. Run **Release** manually from `main` with the existing tag, the failed
platforms and the failed run's ID (the number in its URL) as `reuse-packages-from-run`:

```bash
gh workflow run release.yml --ref main -f tag=v0.1.0 -f platforms=linux-x86_64 \
  -f reuse-packages-from-run=<run-id>
```

## License

Licensed under either of
<a href="LICENSE-APACHE">Apache License Version 2.0</a> or <a href="LICENSE-MIT">MIT license</a>
at your option.

## Contributions

Contributions are welcome.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
