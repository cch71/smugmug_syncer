# SmugMug Synchronizer

This project is starting out as a tool to backup/mirror SmugMug accounts.

As an offshoot I am hoping to add the ability in the future to take the meta
data and image tags and feed that through AI to generate tags and have that
saved back up into the SmugMug Account

## Building

- Install rust tools: [How to install Rust](https://www.rust-lang.org/tools/install)
- Clone the repo
- cd cloned repo
- run cargo build --release
- run it

```console
➜  git clone https://github.com/cch71/smugmug_syncer syncer
Cloning into 'syncer'...
remote: Enumerating objects: 43, done.
remote: Counting objects: 100% (43/43), done.
remote: Compressing objects: 100% (22/22), done.
remote: Total 43 (delta 23), reused 40 (delta 20), pack-reused 0 (from 0)
Receiving objects: 100% (43/43), 47.61 KiB | 650.00 KiB/s, done.
Resolving deltas: 100% (23/23), done.

➜  cd syncer

➜  cargo build --release
   Compiling proc-macro2 v1.0.94
   Compiling unicode-ident v1.0.18
   ...

➜  target/release/smugmug-syncer --help
```

## Getting started

_If building from source (Binary isn't provided yet) then you need to get an API key from SmugMug_

### To synchronize metadata

```console
smugmug-syncer --syncto ~/Pictures/SmugMugArchive sync --download-artifact-info --nickname apidemo
```

- Create a folder if it doesn't exist at ~/Pictures/SmugMugArchive
- Use the account `apidemo` which equates to https://apidemo.smugmug.com
- Download all Folder/Album and, with the --download-artifact-info, download the images for the given account.
  - this does not download the images/videos themselves just the metadata

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
- `-c` will cause sizes to be calculated

Unless otherwise stated, the queries flags do not require the images/video artifacts to be downloaded.

## CLI Usage

General help:

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

Sync help:

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

Query help

```console
Performs analysis of synced data

Usage: smugmug-syncer query [OPTIONS]

Options:
  -p, --print-tree
          Prints the Folder/Album Tree
  -l, --list-images-from-album-id <LIST_IMAGES_FROM_ALBUM_ID>
          Show images for Album
  -c, --calc-size
          Calculates size synchronized data would take up
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

_A .env file can be created in the working directory that contains these as well._

## License

Licensed under either of <a href="LICENSE-APACHE">Apache License Version
2.0</a> or <a href="LICENSE-MIT">MIT license</a> at your option.

## Contributions

Contributions are welcome.

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in this crate by you, as defined in the Apache-2.0 license, shall
be dual licensed as above, without any additional terms or conditions.
