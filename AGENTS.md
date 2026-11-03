# AGENTS.md

Guidance for coding agents working on `smugmug_syncer`, a Rust CLI that backs up a SmugMug
account to a local folder. See [README.md](README.md) for usage and
[HOWTO_LABEL_IMAGES.md](HOWTO_LABEL_IMAGES.md) for the face-labeling workflow.

## Rules

### Put SmugMug API access in the `smugmug` crate

Calls to the SmugMug API (endpoints, response types, OAuth1, paging and expansions) belong in
the `smugmug` crate (source in `../smugmug_rs`, published on crates.io), not in this repo. This
repo handles storage, verification, retries and the sync flow. While a crate change is
unpublished, a temporary `path = "../smugmug_rs"` dependency can be used; switch back to the
crates.io version once it's released.

### Reuse shared code; don't reimplement it

Put duplicate functionality in common functions instead of writing it again.

- Before writing a helper, search `src/` for one that already does the job: path building,
  MD5/size checks, retrying downloads, reading and writing metadata, and so on. Use it, or
  extend it if it needs to do more.
- If you find yourself copying a block of code, move it into one shared function and call that
  from both places. Do this in the same change.
- All on-disk locations come from `PathFinder` (`src/path_finder.rs`). Add any new location as
  a constant and getter there; don't build paths inline.

### Where metadata is stored

**Metadata about artifacts** (anything describing individual images or videos, such as
checksums, sizes, alternate renditions or detection results) goes in the `.smugmug_db`
directory in a polars/parquet format.

- Don't write per-artifact sidecar files (for example `artifacts/<id>.json`), and don't create
  new ad-hoc JSON files for artifact metadata.
- Get the file or directory location from `PathFinder`, and read it with the existing polars
  helpers in `src/parquet_store.rs` (`get_dataframe`, `save_dataframe`, `save_in_batches`)
  rather than writing a new reader or writer.
- Keep a column's type the same in every file of a directory (e.g. don't switch `UInt32` to
  `UInt64`). Scanning a directory whose files disagree on a column's type fails.
- Key rows by the SmugMug `ImageKey`, so the data can be joined with the image metadata.

**Metadata about the syncer as a whole** (sync timestamps, run state, settings) can go in
`.smugmug_db/props.json`. Any change must keep the values already in that file:

- Add new fields as `Option<_>` with `#[serde(default)]`, so files written by older versions
  still load.
- Update the file by reading it, changing only the fields you own, and writing it back. Follow
  the read-merge-write pattern used for `Props` in `src/local_folder.rs`, and never write a
  freshly built struct over the file.
- Keep keys the current code doesn't know about. For example, give `Props` a
  `#[serde(flatten)]` catch-all map, so rewriting the file doesn't drop fields added by other
  versions or tools.

## Checks before finishing

- `cargo build --release` and `cargo clippy --release` add no new errors or warnings.
- Changes to reading or writing metadata still load an existing `.smugmug_db` written by the
  previous version.
