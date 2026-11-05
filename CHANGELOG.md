# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). `cargo release` moves the Unreleased section under
the new version number.

Changes to the `.smugmug_db` layout also bump the database version (`DB_VERSION` in
`src/local_folder.rs`, stored as `db_version` in `.smugmug_db/props.json`). Note the new
database version in the entry.

## [Unreleased]

## [1.0.0] - 2026-11-05

### Added

- `db_version` in `.smugmug_db/props.json` records the semver version of the archive's
  database layout (currently `1.0.0`). Archives without it are treated as `1.0.0` and stamped
  the next time they are opened. The syncer refuses to write to an archive whose major
  database version differs from its own, and never lowers a newer minor version.
- `smugmug-syncer --version` prints the version; `--version` (long form) also prints the
  database version.
- Linux and macOS release binaries on GitHub, built by the release workflow.
- CI (fmt, clippy, tests), CodeQL, `cargo deny` and Dependabot.

[Unreleased]: https://github.com/cch71/smugmug_syncer/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/cch71/smugmug_syncer/releases/tag/v1.0.0
