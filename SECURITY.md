# Security policy

## Supported versions

Only the latest release gets security fixes.

## Reporting a vulnerability

Please don't open a public issue. Report it privately through
[GitHub's private vulnerability reporting](https://github.com/cch71/smugmug_syncer/security/advisories/new).
Include the version, your platform, and steps to reproduce.

You should get a reply within a week. Once a fix is released, the advisory is published with
credit to you, unless you'd rather stay anonymous.

## What's in scope

The syncer handles SmugMug OAuth tokens and writes to the local archive. Issues such as
leaking tokens from `SMUGMUG_AUTH_CACHE` or logs, writing outside the `--syncto` directory
(for example through crafted image keys or file names from the API), or crashes from crafted
`.smugmug_db` files are all in scope.

## Automated checks

Every push and pull request runs:

- CodeQL analysis of the Rust code and the GitHub workflows (`.github/workflows/codeql.yml`)
- `cargo deny`, which checks for RustSec advisories, disallowed licenses, and dependencies
  from unknown sources (`deny.toml`)
- Dependency review on pull requests
- OpenSSF Scorecard on `main`

Dependabot proposes dependency and action updates weekly.
