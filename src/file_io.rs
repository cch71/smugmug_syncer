/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

//! File helpers shared by artifact downloading and verification

use std::{
    fs::File,
    io::{BufReader, Read},
    path::{Path, PathBuf},
};

use anyhow::Result;

/// Calculates the md5sum and size of a file without reading it all into memory
pub fn file_md5(path: &Path) -> Result<(String, u64)> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut ctx = md5::Context::new();
    let mut buf = vec![0u8; 1024 * 1024];
    let mut size = 0u64;
    loop {
        let n = reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        ctx.consume(&buf[..n]);
        size += n as u64;
    }
    Ok((format!("{:x}", ctx.finalize()), size))
}

/// Calculates the md5sum of data that is already in memory
pub fn bytes_md5(data: &[u8]) -> String {
    format!("{:x}", md5::compute(data))
}

const PARTIAL_FILE_EXT: &str = "part";

// Temp file used while writing the given path, e.g. `KEY.mp4` -> `KEY.mp4.part`
fn partial_path(path: &Path) -> PathBuf {
    let mut tmp_path = path.as_os_str().to_owned();
    tmp_path.push(".");
    tmp_path.push(PARTIAL_FILE_EXT);
    PathBuf::from(tmp_path)
}

/// Writes the data to a temp file then renames it into place, so an interrupted write never
/// leaves a partial file that looks complete
pub fn write_file_atomically(path: &Path, data: &[u8]) -> Result<()> {
    let tmp_path = partial_path(path);
    std::fs::write(&tmp_path, data)?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Removes temp files left in the directory by writes that were interrupted.  Returns how many
/// were removed.  A missing directory has none.
pub fn remove_partial_files(dir: &Path) -> Result<usize> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(0),
        Err(err) => return Err(err.into()),
    };
    let mut removed = 0;
    for entry in entries {
        let path = entry?.path();
        if path.is_file() && path.extension().is_some_and(|v| v == PARTIAL_FILE_EXT) {
            std::fs::remove_file(&path)?;
            removed += 1;
        }
    }
    if removed > 0 {
        log::info!(
            "Removed {} partial files from interrupted writes in: {}",
            removed,
            dir.display()
        );
    }
    Ok(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_writes_use_a_temp_file_per_target() {
        let dir = std::env::temp_dir().join(format!("file_io_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();

        // A poster and its video must not share a temp file
        assert_ne!(
            partial_path(&dir.join("KEY")),
            partial_path(&dir.join("KEY.mp4"))
        );

        write_file_atomically(&dir.join("KEY.mp4"), b"video").unwrap();
        assert_eq!(std::fs::read(dir.join("KEY.mp4")).unwrap(), b"video");
        assert_eq!(
            file_md5(&dir.join("KEY.mp4")).unwrap().0,
            bytes_md5(b"video")
        );

        std::fs::write(dir.join("KEY.part"), b"partial").unwrap();
        std::fs::write(dir.join("x.parquet.part"), b"partial").unwrap();
        assert_eq!(remove_partial_files(&dir).unwrap(), 2);
        assert!(dir.join("KEY.mp4").exists());
        assert_eq!(remove_partial_files(&dir.join("missing")).unwrap(), 0);

        std::fs::remove_dir_all(dir).unwrap();
    }
}
