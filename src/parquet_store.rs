/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

//! Reading and writing DataFrames stored as a directory of parquet files

use std::path::{Path, PathBuf};

use anyhow::Result;
use polars::prelude::*;
use uuid::Uuid;

use crate::file_io::write_file_atomically;

/// Returns true if the given directory contains any parquet files
pub(crate) fn has_parquet_files(root_path: &Path) -> bool {
    std::fs::read_dir(root_path).is_ok_and(|mut entries| {
        entries.any(|v| v.is_ok_and(|v| v.path().extension().is_some_and(|ext| ext == "parquet")))
    })
}

/// Reads the parquet files in the given directory and returns a DataFrame
pub(crate) fn get_dataframe<E: AsRef<[Expr]>>(root_path: &Path, exprs: E) -> Result<LazyFrame> {
    let parquet_glob = root_path.join("*.parquet");
    let lf = LazyFrame::scan_parquet(
        PlRefPath::try_from_pathbuf(parquet_glob)?,
        Default::default(),
    )?;
    Ok(lf.select(exprs))
}

/// Saves the DataFrame as a new parquet file in the given directory, creating it if needed.
/// Each save adds a file so earlier data is never rewritten.  The file is written atomically,
/// since a partly written parquet file would make every later scan of the directory fail.
pub(crate) fn save_dataframe(
    df: &mut DataFrame,
    root_path: &Path,
    prefix: &str,
) -> Result<PathBuf> {
    std::fs::create_dir_all(root_path)?;

    let file_id = Uuid::new_v4().simple().to_string();
    let file_path = root_path.join(format!("{}_{}.parquet", prefix, file_id));

    log::debug!("Saving: {} at: {}", df, file_path.display());

    let mut data = Vec::new();
    ParquetWriter::new(&mut data)
        .with_compression(ParquetCompression::Zstd(None))
        .finish(df)?;
    write_file_atomically(&file_path, &data)?;

    Ok(file_path)
}

/// Receives records until the channel closes, saving them in batches of `batch_size` so an
/// interrupted run keeps what was already received
pub(crate) async fn save_in_batches<T>(
    mut rx: tokio::sync::mpsc::UnboundedReceiver<T>,
    batch_size: usize,
    mut save: impl FnMut(&[T]) -> Result<()>,
) -> Result<()> {
    let mut batch = Vec::new();
    while let Some(record) = rx.recv().await {
        batch.push(record);
        if batch.len() >= batch_size {
            save(&batch)?;
            batch.clear();
        }
    }
    if !batch.is_empty() {
        save(&batch)?;
    }
    Ok(())
}
