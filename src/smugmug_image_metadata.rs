/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

//! Local record of the EXIF/IPTC metadata SmugMug holds for each artifact.  SmugMug keeps this
//! even when the file it serves for download has had its EXIF stripped.

use std::{collections::HashSet, path::Path};

use anyhow::Result;
use polars::prelude::*;
use smugmug::v2::{Image, ImageMetadata};

use crate::parquet_store::{get_dataframe, has_parquet_files, save_dataframe};

/// One artifact's metadata.  Commonly used fields get their own column and every field SmugMug
/// returned is kept in `metadata_json`, which is empty when SmugMug had no metadata for it.
#[derive(Debug, Clone)]
pub struct ImageMetadataRecord {
    pub image_key: String,
    pub date_time_original: Option<String>,
    pub serial: Option<u64>,
    pub original_width: Option<u64>,
    pub original_height: Option<u64>,
    pub original_size: Option<u64>,
    pub make: Option<String>,
    pub model: Option<String>,
    pub date_digitized: Option<String>,
    pub latitude: Option<f64>,
    pub longitude: Option<f64>,
    pub metadata_json: Option<String>,
}

impl ImageMetadataRecord {
    /// Builds the record from an image and the metadata SmugMug returned for it, if any
    pub fn new(image: &Image, metadata: Option<&ImageMetadata>) -> Result<Self> {
        let Some(metadata) = metadata else {
            return Ok(Self {
                image_key: image.image_key.clone(),
                date_time_original: image.date_time_original.map(|v| v.to_rfc3339()),
                serial: image.serial,
                original_width: image.original_width,
                original_height: image.original_height,
                original_size: image.original_size,
                make: None,
                model: None,
                date_digitized: None,
                latitude: None,
                longitude: None,
                metadata_json: None,
            });
        };

        // SmugMug reports 0/0 when there is no location
        let (latitude, longitude) = if metadata.has_location() {
            (metadata.latitude(), metadata.longitude())
        } else {
            (None, None)
        };
        Ok(Self {
            image_key: image.image_key.clone(),
            date_time_original: image.date_time_original.map(|v| v.to_rfc3339()),
            serial: image.serial,
            original_width: image.original_width,
            original_height: image.original_height,
            original_size: image.original_size,
            make: metadata.make().map(str::to_string),
            model: metadata.model().map(str::to_string),
            date_digitized: metadata.date_digitized().map(str::to_string),
            latitude,
            longitude,
            metadata_json: Some(serde_json::to_string(metadata.fields())?),
        })
    }

    /// Image keys that already have metadata saved
    pub fn load_recorded_keys(image_metadata_dir: &Path) -> Result<HashSet<String>> {
        if !has_parquet_files(image_metadata_dir) {
            return Ok(HashSet::new());
        }
        let df = get_dataframe(image_metadata_dir, [col("image_key")])?.collect()?;
        Ok(df
            .column("image_key")?
            .str()?
            .iter()
            .flatten()
            .map(str::to_string)
            .collect())
    }

    /// Saves the records as a new parquet file
    pub fn save_all(image_metadata_dir: &Path, records: &[Self]) -> Result<()> {
        let mut df = df![
            "image_key" => records.iter().map(|v| v.image_key.as_str()).collect::<Vec<_>>(),
            "date_time_original" => records.iter().map(|v| v.date_time_original.as_deref()).collect::<Vec<_>>(),
            "serial" => records.iter().map(|v| v.serial).collect::<Vec<_>>(),
            "original_width" => records.iter().map(|v| v.original_width).collect::<Vec<_>>(),
            "original_height" => records.iter().map(|v| v.original_height).collect::<Vec<_>>(),
            "original_size" => records.iter().map(|v| v.original_size).collect::<Vec<_>>(),
            "make" => records.iter().map(|v| v.make.as_deref()).collect::<Vec<_>>(),
            "model" => records.iter().map(|v| v.model.as_deref()).collect::<Vec<_>>(),
            "date_digitized" => records.iter().map(|v| v.date_digitized.as_deref()).collect::<Vec<_>>(),
            "latitude" => records.iter().map(|v| v.latitude).collect::<Vec<_>>(),
            "longitude" => records.iter().map(|v| v.longitude).collect::<Vec<_>>(),
            "metadata_json" => records.iter().map(|v| v.metadata_json.as_deref()).collect::<Vec<_>>(),
        ]?;
        save_dataframe(&mut df, image_metadata_dir, "image_metadata")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(image_key: &str) -> ImageMetadataRecord {
        ImageMetadataRecord {
            image_key: image_key.to_string(),
            date_time_original: Some("2012-01-22T05:45:54+00:00".to_string()),
            serial: Some(1),
            original_width: Some(3264),
            original_height: Some(2448),
            original_size: Some(2261412),
            make: Some("Samsung".to_string()),
            model: None,
            date_digitized: None,
            latitude: Some(30.5),
            longitude: None,
            metadata_json: Some(r#"{"Make":"Samsung"}"#.to_string()),
        }
    }

    #[test]
    fn saved_records_are_loaded_back() {
        let dir = std::env::temp_dir().join(format!("image_metadata_{}", uuid::Uuid::new_v4()));
        assert!(
            ImageMetadataRecord::load_recorded_keys(&dir)
                .unwrap()
                .is_empty()
        );

        ImageMetadataRecord::save_all(&dir, &[record("a"), record("b")]).unwrap();
        ImageMetadataRecord::save_all(&dir, &[record("c")]).unwrap();

        let keys = ImageMetadataRecord::load_recorded_keys(&dir).unwrap();
        assert_eq!(keys, HashSet::from(["a", "b", "c"].map(str::to_string)));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
