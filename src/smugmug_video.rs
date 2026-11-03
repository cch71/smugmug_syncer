/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

//! Local record of the video downloaded for each video artifact.  For videos an Image's
//! ArchivedUri only points at a JPEG poster frame; the video itself comes from
//! [`smugmug::v2::Image::largest_video_with_client`].

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
};

use anyhow::Result;
use polars::prelude::*;
use smugmug::v2::LargestVideo;

use crate::parquet_store::{get_dataframe, has_parquet_files, save_dataframe};

/// What SmugMug reported about the downloaded video, used to verify it later
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub size: Option<u64>,
    pub md5: Option<String>,
    pub ext: String,
    pub width: Option<u64>,
    pub height: Option<u64>,
    pub duration: Option<String>,
}

impl From<&LargestVideo> for VideoInfo {
    fn from(video: &LargestVideo) -> Self {
        Self {
            size: video.size,
            md5: video.md5.clone(),
            ext: video.ext.clone(),
            width: video.width,
            height: video.height,
            duration: video.duration.clone(),
        }
    }
}

impl VideoInfo {
    /// Returns true if the given md5sum and size match this video.  Values SmugMug didn't
    /// report aren't checked.
    pub fn matches(&self, md5sum: &str, size: u64) -> bool {
        self.size.is_none_or(|v| v == size) && self.md5.as_ref().is_none_or(|v| v == md5sum)
    }

    /// Path where the video for the given image key is stored
    pub fn video_path(&self, artifacts_dir: &Path, image_key: &str) -> PathBuf {
        artifacts_dir.join(format!("{}.{}", image_key, self.ext))
    }

    /// Loads the saved video information, keyed by image key
    pub fn load_all(video_info_dir: &Path) -> Result<HashMap<String, Self>> {
        if !has_parquet_files(video_info_dir) {
            return Ok(HashMap::new());
        }

        let df = get_dataframe(
            video_info_dir,
            [
                col("image_key"),
                col("size"),
                col("md5"),
                col("ext"),
                col("width").cast(DataType::UInt64),
                col("height").cast(DataType::UInt64),
                col("duration"),
            ],
        )?
        .collect()?;

        let keys = df.column("image_key")?.str()?;
        let sizes = df.column("size")?.u64()?;
        let md5s = df.column("md5")?.str()?;
        let exts = df.column("ext")?.str()?;
        let widths = df.column("width")?.u64()?;
        let heights = df.column("height")?.u64()?;
        let durations = df.column("duration")?.str()?;

        let mut infos = HashMap::new();
        for i in 0..df.height() {
            let (Some(key), Some(ext)) = (keys.get(i), exts.get(i)) else {
                continue;
            };
            infos.insert(
                key.to_string(),
                Self {
                    size: sizes.get(i),
                    md5: md5s.get(i).map(str::to_string),
                    ext: ext.to_string(),
                    width: widths.get(i),
                    height: heights.get(i),
                    duration: durations.get(i).map(str::to_string),
                },
            );
        }
        Ok(infos)
    }

    /// Saves video information for the given image keys as a new parquet file
    pub fn save_all(video_info_dir: &Path, infos: &[(String, Self)]) -> Result<()> {
        let mut df = df![
            "image_key" => infos.iter().map(|(k, _)| k.as_str()).collect::<Vec<_>>(),
            "size" => infos.iter().map(|(_, v)| v.size).collect::<Vec<_>>(),
            "md5" => infos.iter().map(|(_, v)| v.md5.as_deref()).collect::<Vec<_>>(),
            "ext" => infos.iter().map(|(_, v)| v.ext.as_str()).collect::<Vec<_>>(),
            // Stored as UInt32 so every file in the directory has the same schema
            "width" => infos.iter().map(|(_, v)| v.width.and_then(|v| u32::try_from(v).ok())).collect::<Vec<_>>(),
            "height" => infos.iter().map(|(_, v)| v.height.and_then(|v| u32::try_from(v).ok())).collect::<Vec<_>>(),
            "duration" => infos.iter().map(|(_, v)| v.duration.as_deref()).collect::<Vec<_>>(),
        ]?;
        save_dataframe(&mut df, video_info_dir, "video_info")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn saved_video_info_is_loaded_back() {
        let dir = std::env::temp_dir().join(format!("video_info_{}", uuid::Uuid::new_v4()));
        assert!(VideoInfo::load_all(&dir).unwrap().is_empty());

        let info = VideoInfo {
            size: Some(172451),
            md5: Some("bb4b31597923f09dd1bdc6758bf38986".to_string()),
            ext: "mp4".to_string(),
            width: Some(320),
            height: None,
            duration: Some("5".to_string()),
        };
        VideoInfo::save_all(&dir, &[("a".to_string(), info.clone())]).unwrap();

        let loaded = VideoInfo::load_all(&dir).unwrap();
        let a = &loaded["a"];
        assert_eq!(
            (a.size, a.md5.as_deref(), a.ext.as_str(), a.width, a.height),
            (Some(172451), info.md5.as_deref(), "mp4", Some(320), None)
        );
        assert!(a.matches("bb4b31597923f09dd1bdc6758bf38986", 172451));
        assert!(!a.matches("bb4b31597923f09dd1bdc6758bf38986", 1));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
