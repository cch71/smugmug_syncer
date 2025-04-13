/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use anyhow::Result;
use cxx::let_cxx_string;
use serde::{Deserialize, Serialize};

#[cxx::bridge]
mod ffi {

    unsafe extern "C++" {
        include!("smugmug_syncer/src_cxx/face_detector.h");

        fn find_face_encodings_in_image(
            image_path: &CxxString,
            face_img_path: &CxxString,
        ) -> Result<UniquePtr<CxxVector<u8>>>;
    }
}

#[derive(Deserialize, Serialize, Debug)]
pub struct FaceRect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

#[derive(Deserialize, Serialize, Debug, Default)]
pub struct FaceDetection {
    pub embeddings: Option<Vec<f64>>,
    pub image_id: String,
    pub face_id: Option<String>,
    pub rect_in_image: Option<FaceRect>,
}

// find faces and return their encodings as well as optionally saving a thumbnail of any found faces
pub(crate) fn find_faces(
    image_path: &str,
    face_image_path: Option<&str>,
) -> Result<Vec<FaceDetection>> {
    let_cxx_string!(image_path_cxxstr = image_path);
    let_cxx_string!(face_image_path_cxxstr = face_image_path.unwrap_or_default());
    let detection = ffi::find_face_encodings_in_image(&image_path_cxxstr, &face_image_path_cxxstr)?;
    let detection: Vec<FaceDetection> = ciborium::from_reader(detection.as_slice())?;
    Ok(detection)
}
