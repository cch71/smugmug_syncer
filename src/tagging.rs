/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

use crate::TaggingArgs;
use anyhow::Result;
use cxx::let_cxx_string;
use serde::{Deserialize, Serialize};

#[cxx::bridge]
mod ffi {

    unsafe extern "C++" {
        include!("smugmug_syncer/src_cxx/face_detector.h");

        fn load_models() -> bool;

        fn find_face_encodings_in_image(
            image_path: &CxxString,
            face_img_path: &CxxString,
        ) -> Result<UniquePtr<CxxVector<u8>>>;
    }
}

#[derive(Deserialize, Serialize, Debug)]
struct FaceRect {
    x: i32,
    y: i32,
    w: i32,
    h: i32,
}

#[derive(Deserialize, Serialize, Debug)]
struct FaceDetection {
    embeddings: Vec<f64>,
    image_id: String,
    face_id: String,
    rect_in_image: FaceRect,
}

fn find_faces(image_path: &str, face_image_path: Option<&str>) -> Result<Vec<FaceDetection>> {
    let_cxx_string!(image_path_cxxstr = image_path);
    let_cxx_string!(face_image_path_cxxstr = face_image_path.unwrap_or_default());
    let detection = ffi::find_face_encodings_in_image(&image_path_cxxstr, &face_image_path_cxxstr)?;
    let detection: Vec<FaceDetection> = ciborium::from_reader(detection.as_slice())?;
    Ok(detection)
}

pub(crate) async fn handle_tagging_req(_path: &str, _args: TaggingArgs) -> Result<()> {
    if !ffi::load_models() {
        return Err(anyhow::anyhow!("Failed to load models"));
    }

    let image_path =
        "/Users/chamilton/OneDrive/SmugMug/CaliHamiltonsSmugMugArchive/artifacts/rFKKfFj";
    let detections = find_faces(image_path, None)?;
    println!("Detection result: {:#?}", detections);
    Ok(())
}
