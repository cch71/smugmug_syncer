/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */
 
#pragma once

#include <vector>
#include <memory>

/*
@brief Find face encodings in an image.
@param image_path Path to the image file.
@param face_img_path Path to the face image file. If this is empty an image will not be saved.
@return A unique pointer to a CBOR encoded vector of uint8_t containing the face encodings.

@node Encoded in the CBOR is an array of detections. If no faces are detected the embeddings
    and face id fields will not be included

[{
    "image_id": "<stem name of image file>",
    "face_id": "<stem name of image file with an _index appended>",
    "embedding": [0.1, 0.2, ...]
}, {
    "image_id": "<stem name of image file>",
    "face_id": "<stem name of image file with an _index appended>",
    "embedding": [0.3, 0.4, ...]
}...]

or if faces are not detected:

[{
    "image_id": "<stem name of image file>"
}]

*/
std::unique_ptr<std::vector<uint8_t>>
find_face_encodings_in_image(std::string const &image_path, std::string const &face_img_path);