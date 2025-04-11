/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */
 
#pragma once

#include <vector>

bool load_models();

std::unique_ptr<std::vector<uint8_t>>
find_face_encodings_in_image(std::string const &image_path, std::string const &face_img_path);