/*
 * Copyright (c) 2025 Craig Hamilton and Contributors.
 * Licensed under either of
 *  - Apache License, Version 2.0 <http://www.apache.org/licenses/LICENSE-2.0> OR
 *  - MIT license <http://opensource.org/licenses/MIT>
 *  at your option.
 */

#include "face_detector.h"
#include "once_lock.h"
#include "dlib_types.h"

#include <iostream>
#include <vector>
#include <string>
#include <optional>
#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <filesystem>
#include <format>
#include <nlohmann/json.hpp>
using json = nlohmann::json;



////////////////////////////////////////////////////
// Generic loader of the model data into dlib net structures
template <typename T>
std::optional<T> get_model(std::string const &model_filename)
{
    auto get_model_filepath = [&model_filename]() -> std::optional<std::filesystem::path> {
        auto env_model_dir = std::getenv("SMUGMUG_SYNC_MODELS_DIR");
        if (env_model_dir != nullptr) {
            auto model_path = std::filesystem::path(env_model_dir) / model_filename;
            if (std::filesystem::exists(model_path)) {
                return model_path;
            }
        }
        if (std::filesystem::exists(model_filename)) {
            return std::filesystem::path(model_filename);
        }
        return std::nullopt;
    };

    const auto model_filepath = get_model_filepath();
    if (!model_filepath)
    {
        std::cerr << "Model file not found: " << model_filename << std::endl;
        return std::nullopt;
    }
    T net;
    // Load the model
    dlib::deserialize((*model_filepath).c_str()) >> net;
    return net;
}

// NOT THREAD SAFE
thread_local auto cnn_net = get_model<cnn_net_type>("mmod_human_face_detector.dat");
thread_local auto shape_predictor = get_model<dlib::shape_predictor>("shape_predictor_68_face_landmarks.dat");
thread_local auto dnn_net = get_model<dnn_net_type>("dlib_face_recognition_resnet_model_v1.dat");

////////////////////////////////////////////////////
// Loads the CNN face detector model
// This model is used to detect faces in images.
// It is a single-shot multi-box detector that uses a CNN to detect faces in images.
std::optional<cnn_net_type> &get_cnn_detector()
{
    return cnn_net;
}

////////////////////////////////////////////////////
// Loads the shape predictor model
// This model is used to detect facial landmarks in images.
std::optional<dlib::shape_predictor> &get_shape_predictor()
{
    return shape_predictor;
}

////////////////////////////////////////////////////
// Loads the DNN face recognition model.
// This model is used to extract facial features from images.
// It is a ResNet model that is trained to recognize faces in images.
std::optional<dnn_net_type> &get_dnn_detector()
{
    return dnn_net;
}

////////////////////////////////////////////////////
// This function uses the CNN face detector to detect faces in an image.
std::vector<dlib::mmod_rect> cnn_face_finder(
    dlib::matrix<dlib::rgb_pixel> image,
    const int upsample_num_times)
{
    dlib::pyramid_down<2> pyr;
    std::vector<dlib::mmod_rect> rects;

    // Up-sampling the image will allow us to detect smaller faces but will cause the
    // program to use more RAM and run longer.
    unsigned int levels = upsample_num_times;
    while (levels > 0)
    {
        levels--;
        dlib::pyramid_up(image, pyr);
    }
    auto &net = *get_cnn_detector();
    auto dets = net(image);

    // Scale the detection locations back to the original image size
    // if the image was upscaled.
    for (auto &&d : dets)
    {
        d.rect = pyr.rect_down(d.rect, upsample_num_times);
        rects.push_back(d);
    }

    return rects;
}

////////////////////////////////////////////////////
// Detect landmarks
dlib::full_object_detection landmarks_finder(
    dlib::matrix<dlib::rgb_pixel> image,
    const dlib::rectangle &face_rect)
{
    // Detect landmarks
    auto &net = *get_shape_predictor();
    return net(image, face_rect);
}

////////////////////////////////////////////////////
// Recognize facial features and encodes them returns the rectangle of the face as well
// as the facial features embeddings
std::tuple<dlib::matrix<double, 0, 1>, dlib::matrix<dlib::rgb_pixel>> facial_features_finder(
    dlib::matrix<dlib::rgb_pixel> image,
    const dlib::full_object_detection &face,
    float padding = 0.25)
{
    dlib::matrix<dlib::rgb_pixel> face_chip;
    dlib::extract_image_chip(image, dlib::get_face_chip_details(face, 150, padding), face_chip);

    // extract descriptors and convert from float vectors to double vectors
    auto &net = *get_dnn_detector();
    auto descriptors = net(std::vector<dlib::matrix<dlib::rgb_pixel>>{face_chip}, 16);

    return std::make_tuple(dlib::matrix_cast<double>(descriptors[0]), face_chip);
}

////////////////////////////////////////////////////
// Read image in
std::optional<dlib::matrix<dlib::rgb_pixel>> read_image(const std::string &filename)
{
    // Load image using OpenCV
    cv::Mat image = cv::imread(filename);
    if (image.empty())
    {
        std::cerr << "Error loading image: " << filename << std::endl;
        return std::nullopt;
    }

    // Resize image
    int width = 600;
    float aspectRatio = static_cast<float>(width) / image.cols;
    int height = static_cast<int>(image.rows * aspectRatio);
    cv::resize(image, image, cv::Size(width, height));

    // Convert from BGR to RGB
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    // Convert to dlib image
    dlib::cv_image<dlib::rgb_pixel> dlib_image(rgb);
    dlib::matrix<dlib::rgb_pixel> dlib_matrix;
    dlib::assign_image(dlib_matrix, dlib_image);
    return dlib_matrix;
}

////////////////////////////////////////////////////
// Find face encodings in image
// This function takes an image path and a face image path as input.
// It detects faces in the image, extracts facial features, and saves the face images to the specified path.
// It returns a CBOR object containing the face encodings and other information.
std::unique_ptr<std::vector<uint8_t>> find_face_encodings_in_image(std::string const &image_path, std::string const &face_img_path)
{
    auto image_id = std::filesystem::path(image_path).stem().string();
    json detections;

    // std::cout << "Reading image: " << image_path << std::endl;
    if (auto img = read_image(image_path))
    {
        //std::cout << "Starting facial detection " << image_path << std::endl;
       
        auto found_faces = cnn_face_finder(*img, 1);
        // Iterate over the detected faces
        for (unsigned long idx = 0; idx < found_faces.size(); idx++)
        {
            auto face_rect = found_faces[idx];

            // Get the facial landmarks
            auto landmarks = landmarks_finder(*img, face_rect);
            if (landmarks.num_parts() == 0)
            {
                std::cout << "No landmarks found in face image." << image_path << std::endl;
                continue;
            }

            // Get the facial features and rect of the face
            auto &&[embeddings, thumbnail] = facial_features_finder(*img, landmarks);
            detections.push_back({
                {"rect_in_image", {{"x", face_rect.rect.left()}, {"y", face_rect.rect.top()}, {"w", face_rect.rect.width()}, {"h", face_rect.rect.height()}}},
                {"image_id", image_id},
                {
                    "face_id",
                    std::format("{}_{}", image_id, idx),
                },
                {"embeddings", embeddings},
            });

            // Save the face image if a path is provided
            if (!face_img_path.empty())
            {

                auto face_img_dir = std::filesystem::path(face_img_path);
                std::filesystem::create_directories(face_img_dir);

                auto face_thumbnail_filename = face_img_dir / std::format("{}_{}.jpg", image_id, idx);
                bool did_save = false;
                try
                {
                    cv::Mat thumbnail_rgb = dlib::toMat(thumbnail);

                    cv::Mat thumbnail_bgr;
                    cv::cvtColor(thumbnail_rgb, thumbnail_bgr, cv::COLOR_RGB2BGR);

                    did_save = cv::imwrite(face_thumbnail_filename.c_str(), thumbnail_bgr);
                }
                catch (const cv::Exception &ex)
                {
                    std::cerr << "Failed saving thumbnail to: " << face_thumbnail_filename.c_str() << " reason: " << ex.what() << std::endl;
                }
                if (!did_save)
                {
                    std::cerr << "Failed saving thumbnail to: " << face_thumbnail_filename.c_str() << std::endl;
                }
            }
        }
    }

    if (detections.empty())
    {
        detections.push_back({
            {"image_id", image_id},
        });
    }
    return std::make_unique<std::vector<uint8_t>>(json::to_cbor(detections));
}

////////////////////////////////////////////////////
// Load models
bool load_models()
{
    // return (std::nullopt != get_cnn_detector() &&
    //         std::nullopt != get_shape_predictor() &&
    //         std::nullopt != get_dnn_detector());
    return true;
}

#if defined(FACE_TAGGER_BUILD_EXE)
////////////////////////////////////////////////////
// Main function
int main(int, char **)
{
    std::cout << "Hello, World!" << std::endl;
    if (!load_models())
    {
        std::cerr << "Failed to load models." << std::endl;
        return 1;
    }

    const std::string path = argv[1];
    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        auto json_detections = find_face_encodings_in_image(entry.path());
        std::cout << "Detections: " << json::from_cbor(json_detections.get()).dump() << std::endl;
    }
    std::cout << "Goodbye, World!" << std::endl;
};
#endif