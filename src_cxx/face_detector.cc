#include <iostream>
#include <vector>
#include <string>
#include <optional>
#include <dlib/matrix.h>
#include <dlib/dnn.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>
#include <filesystem>
#include <format>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

#include "once_lock.h"
#include "face_detector.h"

template <long num_filters, typename SUBNET>
using con5d = dlib::con<num_filters, 5, 5, 2, 2, SUBNET>;
template <long num_filters, typename SUBNET>
using con5 = dlib::con<num_filters, 5, 5, 1, 1, SUBNET>;

template <typename SUBNET>
using downsampler = dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<32, dlib::relu<dlib::affine<con5d<16, SUBNET>>>>>>>>>;
template <typename SUBNET>
using rcon5 = dlib::relu<dlib::affine<con5<45, SUBNET>>>;

using cnn_net_type = dlib::loss_mmod<dlib::con<1, 9, 9, 1, 1, rcon5<rcon5<rcon5<downsampler<dlib::input_rgb_image_pyramid<dlib::pyramid_down<6>>>>>>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N, BN, 1, dlib::tag1<SUBNET>>>;

template <template <int, template <typename> class, int, typename> class block, int N, template <typename> class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2, 2, 2, 2, dlib::skip1<dlib::tag2<block<N, BN, 2, dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block = BN<dlib::con<N, 3, 3, 1, 1, dlib::relu<BN<dlib::con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET>
using ares = dlib::relu<residual<block, N, dlib::affine, SUBNET>>;
template <int N, typename SUBNET>
using ares_down = dlib::relu<residual_down<block, N, dlib::affine, SUBNET>>;

template <typename SUBNET>
using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET>
using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET>
using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET>
using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET>
using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using dnn_net_type = dlib::loss_metric<dlib::fc_no_bias<128, dlib::avg_pool_everything<
                                                                 alevel0<
                                                                     alevel1<
                                                                         alevel2<
                                                                             alevel3<
                                                                                 alevel4<
                                                                                     dlib::max_pool<3, 3, 2, 2, dlib::relu<dlib::affine<dlib::con<32, 7, 7, 2, 2, dlib::input_rgb_image_sized<150>>>>>>>>>>>>>;
static OnceLock<std::optional<cnn_net_type>> cnn_net;
static OnceLock<std::optional<dlib::shape_predictor>> shape_predictor;
static OnceLock<std::optional<dnn_net_type>> dnn_net;

////////////////////////////////////////////////////
//
template <typename T>
std::optional<T> get_model(std::string const &model_filename)
{
    if (!std::filesystem::exists(model_filename))
    {
        std::cerr << "Model file not found: " << model_filename << std::endl;
        return std::nullopt;
    }
    T net;
    // Load the model
    dlib::deserialize(model_filename) >> net;
    return net;
}

////////////////////////////////////////////////////
//
std::optional<cnn_net_type> &get_cnn_detector()
{
    return cnn_net.get_or_init(get_model<cnn_net_type>, "mmod_human_face_detector.dat");
}

////////////////////////////////////////////////////
//
std::optional<dlib::shape_predictor> &get_shape_predictor()
{
    return shape_predictor.get_or_init(get_model<dlib::shape_predictor>, "shape_predictor_68_face_landmarks.dat");
}

////////////////////////////////////////////////////
//
std::optional<dnn_net_type> &get_dnn_detector()
{
    return dnn_net.get_or_init(get_model<dnn_net_type>, "dlib_face_recognition_resnet_model_v1.dat");
}

////////////////////////////////////////////////////
//
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
// Recognize faces features
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
std::unique_ptr<std::vector<uint8_t>> find_face_encodings_in_image(std::string const &image_path, std::string const &face_img_path)
{
    auto image_id = std::filesystem::path(image_path).stem().string();
    json detections;
    if (auto img = read_image(image_path))
    {
        auto found_faces = cnn_face_finder(*img, 1);

        for (unsigned long idx = 0; idx < found_faces.size(); idx++)
        {
            auto face_rect = found_faces[idx];
            auto landmarks = landmarks_finder(*img, face_rect);
            if (landmarks.num_parts() == 0)
            {
                std::cout << "No landmarks found in face image." << std::endl;
                continue;
            }
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
bool load_models()
{
    return (std::nullopt != get_cnn_detector() &&
            std::nullopt != get_shape_predictor() &&
            std::nullopt != get_dnn_detector());
}

#if defined(FACE_TAGGER_BUILD_EXE)
////////////////////////////////////////////////////
int main(int, char **)
{
    std::cout << "Hello, World!" << std::endl;
    if (!load_models())
    {
        std::cerr << "Failed to load models." << std::endl;
        return 1;
    }

    // const std::string path = "/Users/chamilton/OneDrive/SmugMug/CaliHamiltonsSmugMugArchive/artifacts/rFKKfFj";
    //  const std::string thumbnail_folder = "./test_folder";
    //  auto json_detections = find_face_encodings_in_image(path, thumbnail_folder);
    //  std::cout << "Detections: " << json::parse(json_detections).dump() << std::endl;

    const std::string path = "/Users/chamilton/OneDrive/SmugMug/CaliHamiltonsSmugMugArchive/artifacts/";
    for (const auto &entry : std::filesystem::directory_iterator(path))
    {
        auto json_detections = find_face_encodings_in_image(entry.path());
        std::cout << "Detections: " << json::parse(json_detections).dump() << std::endl;
    }
    std::cout << "Goodbye, World!" << std::endl;
};
#endif