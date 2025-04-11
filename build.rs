fn main() {
    let dlib = pkg_config::probe_library("dlib-1").unwrap();
    let no_json = pkg_config::probe_library("nlohmann_json").unwrap();
    let cv = pkg_config::probe_library("opencv4").unwrap();

    cxx_build::bridge("src/tagging.rs")
        .file("src_cxx/face_detector.cc")
        .std("c++20")
        .includes(cv.include_paths)
        .includes(dlib.include_paths)
        .includes(no_json.include_paths)
        .compile("face_detector");

    println!("cargo:rerun-if-changed=src/tagging.rs");
    println!("cargo:rerun-if-changed=src_cxx/face_detector.cc");
    println!("cargo:rerun-if-changed=src_cxx/face_detector.h");
}
