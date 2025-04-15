fn main() {
    let dlib = pkg_config::probe_library("dlib-1").unwrap();
    let nl_json = pkg_config::probe_library("nlohmann_json").unwrap();
    let cv = pkg_config::probe_library("opencv4").unwrap();

    cxx_build::bridge("src/face_detector.rs")
        .file("src_cxx/face_detector.cc")
        .std("c++20")
        .flag("-Wno-psabi")
        .includes(cv.include_paths)
        .includes(dlib.include_paths)
        .includes(nl_json.include_paths)
        .compile("face_detector");

    #[cfg(target_os = "linux")]
    {
        let blas = pkg_config::probe_library("blas").unwrap();
        let lapack = pkg_config::probe_library("lapack").unwrap();
        blas.libs
            .iter()
            .for_each(|lib| println!("cargo:rustc-link-lib={}", lib));
        lapack
            .libs
            .iter()
            .for_each(|lib| println!("cargo:rustc-link-lib={}", lib));
        nl_json
            .libs
            .iter()
            .for_each(|lib| println!("cargo:rustc-link-lib={}", lib));
        cv.libs
            .iter()
            .for_each(|lib| println!("cargo:rustc-link-lib={}", lib));
        dlib.libs
            .iter()
            .for_each(|lib| println!("cargo:rustc-link-lib={}", lib));
    }

    println!("cargo:rerun-if-changed=src/tagging.rs");
    println!("cargo:rerun-if-changed=src_cxx/face_detector.cc");
    println!("cargo:rerun-if-changed=src_cxx/face_detector.h");
    println!("cargo:rerun-if-changed=src_cxx/once_lock.h");
    println!("cargo:rerun-if-changed=src_cxx/dlib_types.h");
}
