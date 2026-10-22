use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    if env::var_os("CARGO_FEATURE_LLAMA").is_none() {
        // The default workspace build does not need a local llama.cpp tree.
        return;
    }
    println!("cargo:rerun-if-changed=src/shim.cpp");
    println!("cargo:rerun-if-changed=src/rustmoe_llama.h");
    println!("cargo:rerun-if-env-changed=LLAMA_CPP_DIR");
    println!("cargo:rerun-if-env-changed=LLAMA_BUILD_DIR");

    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest.parent().unwrap().parent().unwrap().to_path_buf();
    let cpp = env::var("LLAMA_CPP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace.join(".models/llama.cpp"));
    let build = env::var("LLAMA_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| workspace.join(".models/llama-build"));

    let include = cpp.join("include");
    let ggml_include = cpp.join("ggml/include");
    let header = include.join("llama.h");
    if !header.exists() {
        panic!(
            "llama.cpp headers not found at {}. Set LLAMA_CPP_DIR or run scripts/fetch-reference.ps1",
            header.display()
        );
    }

    let mut cc = cc::Build::new();
    cc.cpp(true)
        .std("c++17")
        .include(&include)
        .include(&ggml_include)
        .file("src/shim.cpp");
    if cfg!(target_env = "msvc") {
        cc.flag_if_supported("/EHsc");
        cc.flag_if_supported("/std:c++17");
    }
    cc.compile("rustmoe_llama_shim");

    let mut lib_dirs = Vec::new();
    for rel in [
        "src/Release",
        "src",
        "src/Debug",
        "ggml/src/Release",
        "ggml/src",
        "ggml/src/Debug",
    ] {
        let p = build.join(rel);
        if p.exists() {
            lib_dirs.push(p);
        }
    }
    for dir in &lib_dirs {
        println!("cargo:rustc-link-search=native={}", dir.display());
    }
    for lib in ["llama", "ggml", "ggml-base", "ggml-cpu"] {
        println!("cargo:rustc-link-lib={lib}");
    }

    if cfg!(target_env = "msvc") {
        let bin = build.join("bin/Release");
        if bin.exists() {
            let target = workspace.join("target");
            let profile = env::var("PROFILE").unwrap_or_else(|_| "debug".into());
            let mut dests = vec![target.join(&profile)];
            let deps = target.join(&profile).join("deps");
            if deps.exists() {
                dests.push(deps);
            }
            let cuda = env::var("CUDA_PATH")
                .ok()
                .map(PathBuf::from)
                .filter(|p| p.join("bin/cudart64_12.dll").exists())
                .or_else(|| {
                    let p = PathBuf::from(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8");
                    if p.join("bin/cudart64_12.dll").exists() {
                        Some(p)
                    } else {
                        None
                    }
                });
            let mut names = vec![
                "llama.dll".to_string(),
                "ggml.dll".to_string(),
                "ggml-base.dll".to_string(),
                "ggml-cpu.dll".to_string(),
                "ggml-cuda.dll".to_string(),
            ];
            if let Some(cuda) = &cuda {
                if let Ok(entries) = fs::read_dir(cuda.join("bin")) {
                    for e in entries.flatten() {
                        let name = e.file_name().to_string_lossy().to_string();
                        if name == "cudart64_12.dll"
                            || name == "cublas64_12.dll"
                            || name == "cublasLt64_12.dll"
                            || name.starts_with("nvJitLink_120")
                        {
                            names.push(name);
                        }
                    }
                }
            }
            for dest in dests {
                if !dest.exists() {
                    continue;
                }
                for name in &names {
                    let src = if name.starts_with("cudart")
                        || name.starts_with("cublas")
                        || name.starts_with("nvJitLink")
                    {
                        cuda.as_ref().map(|c| c.join("bin").join(name))
                    } else {
                        Some(bin.join(name))
                    };
                    if let Some(src) = src {
                        if src.exists() {
                            let _ = fs::copy(&src, dest.join(name));
                        }
                    }
                }
            }
        }
    }
}
