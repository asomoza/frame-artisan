use std::path::PathBuf;

/// Detected GPU type for PyTorch index selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuType {
    NvidiaCuda,
    AmdRocm,
    Cpu,
}

impl std::fmt::Display for GpuType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuType::NvidiaCuda => write!(f, "NVIDIA (CUDA)"),
            GpuType::AmdRocm => write!(f, "AMD (ROCm)"),
            GpuType::Cpu => write!(f, "CPU only"),
        }
    }
}

/// Returns the PyTorch index URL for the given GPU type.
pub fn torch_index_url(gpu: GpuType) -> &'static str {
    match gpu {
        GpuType::NvidiaCuda => "https://download.pytorch.org/whl/cu130",
        GpuType::AmdRocm => "https://download.pytorch.org/whl/rocm6.3",
        GpuType::Cpu => "https://download.pytorch.org/whl/cpu",
    }
}

/// PyTorch version to install.
const TORCH_VERSION: &str = "2.10.0";

/// Returns the torch wheel URL for the given GPU type and current platform.
/// The wheel filename encodes: torch-{version}+{variant}-cp{pyver}-cp{pyver}-{platform}.whl
pub fn torch_wheel_url(gpu: GpuType) -> String {
    let variant = match gpu {
        GpuType::NvidiaCuda => "cu130",
        GpuType::AmdRocm => "rocm6.4",
        GpuType::Cpu => "cpu",
    };

    let pyver = PYTHON_VERSION.replace('.', ""); // "3.12" -> "312"
    let platform_tag = platform_wheel_tag();

    let index_path = match gpu {
        GpuType::NvidiaCuda => "cu130",
        GpuType::AmdRocm => "rocm6.4",
        GpuType::Cpu => "cpu",
    };

    format!(
        "https://download.pytorch.org/whl/{index_path}/torch-{TORCH_VERSION}%2B{variant}-cp{pyver}-cp{pyver}-{platform_tag}.whl"
    )
}

/// Returns the platform wheel tag for the current OS/arch.
fn platform_wheel_tag() -> &'static str {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    { "manylinux_2_28_x86_64" }

    #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
    { "manylinux_2_28_aarch64" }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    { "macosx_14_0_x86_64" }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    { "macosx_14_0_arm64" }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    { "win_amd64" }
}

/// uv binary download URL per platform.
pub fn uv_download_url() -> &'static str {
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    { "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-unknown-linux-gnu.tar.gz" }

    #[cfg(all(target_os = "macos", target_arch = "x86_64"))]
    { "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-apple-darwin.tar.gz" }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    { "https://github.com/astral-sh/uv/releases/latest/download/uv-aarch64-apple-darwin.tar.gz" }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    { "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" }
}

/// uv binary name inside the archive.
pub fn uv_binary_name() -> &'static str {
    #[cfg(target_os = "windows")]
    { "uv.exe" }

    #[cfg(not(target_os = "windows"))]
    { "uv" }
}

/// Default installation directory.
pub fn default_install_dir() -> PathBuf {
    #[cfg(target_os = "linux")]
    {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("FrameArtisan")
    }

    #[cfg(target_os = "macos")]
    {
        dirs::data_dir()
            .unwrap_or_else(|| PathBuf::from("/tmp"))
            .join("FrameArtisan")
    }

    #[cfg(target_os = "windows")]
    {
        dirs::data_local_dir()
            .unwrap_or_else(|| PathBuf::from("C:\\Users\\Default\\AppData\\Local"))
            .join("FrameArtisan")
    }
}

/// Python version to install.
pub const PYTHON_VERSION: &str = "3.12";

/// Frame Artisan release tarball URL.
pub fn app_release_url() -> String {
    let version = "0.1.0";
    format!("https://github.com/asomoza/frame-artisan/archive/refs/tags/v{version}.tar.gz")
}
