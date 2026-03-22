use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::process::Command;
use tokio::sync::Mutex;

use crate::downloader;
use crate::platform;

/// Shared state between the async install task and the UI.
#[derive(Debug, Clone)]
pub struct InstallProgress {
    pub stage: String,
    pub detail: String,
    pub bytes_downloaded: u64,
    pub bytes_total: Option<u64>,
    pub finished: bool,
    pub error: Option<String>,
}

impl Default for InstallProgress {
    fn default() -> Self {
        Self {
            stage: "Waiting...".into(),
            detail: String::new(),
            bytes_downloaded: 0,
            bytes_total: None,
            finished: false,
            error: None,
        }
    }
}

pub type SharedProgress = Arc<Mutex<InstallProgress>>;

/// Run the full installation pipeline.
pub async fn run_install(
    install_dir: PathBuf,
    gpu: platform::GpuType,
    progress: SharedProgress,
) {
    log::info!("Starting installation: dir={}, gpu={gpu}", install_dir.display());
    let result = run_install_inner(&install_dir, gpu, &progress).await;
    let mut p = progress.lock().await;
    match result {
        Ok(()) => {
            log::info!("Installation complete: {}", install_dir.display());
            p.stage = "Installation complete!".into();
            p.detail = format!("Installed to {}", install_dir.display());
            p.finished = true;
        }
        Err(e) => {
            log::error!("Installation failed: {e}");
            p.error = Some(e);
        }
    }
}

async fn run_install_inner(
    install_dir: &Path,
    gpu: platform::GpuType,
    progress: &SharedProgress,
) -> Result<(), String> {
    tokio::fs::create_dir_all(install_dir)
        .await
        .map_err(|e| format!("Failed to create install dir: {e}"))?;

    // Step 1: Download uv
    let uv_path = install_dir.join(platform::uv_binary_name());
    if !uv_path.exists() {
        set_stage(progress, "Downloading uv...", "").await;
        let archive_path = install_dir.join("uv_archive");
        download_with_progress(platform::uv_download_url(), &archive_path, progress).await?;

        set_stage(progress, "Extracting uv...", "").await;
        extract_uv(&archive_path, install_dir).await?;
        tokio::fs::remove_file(&archive_path).await.ok();
    }

    // Step 2: Install Python via uv
    set_stage(progress, "Installing Python...", platform::PYTHON_VERSION).await;
    run_command(
        &uv_path,
        &["python", "install", platform::PYTHON_VERSION],
        install_dir,
    )
    .await?;

    // Step 3: Create virtual environment
    let venv_path = install_dir.join(".venv");
    if !venv_path.exists() {
        set_stage(progress, "Creating virtual environment...", "").await;
        run_command(
            &uv_path,
            &[
                "venv",
                venv_path.to_str().unwrap(),
                "--python",
                platform::PYTHON_VERSION,
            ],
            install_dir,
        )
        .await?;
    }

    // Step 4: Download torch wheel (heavy — needs progress bar)
    let torch_wheel_url = platform::torch_wheel_url(gpu);
    let wheel_filename = torch_wheel_url
        .rsplit('/')
        .next()
        .unwrap_or("torch.whl");
    // Decode URL-encoded filename (e.g. %2B -> +)
    let wheel_filename = urldecode(wheel_filename);
    let wheel_path = install_dir.join("cache").join(&wheel_filename);

    if !wheel_path.exists() {
        set_stage(progress, "Downloading PyTorch...", &format!("{gpu}")).await;
        download_with_progress(&torch_wheel_url, &wheel_path, progress).await?;
    }

    // Step 5: Install torch from local wheel
    set_stage(progress, "Installing PyTorch...", "from local wheel").await;
    run_command(
        &uv_path,
        &[
            "pip",
            "install",
            wheel_path.to_str().unwrap(),
            "--python",
            venv_path.to_str().unwrap(),
        ],
        install_dir,
    )
    .await?;

    // Step 6: Download and install Frame Artisan from release
    let app_url = platform::app_release_url();
    let app_tarball = install_dir.join("cache").join("frame-artisan.tar.gz");

    if !app_tarball.exists() {
        set_stage(progress, "Downloading Frame Artisan...", "").await;
        download_with_progress(&app_url, &app_tarball, progress).await?;
    }

    set_stage(progress, "Installing Frame Artisan...", "this may take a minute").await;
    let torch_index = platform::torch_index_url(gpu);
    run_command(
        &uv_path,
        &[
            "pip",
            "install",
            app_tarball.to_str().unwrap(),
            "--extra-index-url",
            torch_index,
            "--python",
            venv_path.to_str().unwrap(),
        ],
        install_dir,
    )
    .await?;

    // Step 7: Create launcher script
    set_stage(progress, "Creating launcher...", "").await;
    create_launcher(install_dir, &venv_path).await?;

    Ok(())
}

async fn set_stage(progress: &SharedProgress, stage: &str, detail: &str) {
    log::info!("[STAGE] {stage} {detail}");
    let mut p = progress.lock().await;
    p.stage = stage.to_string();
    p.detail = detail.to_string();
    p.bytes_downloaded = 0;
    p.bytes_total = None;
}

async fn download_with_progress(
    url: &str,
    dest: &Path,
    progress: &SharedProgress,
) -> Result<(), String> {
    let progress = Arc::clone(progress);
    downloader::download_file(url, dest, move |downloaded, total| {
        // Use try_lock to avoid blocking the download on the UI
        if let Ok(mut p) = progress.try_lock() {
            p.bytes_downloaded = downloaded;
            p.bytes_total = total;
        }
    })
    .await?;
    Ok(())
}

async fn run_command(program: &Path, args: &[&str], cwd: &Path) -> Result<(), String> {
    let args_str = args.join(" ");
    log::info!("Running: {} {args_str}", program.display());
    let output = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .output()
        .await
        .map_err(|e| {
            log::error!("Failed to run {}: {e}", program.display());
            format!("Failed to launch command. Check the log file for details.")
        })?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !stdout.is_empty() {
        log::info!("stdout: {stdout}");
    }
    if !stderr.is_empty() {
        log::info!("stderr: {stderr}");
    }

    if !output.status.success() {
        // Log the full error for debugging
        log::error!("Command failed: {} {args_str}", program.display());
        log::error!("Exit code: {}", output.status);
        log::error!("Full stderr:\n{stderr}");

        // Return a short message for the UI — the log has the full details
        let short_error = stderr
            .lines()
            .filter(|l| !l.trim().is_empty())
            .last()
            .unwrap_or("Unknown error");
        return Err(format!(
            "Step failed: {short_error}\n\nSee the log file for full details."
        ));
    }

    log::info!("Command succeeded: exit {}", output.status);
    Ok(())
}

async fn extract_uv(archive_path: &Path, dest_dir: &Path) -> Result<(), String> {
    let archive_path = archive_path.to_path_buf();
    let dest_dir = dest_dir.to_path_buf();

    tokio::task::spawn_blocking(move || {
        let file =
            std::fs::File::open(&archive_path).map_err(|e| format!("Open archive: {e}"))?;

        let ext = archive_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        if ext == "zip" || archive_path.to_string_lossy().ends_with(".zip") {
            // Windows: zip archive
            let mut archive =
                zip::ZipArchive::new(file).map_err(|e| format!("Read zip: {e}"))?;
            for i in 0..archive.len() {
                let mut entry = archive
                    .by_index(i)
                    .map_err(|e| format!("Zip entry: {e}"))?;
                let name = entry.name().to_string();
                if name.ends_with("uv.exe") || name.ends_with("uv") {
                    let out_path = dest_dir.join(platform::uv_binary_name());
                    let mut out_file = std::fs::File::create(&out_path)
                        .map_err(|e| format!("Create uv binary: {e}"))?;
                    std::io::copy(&mut entry, &mut out_file)
                        .map_err(|e| format!("Extract uv: {e}"))?;

                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        std::fs::set_permissions(
                            &out_path,
                            std::fs::Permissions::from_mode(0o755),
                        )
                        .ok();
                    }
                    return Ok(());
                }
            }
            Err("uv binary not found in zip".into())
        } else {
            // Linux/macOS: tar.gz
            let gz = flate2::read::GzDecoder::new(file);
            let mut archive = tar::Archive::new(gz);
            for entry in archive.entries().map_err(|e| format!("Read tar: {e}"))? {
                let mut entry = entry.map_err(|e| format!("Tar entry: {e}"))?;
                let path = entry.path().map_err(|e| format!("Entry path: {e}"))?;
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("");
                if name == "uv" {
                    let out_path = dest_dir.join("uv");
                    entry
                        .unpack(&out_path)
                        .map_err(|e| format!("Unpack uv: {e}"))?;

                    #[cfg(unix)]
                    {
                        use std::os::unix::fs::PermissionsExt;
                        std::fs::set_permissions(
                            &out_path,
                            std::fs::Permissions::from_mode(0o755),
                        )
                        .ok();
                    }
                    return Ok(());
                }
            }
            Err("uv binary not found in tar.gz".into())
        }
    })
    .await
    .map_err(|e| format!("Extract task failed: {e}"))?
}

async fn create_launcher(install_dir: &Path, venv_path: &Path) -> Result<(), String> {
    #[cfg(unix)]
    {
        let launcher = install_dir.join("frameartisan");
        let content = format!(
            "#!/bin/sh\nexec \"{}\" -m frameartisan \"$@\"\n",
            venv_path.join("bin/python").display()
        );
        tokio::fs::write(&launcher, content)
            .await
            .map_err(|e| format!("Write launcher: {e}"))?;

        // Make executable
        let launcher_clone = launcher.clone();
        tokio::task::spawn_blocking(move || {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&launcher_clone, std::fs::Permissions::from_mode(0o755)).ok();
        })
        .await
        .ok();
    }

    #[cfg(target_os = "windows")]
    {
        let launcher = install_dir.join("frameartisan.bat");
        let content = format!(
            "@echo off\r\n\"{}\" -m frameartisan %*\r\n",
            venv_path.join("Scripts\\python.exe").display()
        );
        tokio::fs::write(&launcher, content)
            .await
            .map_err(|e| format!("Write launcher: {e}"))?;
    }

    Ok(())
}

/// Simple URL decode for %XX sequences.
fn urldecode(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '%' {
            let hex: String = chars.by_ref().take(2).collect();
            if let Ok(byte) = u8::from_str_radix(&hex, 16) {
                result.push(byte as char);
            } else {
                result.push('%');
                result.push_str(&hex);
            }
        } else {
            result.push(c);
        }
    }
    result
}
