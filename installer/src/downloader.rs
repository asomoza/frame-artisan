use futures_util::StreamExt;
use reqwest::Client;
use std::path::Path;
use tokio::io::AsyncWriteExt;

/// Downloads a file with progress tracking.
/// Returns total bytes downloaded.
pub async fn download_file(
    url: &str,
    dest: &Path,
    progress: impl Fn(u64, Option<u64>),
) -> Result<u64, String> {
    log::info!("Downloading: {url}");
    log::info!("Destination: {}", dest.display());

    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .user_agent("frame-artisan-installer/0.1.0")
        .build()
        .map_err(|e| {
            log::error!("Failed to create HTTP client: {e}");
            format!("Failed to create HTTP client: {e}")
        })?;

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| {
            log::error!("Download request failed for {url}: {e}");
            format!("Download request failed: {e}")
        })?;

    let status = response.status();
    let final_url = response.url().to_string();
    log::info!("Response status: {status}, final URL: {final_url}");

    if !status.is_success() {
        let msg = format!("Download failed with status {status} for {url} (redirected to {final_url})");
        log::error!("{msg}");
        return Err(msg);
    }

    let total_size = response.content_length();
    log::info!("Content-Length: {total_size:?}");

    // Ensure parent directory exists
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| {
                log::error!("Failed to create directory {}: {e}", parent.display());
                format!("Failed to create directory: {e}")
            })?;
    }

    let mut file = tokio::fs::File::create(dest)
        .await
        .map_err(|e| {
            log::error!("Failed to create file {}: {e}", dest.display());
            format!("Failed to create file: {e}")
        })?;

    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| {
            log::error!("Error reading download stream: {e}");
            format!("Error reading download stream: {e}")
        })?;
        file.write_all(&chunk)
            .await
            .map_err(|e| {
                log::error!("Error writing to file: {e}");
                format!("Error writing to file: {e}")
            })?;
        downloaded += chunk.len() as u64;
        progress(downloaded, total_size);
    }

    file.flush()
        .await
        .map_err(|e| format!("Error flushing file: {e}"))?;

    log::info!("Download complete: {} bytes", downloaded);
    Ok(downloaded)
}
