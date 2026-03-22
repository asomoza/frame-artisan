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
    let client = Client::builder()
        .redirect(reqwest::redirect::Policy::limited(10))
        .build()
        .map_err(|e| format!("Failed to create HTTP client: {e}"))?;

    let response = client
        .get(url)
        .send()
        .await
        .map_err(|e| format!("Download request failed: {e}"))?;

    if !response.status().is_success() {
        return Err(format!("Download failed with status: {}", response.status()));
    }

    let total_size = response.content_length();

    // Ensure parent directory exists
    if let Some(parent) = dest.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("Failed to create directory: {e}"))?;
    }

    let mut file = tokio::fs::File::create(dest)
        .await
        .map_err(|e| format!("Failed to create file: {e}"))?;

    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Error reading download stream: {e}"))?;
        file.write_all(&chunk)
            .await
            .map_err(|e| format!("Error writing to file: {e}"))?;
        downloaded += chunk.len() as u64;
        progress(downloaded, total_size);
    }

    file.flush()
        .await
        .map_err(|e| format!("Error flushing file: {e}"))?;

    Ok(downloaded)
}
