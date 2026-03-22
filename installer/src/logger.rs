use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use log::LevelFilter;

/// Initialize file-based logging. Returns the log file path.
pub fn init() -> PathBuf {
    let log_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join("FrameArtisan");

    fs::create_dir_all(&log_dir).ok();

    let log_path = log_dir.join("installer.log");

    // Truncate on each run so the log is fresh
    let file = fs::File::create(&log_path).expect("Failed to create log file");
    let file = Mutex::new(file);

    let log_path_clone = log_path.clone();

    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .format(move |_buf, record| {
            let now = chrono::Local::now().format("%Y-%m-%d %H:%M:%S");
            let line = format!("[{now}] [{:>5}] {}\n", record.level(), record.args());

            // Write to file
            if let Ok(mut f) = file.lock() {
                f.write_all(line.as_bytes()).ok();
                f.flush().ok();
            }

            // Also print to stderr for debugging
            eprint!("{line}");

            Ok(())
        })
        .init();

    log_path_clone
}
