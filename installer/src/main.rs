mod downloader;
mod install_steps;
mod logger;
mod platform;
mod ui;

use eframe::egui;

fn main() -> eframe::Result<()> {
    let log_path = logger::init();
    log::info!("Frame Artisan Installer v{}", env!("CARGO_PKG_VERSION"));
    log::info!("Log file: {}", log_path.display());
    log::info!("OS: {}, Arch: {}", std::env::consts::OS, std::env::consts::ARCH);

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([500.0, 340.0])
            .with_resizable(false),
        ..Default::default()
    };

    eframe::run_native(
        "Frame Artisan Installer",
        options,
        Box::new(|_cc| Ok(Box::new(ui::InstallerApp::new()))),
    )
}
