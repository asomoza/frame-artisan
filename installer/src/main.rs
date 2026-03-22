mod downloader;
mod install_steps;
mod platform;
mod ui;

use eframe::egui;

fn main() -> eframe::Result<()> {
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
