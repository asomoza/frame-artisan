use eframe::egui;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;

use crate::install_steps::{self, InstallProgress, SharedProgress};
use crate::platform::{self, GpuType};

/// The main application state for the installer UI.
pub struct InstallerApp {
    /// Current screen
    screen: Screen,
    /// Selected GPU type
    gpu: GpuType,
    /// Installation directory
    install_dir: String,
    /// Shortcut options
    add_menu: bool,
    add_desktop: bool,
    /// Shared progress state (accessed by both UI and async task)
    progress: SharedProgress,
    /// Tokio runtime for async operations
    runtime: tokio::runtime::Runtime,
}

enum Screen {
    Setup,
    Installing,
    Done,
}

impl InstallerApp {
    pub fn new() -> Self {
        let default_dir = platform::default_install_dir();
        Self {
            screen: Screen::Setup,
            gpu: GpuType::NvidiaCuda,
            install_dir: default_dir.to_string_lossy().into_owned(),
            add_menu: true,
            add_desktop: false,
            progress: Arc::new(Mutex::new(InstallProgress::default())),
            runtime: tokio::runtime::Runtime::new().expect("Failed to create tokio runtime"),
        }
    }

    fn draw_setup(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(16.0);
            ui.heading("Frame Artisan Installer");
            ui.add_space(16.0);

            ui.label("GPU Type:");
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.gpu, GpuType::NvidiaCuda, "NVIDIA (CUDA)");
                ui.selectable_value(&mut self.gpu, GpuType::AmdRocm, "AMD (ROCm)");
                ui.selectable_value(&mut self.gpu, GpuType::Cpu, "CPU only");
            });

            ui.add_space(12.0);
            ui.label("Install location:");
            ui.text_edit_singleline(&mut self.install_dir);

            ui.add_space(12.0);
            ui.label("Shortcuts:");
            ui.checkbox(&mut self.add_menu, "Add to application menu");
            ui.checkbox(&mut self.add_desktop, "Add desktop shortcut");

            ui.add_space(24.0);

            if ui
                .add_sized([120.0, 36.0], egui::Button::new("Install"))
                .clicked()
            {
                self.start_install();
            }
        });
    }

    fn draw_installing(&mut self, ctx: &egui::Context) {
        // Request repaint every frame while installing
        ctx.request_repaint();

        // Read progress without blocking the UI
        let progress = self.progress.clone();
        let snapshot = self.runtime.block_on(async { progress.lock().await.clone() });

        if snapshot.finished || snapshot.error.is_some() {
            self.screen = Screen::Done;
            return;
        }

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(16.0);
            ui.heading("Installing...");
            ui.add_space(16.0);

            ui.strong(&snapshot.stage);
            if !snapshot.detail.is_empty() {
                ui.label(&snapshot.detail);
            }

            ui.add_space(12.0);

            // Progress bar
            if let Some(total) = snapshot.bytes_total {
                if total > 0 {
                    let fraction = snapshot.bytes_downloaded as f32 / total as f32;
                    let downloaded_mb = snapshot.bytes_downloaded as f64 / 1_048_576.0;
                    let total_mb = total as f64 / 1_048_576.0;

                    ui.add(
                        egui::ProgressBar::new(fraction)
                            .text(format!("{downloaded_mb:.1} / {total_mb:.1} MB")),
                    );
                }
            } else if snapshot.bytes_downloaded > 0 {
                let downloaded_mb = snapshot.bytes_downloaded as f64 / 1_048_576.0;
                ui.add(
                    egui::ProgressBar::new(0.0)
                        .text(format!("{downloaded_mb:.1} MB downloaded")),
                );
            } else {
                ui.spinner();
            }
        });
    }

    fn draw_done(&mut self, ctx: &egui::Context) {
        let snapshot = self.runtime.block_on(async { self.progress.lock().await.clone() });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add_space(16.0);

            if let Some(ref error) = snapshot.error {
                ui.heading("Installation Failed");
                ui.add_space(8.0);
                ui.colored_label(egui::Color32::from_rgb(255, 100, 100), error);
                ui.add_space(16.0);
                let log_path = dirs::data_local_dir()
                    .unwrap_or_default()
                    .join("FrameArtisan")
                    .join("installer.log");
                ui.label("Full error details are in the log file:");
                ui.add_space(4.0);
                ui.code(log_path.to_string_lossy().to_string());
                ui.add_space(4.0);
                ui.label("Please include this file when reporting issues.");
                ui.add_space(16.0);
                if ui.button("Retry").clicked() {
                    self.screen = Screen::Setup;
                }
            } else {
                ui.heading("Installation Complete!");
                ui.add_space(8.0);
                ui.label(&snapshot.detail);
                ui.add_space(16.0);
                ui.label("You can now run Frame Artisan from:");
                ui.add_space(4.0);

                #[cfg(unix)]
                let launcher = PathBuf::from(&self.install_dir).join("frameartisan");
                #[cfg(target_os = "windows")]
                let launcher = PathBuf::from(&self.install_dir).join("frameartisan.bat");

                ui.code(launcher.to_string_lossy().to_string());

                ui.add_space(16.0);
                if ui.button("Close").clicked() {
                    std::process::exit(0);
                }
            }
        });
    }

    fn start_install(&mut self) {
        self.screen = Screen::Installing;

        let install_dir = PathBuf::from(&self.install_dir);
        let gpu = self.gpu;
        let progress = Arc::clone(&self.progress);

        // Reset progress
        self.runtime.block_on(async {
            let mut p = progress.lock().await;
            *p = InstallProgress::default();
        });

        // Spawn the install task on the tokio runtime
        let add_menu = self.add_menu;
        let add_desktop = self.add_desktop;
        self.runtime.spawn(async move {
            install_steps::run_install(install_dir, gpu, add_menu, add_desktop, progress).await;
        });
    }
}

impl eframe::App for InstallerApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        match self.screen {
            Screen::Setup => self.draw_setup(ctx),
            Screen::Installing => self.draw_installing(ctx),
            Screen::Done => self.draw_done(ctx),
        }
    }
}
