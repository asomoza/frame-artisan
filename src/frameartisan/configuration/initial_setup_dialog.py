import os

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import QDialog, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from frameartisan.app.directories import DirectoriesObject
from frameartisan.app.preferences import PreferencesObject


class InitialSetupDialog(QDialog):
    border_color = QColor("#ff6b6b6b")

    # Keys the user must set before finishing
    REQUIRED_PATHS = (
        ("data_path", "Select data directory"),
        ("models_diffusers", "Select diffusers models directory"),
        ("models_loras", "Select LoRAs directory"),
        ("models_controlnets", "Select ControlNets directory"),
        ("outputs_videos", "Select video outputs directory"),
        ("outputs_source_images", "Select source images outputs directory"),
        ("outputs_source_videos", "Select source videos outputs directory"),
        ("outputs_source_audio", "Select source audio outputs directory"),
        ("cache_path", "Select cache directory"),
    )

    def __init__(self, directories: DirectoriesObject, preferences: PreferencesObject, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dialog_width = 1000
        self.dialog_height = 800
        self.setFixedSize(self.dialog_width, self.dialog_height)

        self.setWindowTitle("Initial Setup")

        # Prevent closing until setup is completed via Finish
        self._allow_close = False
        self.setWindowFlag(Qt.WindowType.WindowCloseButtonHint, False)

        self.directories = directories
        self.preferences = preferences

        self._settings = QSettings("ZCode", "FrameArtisan")
        self._path_labels: dict[str, QLabel] = {}
        self._path_select_buttons: list[QPushButton] = []
        self._finish_button: QPushButton | None = None
        self._defaults_button: QPushButton | None = None

        self.init_ui()
        self._sync_from_current_values()
        self._update_finish_enabled()

    def reject(self) -> None:
        """Block Esc / reject() while setup is required."""
        if self._allow_close:
            return super().reject()

        QMessageBox.information(
            self,
            "Setup required",
            "Please complete the initial setup before continuing.",
        )

    def closeEvent(self, event):
        """Block window manager close (X / Alt+F4) while setup is required."""
        if self._allow_close:
            event.accept()
            return

        event.ignore()
        QMessageBox.information(
            self,
            "Setup required",
            "Please complete the initial setup before continuing.",
        )

    def init_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(10)

        title = QLabel("Choose required directories")
        title.setStyleSheet("font-size: 18px; font-weight: 600;")
        main_layout.addWidget(title)

        hint = QLabel("All paths must be selected to continue.")
        hint.setStyleSheet("color: #9aa0a6;")
        main_layout.addWidget(hint)

        main_layout.addSpacing(10)

        for key, button_text in self.REQUIRED_PATHS:
            row = self._make_path_row(key=key, button_text=button_text)
            main_layout.addWidget(row)

        self._apply_uniform_path_button_width()

        main_layout.addStretch()

        # Buttons
        buttons_row = QHBoxLayout()
        buttons_row.setContentsMargins(0, 0, 0, 0)
        buttons_row.setSpacing(10)

        self._defaults_button = QPushButton("Set defaults")
        self._defaults_button.clicked.connect(self.set_defaults)

        self._finish_button = QPushButton("Finish setup")
        self._finish_button.setEnabled(False)
        self._finish_button.clicked.connect(self.finish_setup)

        buttons_row.addWidget(self._defaults_button)
        buttons_row.addStretch(1)
        buttons_row.addWidget(self._finish_button)

        main_layout.addLayout(buttons_row)

        self.setLayout(main_layout)

    def _apply_uniform_path_button_width(self) -> None:
        """Make all 'Select ...' buttons the same width for a uniform layout."""
        if not self._path_select_buttons:
            return

        max_width = max(btn.sizeHint().width() for btn in self._path_select_buttons)
        for btn in self._path_select_buttons:
            btn.setFixedWidth(max_width)

    def _make_path_row(self, key: str, button_text: str) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        label_title = QLabel(f"{key}:")
        label_title.setMinimumWidth(170)

        label_value = QLabel("(not set)")
        label_value.setStyleSheet("color: #9aa0a6;")
        label_value.setWordWrap(True)

        button = QPushButton(button_text)
        button.clicked.connect(lambda _=False, k=key: self.select_directory(k))
        self._path_select_buttons.append(button)

        self._path_labels[key] = label_value

        layout.addWidget(label_title)
        layout.addWidget(label_value, 1)
        layout.addWidget(button)

        container.setLayout(layout)
        return container

    def _sync_from_current_values(self) -> None:
        """If directories already contain values, reflect them in the UI."""
        for key, _ in self.REQUIRED_PATHS:
            current_value = getattr(self.directories, key, "") or ""
            if current_value:
                self._set_path_label(key, current_value)

    def _set_path_label(self, key: str, path: str) -> None:
        label = self._path_labels.get(key)
        if not label:
            return
        label.setText(path)
        label.setStyleSheet("")  # reset "not set" style

    def _is_complete(self) -> bool:
        for key, _ in self.REQUIRED_PATHS:
            value = getattr(self.directories, key, "") or ""
            if not value:
                return False
        return True

    def _update_finish_enabled(self) -> None:
        if self._finish_button is not None:
            self._finish_button.setEnabled(self._is_complete())

    def _default_paths(self) -> dict[str, str]:
        # Default base under the user's home directory.
        base = os.path.join(os.path.expanduser("~"), "FrameArtisan")
        return {
            "data_path": os.path.join(base, "data"),
            "models_diffusers": os.path.join(base, "models", "diffusers"),
            "models_loras": os.path.join(base, "models", "loras"),
            "models_controlnets": os.path.join(base, "models", "controlnets"),
            "outputs_videos": os.path.join(base, "outputs", "videos"),
            "outputs_source_images": os.path.join(base, "outputs", "source_images"),
            "outputs_source_videos": os.path.join(base, "outputs", "source_videos"),
            "outputs_source_audio": os.path.join(base, "outputs", "source_audio"),
            "cache_path": os.path.join(base, "cache"),
        }

    def set_defaults(self) -> None:
        defaults = self._default_paths()

        # Ensure the directories exist and persist them immediately (same behavior as manual selection).
        for key, _ in self.REQUIRED_PATHS:
            path = defaults.get(key, "")
            if not path:
                continue

            os.makedirs(path, exist_ok=True)

            setattr(self.directories, key, path)
            self._settings.setValue(key, path)
            self._set_path_label(key, path)

        self._update_finish_enabled()

    def finish_setup(self):
        # Persist all values once everything is set (and again for safety).
        for key, _ in self.REQUIRED_PATHS:
            value = getattr(self.directories, key, "") or ""
            self._settings.setValue(key, value)

        # Allow the dialog to close only via successful Finish
        self._allow_close = True
        self.close()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        pen = QPen(self.border_color)
        pen.setWidth(1)
        painter.setPen(pen)
        painter.drawLine(0, 0, 0, self.height())
        painter.drawLine(0, 0, self.width(), 0)
        painter.drawLine(self.width(), 0, self.width(), self.height())
        painter.drawLine(0, self.height(), self.width(), self.height())

    def select_directory(self, key: str):
        home_dir = os.path.expanduser("~")

        dialog = QFileDialog(self)
        options = (
            QFileDialog.Option.ShowDirsOnly
            | QFileDialog.Option.DontUseNativeDialog
            | QFileDialog.Option.ReadOnly
            | QFileDialog.Option.HideNameFilterDetails
        )
        dialog.setOptions(options)

        selected_path = dialog.getExistingDirectory(self, "Select a directory", home_dir)
        if not selected_path:
            return  # user canceled

        setattr(self.directories, key, selected_path)
        self._settings.setValue(key, selected_path)
        self._set_path_label(key, selected_path)
        self._update_finish_enabled()
