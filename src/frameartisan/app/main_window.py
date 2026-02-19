import logging
import os

from PyQt6.QtCore import QEasingCurve, QPoint, QPropertyAnimation, QSettings, QTimer
from PyQt6.QtWidgets import QApplication, QFrame, QHBoxLayout, QMainWindow, QStatusBar, QVBoxLayout, QWidget

from frameartisan.app.app import set_app_database_path, set_app_directories
from frameartisan.app.directories import DirectoriesObject
from frameartisan.app.event_bus import EventBus
from frameartisan.app.modules import MODULES
from frameartisan.app.preferences import PreferencesObject
from frameartisan.app.snackbar import SnackBar
from frameartisan.utils.database import Database


logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self,
        directories: DirectoriesObject,
        preferences: PreferencesObject,
        *args,
        splash_timer_duration: int = 2000,
        snackbar_duration: int = 3000,
        snackbar_animation_duration: int = 300,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        splash_timer = QTimer()
        splash_timer.singleShot(splash_timer_duration, self.close_splash)
        self.window_loaded = False
        self.timer_finished = False

        self.setWindowTitle("Frame Artisan")
        self.setMinimumSize(1300, 800)

        self.directories = directories
        self.preferences = preferences
        self.module = None

        self.settings = QSettings("ZCode", "FrameArtisan")

        self.settings.beginGroup("main_window")
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

        windowState = self.settings.value("windowState")
        if windowState is not None:
            self.restoreState(windowState)
        self.settings.endGroup()

        self.init_ui()

        self.snackbar_queue = []
        self.snackbar_duration = snackbar_duration
        self.snackbar_animation = QPropertyAnimation(self.snackbar, b"pos")
        self.snackbar_animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.snackbar_animation.setDuration(snackbar_animation_duration)
        self.snackbar_animation.finished.connect(self.on_snackbar_animation_finished)
        self.snackbar_hide_animation = False
        self.snackbar_closed = True
        self.snackbar_timer = QTimer()
        self.snackbar_timer.timeout.connect(self.hide_snackbar)

        self.event_bus = EventBus()

        self.event_bus.subscribe("show_snackbar", self.on_show_snackbar_event)
        self.event_bus.subscribe("status_message", self.on_status_message_event)

        # import heavy libraries like torch here
        import torch  # noqa: F401

        self.database = None
        set_app_directories(self.directories)
        self.init_database()

        self.settings.beginGroup("gui")
        self.gui_options = {
            "left_menu_expanded": self.settings.value("left_menu_expanded", True, type=bool),
            "current_module": self.settings.value("current_module", "Generation", type=str),
        }
        self.settings.endGroup()

        _, module_class = MODULES[self.gui_options.get("current_module")]
        self.load_module(module_class, self.gui_options.get("current_module"))

        self.window_loaded = True
        if self.timer_finished:
            QApplication.instance().close_splash()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        content_layout = QHBoxLayout()
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)

        workspace = QFrame()
        workspace.setObjectName("workspace")
        workspace.setFrameStyle(QFrame.Shape.Box | QFrame.Shadow.Sunken)
        self.workspace_layout = QVBoxLayout()
        self.workspace_layout.setContentsMargins(0, 0, 0, 0)
        self.workspace_layout.setSpacing(0)
        workspace.setLayout(self.workspace_layout)
        content_layout.addWidget(workspace)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        content_layout.setStretch(0, 0)
        content_layout.setStretch(1, 12)

        main_layout.addLayout(content_layout)

        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        self.snackbar = SnackBar(self)
        self.snackbar.closed.connect(self.on_snackbar_close)

    def close_splash(self):
        if self.window_loaded:
            QApplication.instance().close_splash()
        else:
            self.timer_finished = True

    def init_database(self):
        db_path = os.path.join(self.directories.data_path, "app.db")
        set_app_database_path(db_path)
        self.database = Database(db_path)

        self.database.execute("""
            CREATE TABLE IF NOT EXISTS model (
                id           INTEGER PRIMARY KEY,
                root_filename TEXT NOT NULL,
                filepath     TEXT NOT NULL,
                name         TEXT NOT NULL,
                version      TEXT NOT NULL,
                model_type   INTEGER NOT NULL,
                hash         TEXT NOT NULL UNIQUE,
                tags         TEXT,
                thumbnail    TEXT,
                triggers     TEXT,
                example      TEXT,
                deleted      INTEGER NOT NULL DEFAULT 0
            )
        """)

        self.database.execute("""
            CREATE TABLE IF NOT EXISTS component (
                id             INTEGER PRIMARY KEY,
                component_type TEXT NOT NULL,
                content_hash   TEXT NOT NULL UNIQUE,
                storage_path   TEXT NOT NULL,
                size_bytes     INTEGER DEFAULT 0,
                architecture   TEXT,
                config_json    TEXT,
                dtype          TEXT
            )
        """)

        self.database.execute("""
            CREATE TABLE IF NOT EXISTS model_component (
                model_id       INTEGER NOT NULL REFERENCES model(id),
                component_type TEXT NOT NULL,
                component_id   INTEGER NOT NULL REFERENCES component(id),
                PRIMARY KEY (model_id, component_type)
            )
        """)

        self.database.execute("""
            CREATE TABLE IF NOT EXISTS lora (
                id            INTEGER PRIMARY KEY,
                root_filename TEXT NOT NULL,
                filepath      TEXT NOT NULL,
                name          TEXT NOT NULL,
                version       TEXT NOT NULL DEFAULT '1.0',
                model_type    INTEGER NOT NULL DEFAULT 0,
                hash          TEXT NOT NULL UNIQUE,
                tags          TEXT,
                thumbnail     TEXT,
                triggers      TEXT,
                example       TEXT,
                deleted       INTEGER NOT NULL DEFAULT 0
            )
        """)

        self.database.execute("""
            CREATE TABLE IF NOT EXISTS model_component_variant (
                model_id       INTEGER NOT NULL REFERENCES model(id),
                component_type TEXT NOT NULL,
                component_id   INTEGER NOT NULL REFERENCES component(id),
                UNIQUE(model_id, component_type, component_id)
            )
        """)

        self.database.execute("""
            CREATE TABLE IF NOT EXISTS model_component_override (
                model_id       INTEGER NOT NULL REFERENCES model(id),
                component_type TEXT NOT NULL,
                component_id   INTEGER NOT NULL REFERENCES component(id),
                UNIQUE(model_id, component_type)
            )
        """)

        self.database.execute("""
            CREATE TABLE IF NOT EXISTS source_file (
                id           INTEGER PRIMARY KEY,
                kind         TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                filepath     TEXT NOT NULL,
                UNIQUE(kind, content_hash)
            )
        """)

    def check_and_show_download_dialog(self):
        """Show the model download dialog if no models exist in the database."""
        row = self.database.fetch_one("SELECT COUNT(*) FROM model WHERE deleted = 0")
        if row is not None and row[0] == 0:
            from frameartisan.configuration.model_download_dialog import ModelDownloadDialog

            dialog = ModelDownloadDialog(self.directories, self.preferences, self.database)
            dialog.exec()

    def closeEvent(self, event):
        self.settings.beginGroup("main_window")
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("windowState", self.saveState())
        self.settings.endGroup()

        self.settings.beginGroup("gui")
        self.settings.setValue("current_module", self.gui_options.get("current_module"))
        self.settings.endGroup()

        if self.module is not None:
            self.module.close()

        for child in self.findChildren(QWidget):
            child.deleteLater()

        super().closeEvent(event)

    def show_snackbar(self, message):
        self.snackbar_queue.append(message)

        if self.snackbar_closed:
            self.snackbar_closed = False
            self.show_next_snackbar()

    def show_next_snackbar(self):
        if not self.snackbar_queue:
            return

        self.snackbar_closed = False
        message = self.snackbar_queue.pop(0)
        self.snackbar.message = message
        snackbar_x = (self.width() - self.snackbar.width()) // 2
        snackbar_y = self.height() - self.snackbar.height() - 40
        self.snackbar.show()

        self.snackbar_animation.setStartValue(QPoint(snackbar_x, snackbar_y + 100))
        self.snackbar_animation.setEndValue(QPoint(snackbar_x, snackbar_y))
        self.snackbar_animation.start()

    def hide_snackbar(self):
        self.snackbar_timer.stop()
        snackbar_x = (self.width() - self.snackbar.width()) // 2
        snackbar_y = self.height() - self.snackbar.height() - 40

        self.snackbar_animation.setStartValue(QPoint(snackbar_x, snackbar_y))
        self.snackbar_animation.setEndValue(QPoint(snackbar_x, snackbar_y + 100))
        self.snackbar_animation.start()

    def on_snackbar_animation_finished(self):
        if self.snackbar_hide_animation:
            self.snackbar_closed = True
            self.snackbar_hide_animation = False
            self.snackbar.hide()

        if self.snackbar_closed:
            self.show_next_snackbar()
            return

        self.snackbar_hide_animation = True
        self.snackbar_timer.start(self.snackbar_duration)

    def on_snackbar_close(self):
        self.snackbar_timer.stop()
        self.snackbar_hide_animation = False
        self.snackbar_closed = True
        self.snackbar.hide()

        self.show_next_snackbar()

    def on_show_snackbar_event(self, data):
        action = data.get("action", "")

        if action == "show":
            message = data.get("message", "")
            self.show_snackbar(message)

    def on_status_message_event(self, data):
        action = data.get("action", "")
        message = data.get("message", "")

        if action == "change":
            self.status_bar.showMessage(message)

    def load_module(self, module_class, label):
        if self.workspace_layout.count() > 0:
            current_module: QWidget = self.workspace_layout.itemAt(0).widget()  # type: ignore
            current_module.close()
            self.workspace_layout.removeWidget(current_module)
            current_module.deleteLater()

        try:
            self.module = module_class(self.directories, self.preferences)
            self.workspace_layout.addWidget(self.module)
            self.gui_options["current_module"] = label
        except TypeError as module_error:
            logger.error("Error loading the module with this message: %s", str(module_error))
            logger.debug("TypeError exception", exc_info=True)
            self.show_snackbar(f"{module_error}")
