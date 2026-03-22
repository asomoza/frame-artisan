import ctypes
import logging
import os
import platform
import shutil
import tempfile
from importlib.resources import files

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtGui import QIcon, QPixmap
from PyQt6.QtWidgets import QApplication, QSplashScreen

from frameartisan.app.directories import DirectoriesObject
from frameartisan.app.main_window import MainWindow
from frameartisan.app.preferences import PreferencesObject
from frameartisan.configuration.initial_setup_dialog import InitialSetupDialog


logger = logging.getLogger(__name__)


class FrameArtisanApplication(QApplication):
    SPLASH_IMG = str(files("frameartisan.theme.images").joinpath("splash.webp"))
    APP_ICON = str(files("frameartisan.theme.images").joinpath("icon.png"))

    def __init__(self, *args, **kwargs):
        myappid = "zcode.frameartisan.010"

        if platform.system() == "Windows":
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)

        super().__init__(*args, **kwargs)

        self.setWindowIcon(QIcon(self.APP_ICON))

        style_data = files("frameartisan.theme").joinpath("stylesheet.qss").read_bytes()
        stylesheet = style_data.decode("utf-8")
        self.setStyleSheet(stylesheet)

        self.temp_path = tempfile.mkdtemp(prefix="frameartisan_")

        self.window = None
        self.splash = None

        self.directories = None
        self.preferences = None

        self.aboutToQuit.connect(self.cleanup_on_exit)

        if not self.check_initial_setup():
            self.dialog = InitialSetupDialog(self.directories, self.preferences)
            self.dialog.exec()

        self.load_main_window()

    def check_initial_setup(self):
        settings = QSettings("ZCode", "FrameArtisan")

        save_video_metadata = settings.value("save_video_metadata", False, type=bool)
        hide_nsfw = settings.value("hide_nsfw", True, type=bool)
        delete_lora_on_import = settings.value("delete_lora_on_import", False, type=bool)
        delete_model_on_import = settings.value("delete_model_on_import", False, type=bool)
        save_source_images = settings.value("save_source_images", False, type=bool)
        save_source_audio = settings.value("save_source_audio", False, type=bool)
        save_source_video = settings.value("save_source_video", False, type=bool)
        auto_save_videos = settings.value("auto_save_videos", False, type=bool)

        self.preferences = PreferencesObject(
            save_video_metadata=save_video_metadata,
            hide_nsfw=hide_nsfw,
            delete_lora_on_import=delete_lora_on_import,
            delete_model_on_import=delete_model_on_import,
            save_source_images=save_source_images,
            save_source_audio=save_source_audio,
            save_source_video=save_source_video,
            auto_save_videos=auto_save_videos,
        )

        data_path = settings.value("data_path", None, type=str)
        models_diffusers = settings.value("models_diffusers", None, type=str)
        models_loras = settings.value("models_loras", None, type=str)
        models_controlnets = settings.value("models_controlnets", None, type=str)
        outputs_videos = settings.value("outputs_videos", None, type=str)
        outputs_source_images = settings.value("outputs_source_images", None, type=str)
        outputs_source_videos = settings.value("outputs_source_videos", None, type=str)
        outputs_source_audio = settings.value("outputs_source_audio", None, type=str)
        if not outputs_source_audio and data_path:
            outputs_source_audio = os.path.join(os.path.dirname(data_path), "outputs", "source_audio")
            os.makedirs(outputs_source_audio, exist_ok=True)
            settings.setValue("outputs_source_audio", outputs_source_audio)
        outputs_lora_masks = settings.value("outputs_lora_masks", None, type=str)
        if not outputs_lora_masks and data_path:
            outputs_lora_masks = os.path.join(os.path.dirname(data_path), "outputs", "lora_masks")
            os.makedirs(outputs_lora_masks, exist_ok=True)
            settings.setValue("outputs_lora_masks", outputs_lora_masks)
        cache_path = settings.value("cache_path", None, type=str)
        if not cache_path and data_path:
            # Auto-set cache_path for existing installations that predate this field.
            cache_path = os.path.join(os.path.dirname(data_path), "cache")
            os.makedirs(cache_path, exist_ok=True)
            settings.setValue("cache_path", cache_path)

        # Set torch inductor cache directory. TORCHINDUCTOR_CACHE_DIR is read
        # lazily (on first compilation), so it can be set after torch is imported.
        if cache_path:
            compile_cache_dir = os.path.join(cache_path, "torch_compile")
            os.makedirs(compile_cache_dir, exist_ok=True)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = compile_cache_dir

        self.directories = DirectoriesObject(
            data_path=data_path,
            models_diffusers=models_diffusers,
            models_loras=models_loras,
            models_controlnets=models_controlnets,
            outputs_videos=outputs_videos,
            outputs_source_images=outputs_source_images,
            outputs_source_videos=outputs_source_videos,
            outputs_source_audio=outputs_source_audio,
            outputs_lora_masks=outputs_lora_masks,
            cache_path=cache_path,
            temp_path=self.temp_path,
        )

        if any(
            not v
            for v in [
                data_path,
                models_diffusers,
                models_loras,
                models_controlnets,
                outputs_videos,
                outputs_source_images,
                outputs_source_videos,
                outputs_source_audio,
                cache_path,
            ]
        ):
            return False
        return True

    def cleanup_on_exit(self):
        try:
            from frameartisan.app.model_manager import get_model_manager

            get_model_manager().clear()
        except Exception as e:
            logger.debug("Failed to clear model manager on exit: %s", e)

        self.cleanup_temp_path()

    def cleanup_temp_path(self):
        if self.temp_path and os.path.exists(self.temp_path):  # check if it exists
            try:
                shutil.rmtree(self.temp_path)
                logger.info(f"Temporary directory '{self.temp_path}' cleaned up successfully.")
            except OSError as e:
                logger.error(f"Error cleaning up temporary directory: {e}")

    def close_splash(self):
        if self.splash:
            self.splash.close()
            self.window.show()
            self.window.check_and_show_download_dialog()

    def load_main_window(self):
        splash_pix = QPixmap(self.SPLASH_IMG)
        self.splash = QSplashScreen(splash_pix)
        self.splash.showMessage(
            "Loading...",
            alignment=Qt.AlignmentFlag.AlignBottom,
            color=Qt.GlobalColor.white,
        )

        self.splash.show()

        self.window = MainWindow(self.directories, self.preferences)
