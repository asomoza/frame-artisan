import ctypes
import glob
import importlib.util
import logging
import logging.config
import os
import sys


# Preload torch DLLs before PyQt6 to avoid DLL search order conflicts on Windows
if sys.platform == "win32":
    _torch_spec = importlib.util.find_spec("torch")
    if _torch_spec and _torch_spec.origin:
        _torch_lib = os.path.join(os.path.dirname(_torch_spec.origin), "lib")
        os.add_dll_directory(_torch_lib)
        _kernel32 = ctypes.WinDLL("kernel32.dll", use_last_error=True)
        _kernel32.LoadLibraryW.restype = ctypes.c_void_p
        for _dll in sorted(glob.glob(os.path.join(_torch_lib, "*.dll"))):
            _kernel32.LoadLibraryW(_dll)

from frameartisan.app.frameartisan_application import FrameArtisanApplication
from frameartisan.app.logging_conf import logging_config


logging.getLogger("PIL").setLevel(logging.ERROR)


def my_exception_hook(exctype, value, traceback):
    print(f"Unhandled exception: {value}")
    sys.__excepthook__(exctype, value, traceback)


sys.excepthook = my_exception_hook


def main():
    from frameartisan.app.model_manager import ModelManager, set_global_model_manager

    set_global_model_manager(ModelManager())
    app = FrameArtisanApplication(sys.argv)
    sys.exit(app.exec())


if __name__ == "__main__":
    os.makedirs(
        os.path.dirname(logging_config["handlers"]["fileHandler"]["filename"]),
        exist_ok=True,
    )
    logging.config.dictConfig(logging_config)
    main()
