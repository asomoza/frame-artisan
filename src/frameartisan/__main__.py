import logging
import logging.config
import os
import sys

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
