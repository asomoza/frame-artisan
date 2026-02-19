"""Application-level singletons for accessing shared state from non-UI code."""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from frameartisan.app.directories import DirectoriesObject


_lock = threading.Lock()
_database_path: str | None = None
_directories: DirectoriesObject | None = None


def set_app_database_path(path: str) -> None:
    global _database_path
    with _lock:
        _database_path = path


def get_app_database_path() -> str | None:
    with _lock:
        return _database_path


def set_app_directories(directories: DirectoriesObject) -> None:
    global _directories
    with _lock:
        _directories = directories


def get_app_directories() -> DirectoriesObject | None:
    with _lock:
        return _directories
