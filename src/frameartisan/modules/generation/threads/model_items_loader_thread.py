import os

from PyQt6.QtCore import QMutex, QThread, QWaitCondition, pyqtSignal

from frameartisan.modules.generation.data_objects.model_item_data_object import ModelItemDataObject
from frameartisan.utils.database import Database


class ModelItemsLoaderThread(QThread):
    status_changed = pyqtSignal(str)
    batch_loaded = pyqtSignal(list)
    finished_loading = pyqtSignal()
    stop_requested = False

    def __init__(self, data_path: str, database_table: str, item_per_batch: int = 50, hide_nsfw: bool = False):
        super().__init__()

        self.database_path = os.path.join(data_path, "app.db")
        self.database_table = database_table
        self.item_per_batch = item_per_batch
        self.hide_nsfw = hide_nsfw

        self.pause_condition = QWaitCondition()
        self.pause_mutex = QMutex()
        self.paused = False

    def stop(self):
        self.stop_requested = True
        self.pause_condition.wakeAll()

    def pause(self):
        self.pause_mutex.lock()
        self.paused = True
        self.pause_mutex.unlock()

    def resume(self):
        self.pause_mutex.lock()
        self.paused = False
        self.pause_condition.wakeAll()
        self.pause_mutex.unlock()

    def run(self):
        self.database = Database(self.database_path)
        self.stop_requested = False
        self.status_changed.emit("Loading...")

        columns = ModelItemDataObject.get_column_names()
        conditions = {"deleted": 0}
        all_items = self.database.select(
            self.database_table, columns, conditions, order_by="name", order_by_direction="ASC"
        )

        batch = []
        for model in all_items:
            if self.stop_requested:
                break

            model_item = ModelItemDataObject.from_tuple(model)

            if self.hide_nsfw and model_item.tags is not None and "nsfw" in model_item.tags:
                continue

            batch.append(model_item)

            if len(batch) >= self.item_per_batch:
                self.batch_loaded.emit(batch)
                batch = []
                self.pause()

                self.pause_mutex.lock()
                if self.paused:
                    self.pause_condition.wait(self.pause_mutex)
                self.pause_mutex.unlock()

        if batch and not self.stop_requested:
            self.batch_loaded.emit(batch)

        self.database.disconnect()
        self.finished_loading.emit()
