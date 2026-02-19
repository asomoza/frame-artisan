import logging

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QAction, QContextMenuEvent
from PyQt6.QtWidgets import QMenu, QPushButton, QSizePolicy


logger = logging.getLogger(__name__)


class GenerateButton(QPushButton):
    auto_save_changed = pyqtSignal(bool)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )

        self.auto_save = False
        self.continuous_generation = False

    def contextMenuEvent(self, event: QContextMenuEvent):
        context_menu = QMenu(self)

        auto_save_action: QAction | None = context_menu.addAction("Auto save")
        auto_save_action.setCheckable(True)
        auto_save_action.setChecked(self.auto_save)
        auto_save_action.triggered.connect(self.handle_auto_save_triggered)

        continuous_generation_action: QAction | None = context_menu.addAction("Continuous generation")
        continuous_generation_action.setCheckable(True)
        continuous_generation_action.setChecked(self.continuous_generation)
        continuous_generation_action.triggered.connect(self.handle_continuous_generation_triggered)

        context_menu.exec(event.globalPos())

    def handle_auto_save_triggered(self, checked):
        self.auto_save = checked
        self.auto_save_changed.emit(checked)
        if checked:
            logger.debug("Auto save enabled")
        else:
            logger.debug("Auto save disabled")

    def handle_continuous_generation_triggered(self, checked):
        self.continuous_generation = checked
        if checked:
            logger.debug("Continuous generation enabled")
        else:
            logger.debug("Continuous generation disabled")
