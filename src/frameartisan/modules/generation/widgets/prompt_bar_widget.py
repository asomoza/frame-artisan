from __future__ import annotations

import random

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QVBoxLayout, QWidget
from transformers import GemmaTokenizer

from frameartisan.app.event_bus import EventBus
from frameartisan.buttons.generate_button import GenerateButton
from frameartisan.modules.generation.widgets.prompt_input_widget import PromptInputWidget


class PromptBarWidget(QWidget):
    generate_clicked = pyqtSignal(object, str, str, bool, bool, bool)
    abort_clicked = pyqtSignal()
    continue_stage2_clicked = pyqtSignal()
    retry_stage1_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.event_bus = EventBus()

        self.previous_positive_prompt = None
        self.previous_negative_prompt = None
        self.previous_seed = None
        self.use_random_seed = True

        self.tokenizer = GemmaTokenizer.from_pretrained("./configs/GemmaTokenizer")
        self.max_tokens = 1024

        self.init_ui()

        self.event_bus.subscribe("json_graph", self.on_json_graph_event)

        self.set_button_generate()

    def init_ui(self) -> None:
        main_layout = QHBoxLayout()
        text_layout = QVBoxLayout()

        positive_prompt_layout = QGridLayout()
        self.positive_prompt = PromptInputWidget(True, 0, self.max_tokens)
        self.positive_prompt.text_changed.connect(self.on_prompt_changed)
        positive_prompt_layout.addWidget(self.positive_prompt, 0, 0)
        text_layout.addLayout(positive_prompt_layout)

        negative_prompt_layout = QGridLayout()
        self.negative_prompt = PromptInputWidget(False, 0, self.max_tokens)
        self.negative_prompt.text_changed.connect(self.on_prompt_changed)
        negative_prompt_layout.addWidget(self.negative_prompt, 0, 0)
        text_layout.addLayout(negative_prompt_layout)

        text_layout.setStretch(0, 2)
        text_layout.setStretch(1, 1)

        main_layout.addLayout(text_layout)

        actions_layout = QVBoxLayout()

        seed_widget = QWidget()
        seed_widget.setMaximumWidth(180)
        seed_layout = QVBoxLayout(seed_widget)
        seed_horizontal_layout = QHBoxLayout()
        seed_label = QLabel("Seed:")
        seed_horizontal_layout.addWidget(seed_label)
        self.seed_text = QLineEdit()
        self.seed_text.setDisabled(True)
        seed_horizontal_layout.addWidget(self.seed_text)
        seed_layout.addLayout(seed_horizontal_layout)
        random_checkbox_layout = QHBoxLayout()
        self.random_checkbox = QCheckBox("Randomize seed")
        self.random_checkbox.setChecked(True)
        self.random_checkbox.clicked.connect(self.randomize_clicked)
        random_checkbox_layout.addWidget(self.random_checkbox)
        random_checkbox_layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        seed_layout.addLayout(random_checkbox_layout)
        actions_layout.addWidget(seed_widget)

        self.generate_button = GenerateButton()
        self.generate_button.setMaximumWidth(180)
        self.generate_button.clicked.connect(self.on_generate_clicked)
        actions_layout.addWidget(self.generate_button)

        self.retry_button = QPushButton("Retry")
        self.retry_button.setMaximumWidth(80)
        self.retry_button.setVisible(False)
        self.retry_button.clicked.connect(self.retry_stage1_clicked.emit)
        actions_layout.addWidget(self.retry_button)

        actions_layout.setStretch(0, 1)
        actions_layout.setStretch(1, 1)
        main_layout.addLayout(actions_layout)

        self._stage1_preview_mode = False

        main_layout.setStretch(0, 10)
        main_layout.setStretch(1, 2)

        self.setLayout(main_layout)

    def randomize_clicked(self):
        if self.random_checkbox.isChecked():
            self.seed_text.setDisabled(True)
            self.use_random_seed = True
        else:
            self.seed_text.setDisabled(False)
            self.use_random_seed = False

    def on_prompt_changed(self):
        prompt = self.sender()
        text = prompt.toPlainText()

        tokens = self.tokenizer(text).input_ids[1:-1]
        num_tokens = len(tokens)

        prompt.update_token_count(num_tokens)

    def set_stage1_preview_mode(self, active: bool) -> None:
        """Toggle the prompt bar between preview mode and normal mode."""
        self._stage1_preview_mode = active
        self.retry_button.setVisible(active)
        if active:
            self.generate_button.setStyleSheet(
                """
                QPushButton {
                    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2266cc, stop: 1 #144080);
                }
                QPushButton:hover {
                    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2b7ee0, stop: 1 #1a5299);
                }
                QPushButton:pressed {
                    background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #3399ff, stop: 1 #2266b3);
                }
                """
            )
            self.generate_button.setText("Continue to Stage 2")
        else:
            self.set_button_generate()

    def on_generate_clicked(self) -> None:
        if self._stage1_preview_mode:
            self.continue_stage2_clicked.emit()
            return
        if self.generate_button.text() == "Abort":
            self.abort_clicked.emit()
            return

        positive_prompt = self.positive_prompt.toPlainText()
        negative_prompt = self.negative_prompt.toPlainText()

        positive_prompt_changed = self.previous_positive_prompt != positive_prompt

        if positive_prompt_changed:
            self.previous_positive_prompt = positive_prompt

        negative_prompt_changed = self.previous_negative_prompt != negative_prompt
        if negative_prompt_changed:
            self.previous_negative_prompt = negative_prompt

        if self.use_random_seed:
            seed = random.randint(0, 2**32 - 1)
            self.seed_text.setText(str(seed))
        else:
            try:
                seed = int(self.seed_text.text())
            except ValueError:
                self.event_bus.publish("show_snackbar", {"action": "show", "message": "Seed is not valid."})
                return

        seed_changed = self.previous_seed != seed
        if seed_changed:
            self.previous_seed = seed

        self.generate_clicked.emit(
            seed, positive_prompt, negative_prompt, positive_prompt_changed, negative_prompt_changed, seed_changed
        )

    def set_generating(self, generating: bool) -> None:
        if generating:
            self.set_button_abort()
        else:
            self.set_button_generate()

    def set_button_generate(self):
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #23923b, stop: 1 #124e1f);
            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2cb349, stop: 1 #176427);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #35d457, stop: 1 #1f8834);
            }
            """
        )
        self.generate_button.setText("Generate")
        self.generate_button.setShortcut("Ctrl+Return")

    def set_button_abort(self):
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #c42b2b, stop: 1 #861e1e);
            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #db2f2f, stop: 1 #9b2222);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #fa3333, stop: 1 #b62828);
            }
            """
        )
        self.generate_button.setText("Abort")
        self.generate_button.setShortcut("Ctrl+Return")

    def set_button_disable(self):
        self.generate_button.setDisabled(True)
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #8b8989, stop: 1 #5f5f5f);
            }
            """
        )

    def set_button_enable(self):
        self.generate_button.setDisabled(False)
        self.generate_button.setStyleSheet(
            """
            QPushButton {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #23923b, stop: 1 #124e1f);
            }
            QPushButton:hover {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #2cb349, stop: 1 #176427);
            }
            QPushButton:pressed {
                background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #35d457, stop: 1 #1f8834);
            }
            """
        )

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_json_graph_event(self, data):
        action = data.get("action")
        if action == "loaded":
            data = data.get("data", {})

            if "positive_prompt" in data:
                positive_prompt = data["positive_prompt"]
                self.previous_positive_prompt = positive_prompt
                self.positive_prompt.setPlainText(positive_prompt)

            if "negative_prompt" in data:
                negative_prompt = data["negative_prompt"]
                self.previous_negative_prompt = negative_prompt
                self.negative_prompt.setPlainText(negative_prompt)

            if "seed" in data:
                seed = data["seed"]
                self.previous_seed = seed
                self.seed_text.setText(str(seed))
                self.random_checkbox.setChecked(False)
                self.seed_text.setDisabled(False)
                self.use_random_seed = False
