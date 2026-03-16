from __future__ import annotations

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QWidget

from frameartisan.modules.generation.lora.lora_added_item import LoraAddedItem
from frameartisan.modules.generation.panels.base_panel import BasePanel


class LoraPanel(BasePanel):
    def __init__(self, *args):
        super().__init__(*args)

        self._items: dict[int, LoraAddedItem] = {}  # DB id → widget

        self.init_ui()

        self.event_bus.subscribe("lora", self.on_lora_event)
        self.event_bus.subscribe("graph_cleared", self._on_graph_cleared)

    def init_ui(self):
        main_layout = QVBoxLayout()

        add_lora_button = QPushButton("Add LoRA")
        add_lora_button.clicked.connect(self.open_lora_dialog)
        main_layout.addWidget(add_lora_button)

        added_loras_widget = QWidget()
        self.loras_layout = QVBoxLayout(added_loras_widget)
        main_layout.addWidget(added_loras_widget)

        main_layout.addStretch()
        self.setLayout(main_layout)

    def open_lora_dialog(self):
        self.event_bus.publish("manage_dialog", {"dialog_type": "lora_manager", "action": "open"})

    def _add_item(
        self,
        lora_id: int,
        lora_hash: str,
        name: str,
        filepath: str = "",
        weight: float = 1.0,
        enabled: bool = True,
        video_strength: float = 1.0,
        audio_strength: float = 1.0,
        granular_transformer_weights_enabled: bool = False,
        granular_transformer_weights: dict | None = None,
        is_slider: bool = False,
        apply_to_second_stage: bool = False,
        spatial_mask_enabled: bool = False,
        spatial_mask_path: str = "",
        temporal_mask_enabled: bool = False,
        temporal_start_frame: int = 0,
        temporal_end_frame: int = -1,
        temporal_fade_in_frames: int = 0,
        temporal_fade_out_frames: int = 0,
    ):
        item = LoraAddedItem(
            lora_id,
            lora_hash,
            name,
            weight,
            enabled,
            video_strength=video_strength,
            audio_strength=audio_strength,
            granular_transformer_weights_enabled=granular_transformer_weights_enabled,
            granular_transformer_weights=granular_transformer_weights,
            is_slider=is_slider,
            apply_to_second_stage=apply_to_second_stage,
            spatial_mask_enabled=spatial_mask_enabled,
            spatial_mask_path=spatial_mask_path,
            temporal_mask_enabled=temporal_mask_enabled,
            temporal_start_frame=temporal_start_frame,
            temporal_end_frame=temporal_end_frame,
            temporal_fade_in_frames=temporal_fade_in_frames,
            temporal_fade_out_frames=temporal_fade_out_frames,
        )
        item.enabled_changed.connect(self._on_item_changed)
        item.remove_clicked.connect(self._on_item_remove)

        # Store filepath on the item for sync
        item.filepath = filepath

        self._items[lora_id] = item
        # Insert before the stretch
        self.loras_layout.insertWidget(self.loras_layout.count() - 1, item)

    def _remove_item(self, lora_id: int):
        item = self._items.pop(lora_id, None)
        if item is not None:
            self.loras_layout.removeWidget(item)
            item.setParent(None)
            item.deleteLater()

    def _on_item_changed(self, _id: int, _value):
        self._sync_to_graph()

    def _on_item_remove(self, lora_id: int):
        self._remove_item(lora_id)
        self._sync_to_graph()

    def _sync_to_graph(self):
        configs = []
        for lora_id, item in self._items.items():
            cfg = item.get_config()
            cfg["filepath"] = getattr(item, "filepath", "")
            configs.append(cfg)

        self.event_bus.publish("generation_change", {"attr": "active_loras", "value": configs})

    def _on_graph_cleared(self, _data: dict) -> None:
        for lora_id in list(self._items):
            self._remove_item(lora_id)

    #########################################################
    ## SUBSCRIBED BUS EVENTS
    #########################################################
    def on_lora_event(self, data: dict):
        action = data.get("action")
        if action == "add":
            lora_id = data.get("id")
            if not lora_id or lora_id in self._items:
                return
            self._add_item(
                lora_id=lora_id,
                lora_hash=data.get("hash", ""),
                name=data.get("name", ""),
                filepath=data.get("filepath", ""),
                weight=data.get("weight", 1.0),
                enabled=data.get("enabled", True),
                video_strength=data.get("video_strength", 1.0),
                audio_strength=data.get("audio_strength", 1.0),
                granular_transformer_weights_enabled=data.get("granular_transformer_weights_enabled", False),
                granular_transformer_weights=data.get("granular_transformer_weights"),
                is_slider=data.get("is_slider", False),
                apply_to_second_stage=data.get("apply_to_second_stage", False),
                spatial_mask_enabled=data.get("spatial_mask_enabled", False),
                spatial_mask_path=data.get("spatial_mask_path", ""),
                temporal_mask_enabled=data.get("temporal_mask_enabled", False),
                temporal_start_frame=data.get("temporal_start_frame", 0),
                temporal_end_frame=data.get("temporal_end_frame", -1),
                temporal_fade_in_frames=data.get("temporal_fade_in_frames", 0),
                temporal_fade_out_frames=data.get("temporal_fade_out_frames", 0),
            )
            self._sync_to_graph()
        elif action == "remove":
            lora_id = data.get("id")
            if lora_id is not None:
                self._remove_item(lora_id)
            self._sync_to_graph()
        elif action == "update_weights":
            lora_id = data.get("id")
            config = data.get("config", {})
            item = self._items.get(lora_id) if lora_id is not None else None
            if item is not None:
                item.update_from_dialog(config)
                self._sync_to_graph()
