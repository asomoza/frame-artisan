"""Tests for GenerationSettings: video_duration, frame_rate, and compute_num_frames."""

from __future__ import annotations

import tempfile

import pytest
from PyQt6.QtCore import QSettings

from frameartisan.modules.generation.generation_settings import (
    GenerationSettings,
    compute_num_frames,
)


@pytest.fixture()
def qsettings():
    """QSettings backed by a temporary INI file."""
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as f:
        path = f.name
    qs = QSettings(path, QSettings.Format.IniFormat)
    yield qs
    qs.sync()


class TestGenerationSettingsDefaults:
    def test_video_duration_default(self):
        s = GenerationSettings()
        assert s.video_duration == 5

    def test_frame_rate_default(self):
        s = GenerationSettings()
        assert s.frame_rate == 24

    def test_group_offload_use_stream_default(self):
        s = GenerationSettings()
        assert s.group_offload_use_stream is False

    def test_group_offload_low_cpu_mem_default(self):
        s = GenerationSettings()
        assert s.group_offload_low_cpu_mem is False

    def test_existing_defaults_unchanged(self):
        s = GenerationSettings()
        assert s.video_width == 640
        assert s.video_height == 352
        assert s.num_inference_steps == 24
        assert s.guidance_scale == 4.0


class TestGenerationSettingsPersistence:
    def test_video_duration_round_trip(self, qsettings):
        s = GenerationSettings(video_duration=10)
        s.save(qsettings)
        qsettings.sync()

        loaded = GenerationSettings.load(qsettings)
        assert loaded.video_duration == 10

    def test_frame_rate_round_trip(self, qsettings):
        s = GenerationSettings(frame_rate=30)
        s.save(qsettings)
        qsettings.sync()

        loaded = GenerationSettings.load(qsettings)
        assert loaded.frame_rate == 30

    def test_defaults_used_when_key_absent(self, qsettings):
        """Loading from an empty QSettings uses field defaults."""
        loaded = GenerationSettings.load(qsettings)
        assert loaded.video_duration == 5
        assert loaded.frame_rate == 24

    def test_all_new_fields_preserved_together(self, qsettings):
        s = GenerationSettings(video_duration=12, frame_rate=15)
        s.save(qsettings)
        qsettings.sync()

        loaded = GenerationSettings.load(qsettings)
        assert loaded.video_duration == 12
        assert loaded.frame_rate == 15

    def test_group_offload_use_stream_round_trip(self, qsettings):
        s = GenerationSettings(group_offload_use_stream=True)
        s.save(qsettings)
        qsettings.sync()
        loaded = GenerationSettings.load(qsettings)
        assert loaded.group_offload_use_stream is True

    def test_group_offload_low_cpu_mem_round_trip(self, qsettings):
        s = GenerationSettings(group_offload_low_cpu_mem=True)
        s.save(qsettings)
        qsettings.sync()
        loaded = GenerationSettings.load(qsettings)
        assert loaded.group_offload_low_cpu_mem is True

    def test_existing_fields_still_persisted(self, qsettings):
        s = GenerationSettings(video_width=1024, video_height=768, guidance_scale=7.5)
        s.save(qsettings)
        qsettings.sync()

        loaded = GenerationSettings.load(qsettings)
        assert loaded.video_width == 1024
        assert loaded.video_height == 768
        assert loaded.guidance_scale == 7.5


class TestSecondPassSettingsDefaults:
    def test_second_pass_enabled_default(self):
        s = GenerationSettings()
        assert s.second_pass_enabled is False

    def test_second_pass_steps_default(self):
        s = GenerationSettings()
        assert s.second_pass_steps == 10

    def test_second_pass_guidance_default(self):
        s = GenerationSettings()
        assert s.second_pass_guidance == 4.0

    def test_second_pass_model_default(self):
        from frameartisan.modules.generation.data_objects.model_data_object import ModelDataObject

        s = GenerationSettings()
        assert isinstance(s.second_pass_model, ModelDataObject)


class TestSecondPassSettingsPersistence:
    def test_second_pass_enabled_round_trip(self, qsettings):
        s = GenerationSettings(second_pass_enabled=True)
        s.save(qsettings)
        qsettings.sync()
        loaded = GenerationSettings.load(qsettings)
        assert loaded.second_pass_enabled is True

    def test_second_pass_steps_round_trip(self, qsettings):
        s = GenerationSettings(second_pass_steps=20)
        s.save(qsettings)
        qsettings.sync()
        loaded = GenerationSettings.load(qsettings)
        assert loaded.second_pass_steps == 20

    def test_second_pass_guidance_round_trip(self, qsettings):
        s = GenerationSettings(second_pass_guidance=7.5)
        s.save(qsettings)
        qsettings.sync()
        loaded = GenerationSettings.load(qsettings)
        assert loaded.second_pass_guidance == 7.5

    def test_defaults_used_when_keys_absent(self, qsettings):
        loaded = GenerationSettings.load(qsettings)
        assert loaded.second_pass_enabled is False
        assert loaded.second_pass_steps == 10
        assert loaded.second_pass_guidance == 4.0


class TestComputeNumFrames:
    """Tests for the compute_num_frames helper (8n+1 snapping)."""

    @pytest.mark.parametrize(
        "duration, fps, expected",
        [
            (1, 24, 25),  # 24 frames → next 8n+1 is 25 (n=3)
            (5, 24, 121),  # 120 frames → next is 121 (n=15)
            (5, 25, 129),  # 125 frames → next is 129 (n=16)
            (4, 25, 105),  # 100 frames → next is 105 (n=13), not 97
            (10, 24, 241),  # 240 frames → next is 241 (n=30)
            (20, 24, 481),  # 480 frames → next is 481 (n=60)
            (20, 30, 601),  # 600 frames → next is 601 (n=75)
            (1, 8, 9),  # 8 frames → next is 9 (n=1), minimum
        ],
    )
    def test_snapping(self, duration, fps, expected):
        assert compute_num_frames(duration, fps) == expected

    def test_result_satisfies_8n_plus_1(self):
        for dur in range(1, 21):
            for fps in (8, 15, 24, 25, 30):
                result = compute_num_frames(dur, fps)
                assert (result - 1) % 8 == 0, f"dur={dur}, fps={fps}, result={result}"

    def test_result_meets_requested_duration(self):
        """The computed frame count should always produce at least the requested duration."""
        for dur in range(1, 21):
            for fps in (8, 15, 24, 25, 30):
                result = compute_num_frames(dur, fps)
                actual_duration = result / fps
                assert actual_duration >= dur, f"dur={dur}, fps={fps}, frames={result}, actual={actual_duration:.2f}s"

    def test_minimum_is_9(self):
        # Even very short durations should produce at least 9 frames (n=1).
        assert compute_num_frames(1, 1) == 9
