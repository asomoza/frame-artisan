import attr


@attr.s(eq=True, slots=True)
class PreferencesObject:
    save_video_metadata: bool = attr.ib(default=False)
    hide_nsfw: bool = attr.ib(default=True)
    delete_lora_on_import: bool = attr.ib(default=False)
    delete_model_on_import: bool = attr.ib(default=False)
    save_source_images: bool = attr.ib(default=False)
    save_source_audio: bool = attr.ib(default=False)
    save_source_video: bool = attr.ib(default=False)
    auto_save_videos: bool = attr.ib(default=False)
