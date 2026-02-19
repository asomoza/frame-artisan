import attr


@attr.s(eq=True)
class DirectoriesObject:
    data_path = attr.ib(type=str)
    models_diffusers = attr.ib(type=str)
    models_loras = attr.ib(type=str)
    models_controlnets = attr.ib(type=str)
    outputs_videos = attr.ib(type=str)
    outputs_source_images = attr.ib(type=str)
    outputs_source_videos = attr.ib(type=str)
    outputs_source_audio = attr.ib(type=str)
    cache_path = attr.ib(type=str)
    temp_path = attr.ib(type=str)
