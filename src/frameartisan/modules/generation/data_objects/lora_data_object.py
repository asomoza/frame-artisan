import attr


@attr.s(slots=True)
class LoraDataObject:
    id: int = attr.ib(default=0)
    name: str = attr.ib(default="")
    filepath: str = attr.ib(default="")
    weight: float = attr.ib(default=1.0)
    enabled: bool = attr.ib(default=True)
    hash: str = attr.ib(default="")
    granular_transformer_weights_enabled: bool = attr.ib(default=False)
    granular_transformer_weights: dict = attr.ib(factory=dict)
    is_slider: bool = attr.ib(default=False)
    video_strength: float = attr.ib(default=1.0)
    audio_strength: float = attr.ib(default=1.0)

    def to_dict(self):
        return attr.asdict(self)

    @classmethod
    def from_dict(cls, data):
        valid_keys = {field.name for field in attr.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})
