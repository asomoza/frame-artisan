import attr


@attr.s(slots=True)
class ModelDataObject:
    name: str = attr.ib(default="")
    version: str = attr.ib(default="")
    filepath: str = attr.ib(default="")
    model_type: int = attr.ib(default=0)
    id: int = attr.ib(default=0)

    @classmethod
    def get_column_names(cls) -> list[str]:
        """Gets the column names from the ModelDataObject class."""
        return [field.name for field in attr.fields(cls)]

    @classmethod
    def from_dict(cls, data):
        """Creates a ModelDataObject from a dictionary."""
        valid_keys = {field.name for field in attr.fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in valid_keys})

    def to_dict(self):
        """Converts the ModelDataObject to a dictionary."""
        return attr.asdict(self)
