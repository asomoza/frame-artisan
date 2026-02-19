import attr


@attr.s(slots=True)
class SchedulerDataObject:
    name: str = attr.ib(default="Euler")
    scheduler_index: int = attr.ib(default=0)
    scheduler_class: str = attr.ib(default="FlowMatchEulerDiscreteScheduler")
    num_train_timesteps: int = attr.ib(default=1000)
    shift: float = attr.ib(default=3.0)
    use_dynamic_shifting: bool = attr.ib(default=False)
    base_shift: float = attr.ib(default=0.5)
    max_shift: float = attr.ib(default=1.15)
    base_image_seq_len: int = attr.ib(default=256)
    max_image_seq_len: int = attr.ib(default=4096)
    invert_sigmas: bool = attr.ib(default=False)
    shift_terminal: float = attr.ib(default=None)
    use_karras_sigmas: bool = attr.ib(default=False)
    use_exponential_sigmas: bool = attr.ib(default=False)
    use_beta_sigmas: bool = attr.ib(default=False)
    time_shift_type: str = attr.ib(default="exponential")
    stochastic_sampling: bool = attr.ib(default=False)

    def reset_to_defaults(self):
        """Resets all attributes of the SchedulerDataObject to their default values."""
        self.name = "Euler"
        self.scheduler_index = 0
        self.scheduler_class = "FlowMatchEulerDiscreteScheduler"
        self.num_train_timesteps = 1000
        self.shift = 3.0
        self.use_dynamic_shifting = False
        self.base_shift = 0.5
        self.max_shift = 1.15
        self.base_image_seq_len = 256
        self.max_image_seq_len = 4096
        self.invert_sigmas = False
        self.shift_terminal = None
        self.use_karras_sigmas = False
        self.use_exponential_sigmas = False
        self.use_beta_sigmas = False
        self.time_shift_type = "exponential"
        self.stochastic_sampling = False

    def to_dict(self):
        """Converts the SchedulerDataObject to a dictionary."""
        return attr.asdict(self)

    @classmethod
    def from_dict(cls, data):
        """Creates a SchedulerDataObject from a dictionary."""
        return cls(**data)

    def update_from_dict(self, data):
        """Updates the SchedulerDataObject from a dictionary."""
        for key, value in data.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def to_config_dict(self):
        """Converts the SchedulerDataObject to a dictionary suitable for from_config()."""
        config_dict = {
            "num_train_timesteps": self.num_train_timesteps,
            "shift": self.shift,
            "use_dynamic_shifting": self.use_dynamic_shifting,
            "base_shift": self.base_shift,
            "max_shift": self.max_shift,
            "base_image_seq_len": self.base_image_seq_len,
            "max_image_seq_len": self.max_image_seq_len,
            "invert_sigmas": self.invert_sigmas,
            "shift_terminal": self.shift_terminal,
            "use_karras_sigmas": self.use_karras_sigmas,
            "use_exponential_sigmas": self.use_exponential_sigmas,
            "use_beta_sigmas": self.use_beta_sigmas,
            "time_shift_type": self.time_shift_type,
            "stochastic_sampling": self.stochastic_sampling,
        }

        return config_dict
