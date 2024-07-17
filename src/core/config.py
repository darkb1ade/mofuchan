from dataclasses import dataclass
from .utils import get_dataset_offset


@dataclass(frozen=True)
class Config:
    # directory
    path_out: str
    path_in: str

    # component
    preprocessor: dict
    feature: dict
    data_spliter: dict
    tuner: dict

    # other
    use_scaler: bool

    def __post_init__(self):
        if self.data_spliter["offset"] == "auto":
            self.data_spliter["offset"] = get_dataset_offset(self.feature)

        if self.tuner["offset"] == "auto":
            self.tuner["offset"] = get_dataset_offset(self.feature)
