from dataclasses import dataclass
from omegaconf.listconfig import ListConfig


@dataclass
class DataConfig:
    n_users: int = 500
    n_items: int = 1000
    rating_scale: ListConfig = ListConfig([1, 5])
    epsilon: float = 0.5
    time_range: int = 40


@dataclass(frozen=True)
class MFConfig:
    n_factors: int = 100
    lr: float = 0.1
    reg: float = 0.5
    n_epochs: int = 10
