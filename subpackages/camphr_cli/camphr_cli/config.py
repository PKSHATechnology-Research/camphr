from dataclasses import dataclass
from typing import Any, Dict, Optional

from camphr.models import NLPConfig


@dataclass
class DataConfig:
    path: str
    ndata: int
    val_size: float


@dataclass
class SchedulerConfig:
    class_: str
    params: Optional[Dict[str, Any]] = None


@dataclass
class TrainInnerConfig:
    data: DataConfig
    niter: int
    nbatch: int
    # TODO: more type
    optimizer: Optional[Dict[str, Any]] = None
    scheduler: Optional[SchedulerConfig] = None


@dataclass
class TrainConfig:
    train: TrainInnerConfig
    model: NLPConfig
    seed: Optional[int] = None
    user_config: Optional[str] = None
