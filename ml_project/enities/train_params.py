from dataclasses import dataclass, field

from .model_params import ModelParams


@dataclass()
class TrainParams:
    model_type: str
    model_params: ModelParams
