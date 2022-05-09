from dataclasses import dataclass
from typing import List, Optional


@dataclass()
class FeatureParams:
    categorical_features: List[str]
    numerical_features: List[str]
    features_to_drop: Optional[List[str]]
    target_col: str
