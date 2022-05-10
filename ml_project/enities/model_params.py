from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    n_neighbors: int = field(default=5)
    criterion: str = field(default="gini")
    max_depth: int = field(default=7)
    min_samples_split: int = field(default=5)
    random_state: int = field(default=17)
