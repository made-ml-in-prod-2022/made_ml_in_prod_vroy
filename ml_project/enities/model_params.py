from dataclasses import dataclass, field


@dataclass()
class ModelParams:
    n_neighbors: int
    random_state: int = field(default=17)
