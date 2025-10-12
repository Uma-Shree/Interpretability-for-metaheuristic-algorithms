from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.Metric import Metric


class Averager(Metric):
    metrics: list[Metric]
    weights: list[int]

    def __init__(self, metrics: list[Metric], weights = None):
        super().__init__()
        self.metrics = metrics
        self.used_evaluations = 0
        if weights is None:
            self.weights = [1 for _ in metrics]
        else:
            self.weights = weights

    def get_labels(self) -> list[str]:
        return [m.__repr__() for m in self.metrics]

    def set_pRef(self, pRef: PRef):
        for m in self.metrics:
            m.set_pRef(pRef)

    def __repr__(self):
        return f"Averager({', '.join(self.get_labels())})"

    def get_scores(self, ps: PS) -> list[float]:
        self.used_evaluations += 1
        return [m.get_single_score(ps) for m in self.metrics]

    def get_normalised_scores(self, ps: PS) -> list[float]:
        return [m.get_single_normalised_score(ps) for m in self.metrics]

    def get_amount_of_metrics(self) -> int:
        return len(self.metrics)

    def get_single_normalised_score(self, ps: PS) -> float:
        self.used_evaluations += 1
        return sum([score * weight
                    for score, weight in zip(self.get_normalised_scores(ps), self.weights)]) / sum(self.weights)


    def get_scores_for_debug(self, ps: PS) -> list[float]:
        return [metric.get_single_normalised_score(ps) for metric in self.metrics]