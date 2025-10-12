import numpy as np

from Core.PRef import PRef
from Core.PS import PS
from Core.PSMetric.FitnessQuality.SignificantlyHighAverage import MannWhitneyU
from Core.PSMetric.Metric import Metric
from VarianceDecisionTree.SplitVariance import SplitVariance


class SplitVarianceAndConsistency(Metric):
    selected_pRef: PRef
    split_variance_metric: SplitVariance
    consistency_metric: MannWhitneyU

    last_results: (float, float)

    def __init__(self,
                 selected_pRef: PRef):
        self.selected_pRef = selected_pRef
        self.split_variance_metric = SplitVariance(selected_pRef)
        self.consistency_metric = MannWhitneyU()  # we don't need to initialise this...
        super().__init__()


    def evaluate(self, ps: PS) -> None:
        try:
            matching_fitnesses, not_matching_fitnesses = self.selected_pRef.fitnesses_of_observations_and_complement(ps)
            split_variance = self.split_variance_metric.get_weighted_variance(matching_fitnesses, not_matching_fitnesses)

            # Ensure split_variance is a float
            split_variance = float(split_variance) if split_variance is not None else 0.0

            if ps.is_empty():
                consistency = 1.0
            else:
                try:
                    consistency = self.consistency_metric.get_p_value(matching_fitnesses, not_matching_fitnesses)
                    consistency = float(consistency) if consistency is not None else 1.0
                except Exception:
                    consistency = 1.0

            self.last_results = (split_variance, consistency)
        except Exception as e:
            self.last_results = (0.0, 1.0)

    def get_split_variance(self, ps: PS) -> float:
        return self.last_results[0]

    def get_consistency(self, ps: PS) -> float:
        return self.last_results[1]