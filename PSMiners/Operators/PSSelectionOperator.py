import heapq
import random
from typing import Optional, Iterable

from Core.EvaluatedPS import EvaluatedPS


class PSSelectionOperator:
    def __init__(self):
        pass

    def __repr__(self):
        raise Exception("An implementation of PSSelectionOperator does not implement __repr__")


    def select_n(self, n: int, pool: list[EvaluatedPS]) -> list[EvaluatedPS]:
        raise Exception(f"An implementation of PSSelectionOperator({self.__repr__()}) does not implement select_n")





class TruncationSelection(PSSelectionOperator):
    on_aggregated_score: bool
    on_metric: bool
    which_metric = Optional[int]
    def __init__(self, on_aggregated_score = True, which_metric = None):
        if on_aggregated_score:
            self.on_aggregated_score = True
            self.on_metric = False
            self.which_metric = None
        elif which_metric is not None:
            self.on_aggregated_score = False
            self.on_metric = True
            self.which_metric = which_metric
        else:
            raise Exception(f"The requested truncation selection method is invalid ({on_aggregated_score =}, {which_metric = }")
        super().__init__()

    def __repr__(self):
        return "TruncationSelection"

    def select_n(self, n: int, pool: list[EvaluatedPS]) -> list[EvaluatedPS]:
        if self.on_aggregated_score:
            return heapq.nlargest(n=n, iterable = pool)
        else:
            return heapq.nlargest(n=n, iterable=pool, key=lambda x: x.metric_scores[self.which_metric])


class TournamentSelection(PSSelectionOperator):
    tournament_size: int

    def __init__(self, tournament_size=2):
        self.tournament_size = tournament_size
        super().__init__()

    def __repr__(self):
        return f"TournamentSelection(tournament = {self.tournament_size})"


    def select_n(self, n: int, pool: list[EvaluatedPS]) -> list[EvaluatedPS]:
        def select_one():
            return max(random.choices(pool, k=self.tournament_size))  # isn't this beautiful? God bless comparable objects

        return [select_one() for _ in range(n)]


class AlternatingSelection(PSSelectionOperator):
    """This is similar to lexicase selection, in a way"""
    selection_methods: list[PSSelectionOperator]
    def __init__(self, selection_methods: Iterable[PSSelectionOperator]):
        self.selection_methods = list(selection_methods)
        super().__init__()


    def __repr__(self):
        return "AlternatingSelection("+(", ".join(f"{s}" for s in self.selection_methods)) + ")"


    def select_n(self, n: int, pool: list[EvaluatedPS]) -> list[EvaluatedPS]:
        which_selection_method: PSSelectionOperator = random.choice(self.selection_methods)
        return which_selection_method.select_n(n, pool)



    @classmethod
    def for_each_metric(cls, amount_of_metrics: int):
        return cls([TruncationSelection(which_metric=i) for i in range(amount_of_metrics)])


