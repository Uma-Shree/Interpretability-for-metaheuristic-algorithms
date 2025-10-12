import random

from Core.EvaluatedPS import EvaluatedPS
from Core.PS import PS


class PSCrossoverOperator:
    def __init__(self):
        pass

    def __repr__(self):
        raise Exception("An implementatin of PSCrossoverOperator does not implement __repr__")


    def crossed(self, mother: EvaluatedPS, father: EvaluatedPS) -> EvaluatedPS:
        raise Exception("An implementation of PSCrossoverOperator does not implement .crossed")



class SinglePointCrossover(PSCrossoverOperator):

    def __init__(self):
        super().__init__()

    def __repr__(self):
        return f"SinglePointCrossover()"

    def crossed(self, mother: PS, father: PS) -> PS:
        crossing_point = random.randrange(len(mother))
        new_values = list(mother.values)[:crossing_point] + list(father.values)[:crossing_point]
        return PS(new_values)