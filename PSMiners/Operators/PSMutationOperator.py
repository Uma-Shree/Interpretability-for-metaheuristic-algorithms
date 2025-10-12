import random
from typing import Optional

import numpy as np

from Core.PS import PS, STAR
from Core.SearchSpace import SearchSpace


class PSMutationOperator:
    search_space: Optional[SearchSpace]

    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space

    def random_value(self, var_index: int) -> int:
        return random.randrange(self.search_space.cardinalities[var_index])

    def __repr__(self):
        return "PSMutationOperator"

    def mutated(self, ps: PS) -> PS:
        raise Exception("An implementation of PSMutationOperator does not implement .mutated")


class SinglePointMutation(PSMutationOperator):
    mutation_probability: float
    chance_of_unfixing: float

    def __init__(self,
                 probability: float,
                 chance_of_unfixing: float,
                 search_space: SearchSpace):
        self.mutation_probability = probability
        self.chance_of_unfixing = chance_of_unfixing
        super().__init__(search_space)

    def __repr__(self):
        return f"SinglePointMutation(unfixing_rate = {self.chance_of_unfixing:.2f})"

    def mutated(self, ps: PS) -> PS:
        def get_mutated_value_for(index: int):
            if ps.values[index] == STAR:
                return self.random_value(index)
            else:
                if random.random() < self.chance_of_unfixing:
                    return STAR
                else:
                    return self.random_value(index)

        new_values = np.copy(ps.values)
        for index in range(len(new_values)):
            if random.random() < self.mutation_probability:
                new_values[index] = get_mutated_value_for(index)

        return PS(new_values)


class MultimodalMutationOperator(PSMutationOperator):
    value_mutation_chance: float

    def __init__(self,
                 value_mutation_chance: float,
                 search_space: SearchSpace):
        self.value_mutation_chance = value_mutation_chance
        super().__init__(search_space)

    def __repr__(self):
        return f"MultimodalMutationOperator(value_mutation_chance = {self.value_mutation_chance})"

    def mutate_which_are_fixed(self, ps: PS) -> PS:
        new_values = np.copy(ps.values)
        mutation_probability = 1 / len(ps)
        for index, value in enumerate(ps.values):
            if random.random() < mutation_probability:
                new_values[index] = STAR if value != STAR else self.random_value(index)
        return PS(new_values)

    def mutate_values(self, ps: PS) -> PS:
        if ps.is_empty():
            return ps
        new_values = np.copy(ps.values)
        fixed_positions = ps.get_fixed_variable_positions()
        mutation_probability = 1 / len(fixed_positions)
        for index in fixed_positions:
            if random.random() < mutation_probability:
                new_values[index] = self.random_value(index)
        return PS(new_values)

    def mutated(self, ps: PS) -> PS:
        if random.random() < self.value_mutation_chance:
            return self.mutate_values(ps)
        else:
            return self.mutate_which_are_fixed(ps)
