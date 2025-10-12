import random
import copy
import numpy as np
import math

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator

class HHO:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator: FSMutationOperator,
                 pop_size=40, max_iter=150):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter

    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:
        solutions = []
        
        # Initialize population (hawks)
        hawks = []
        for _ in range(self.pop_size):
            hawk = EvaluatedFS(FullSolution.random(self.search_space), 0)
            hawk.fitness = self.evaluator.evaluate(hawk)
            hawks.append(hawk)
            solutions.append(copy.deepcopy(hawk))
        
        # Find best hawk (rabbit)
        rabbit = max(hawks)
        
        iteration = 0
        
        while iteration < max_trace:
            E1 = 2 * (1 - iteration / max_trace)  # Energy decreases from 2 to 0
            
            for i, hawk in enumerate(hawks):
                E0 = 2 * random.random() - 1  # Random energy
                E = 2 * E0 * (1 - iteration / max_trace)  # Escaping energy
                
                # Exploration phase
                if abs(E) >= 1:
                    new_values = self.exploration_phase(hawk.values, hawks)
                # Exploitation phase
                else:
                    r = random.random()
                    if r >= 0.5 and abs(E) < 0.5:
                        # Soft besiege
                        new_values = self.soft_besiege(hawk.values, rabbit.values, E)
                    elif r >= 0.5 and abs(E) >= 0.5:
                        # Hard besiege
                        new_values = self.hard_besiege(hawk.values, rabbit.values, E)
                    elif r < 0.5 and abs(E) < 0.5:
                        # Soft besiege with progressive rapid dives
                        new_values = self.soft_besiege_progressive(hawk.values, rabbit.values, E)
                    else:
                        # Hard besiege with progressive rapid dives
                        new_values = self.hard_besiege_progressive(hawk.values, rabbit.values, E)
                
                # Ensure bounds
                new_values = self.apply_bounds(new_values)
                
                # Evaluate new position
                new_solution = FullSolution(new_values)
                new_fitness = self.evaluator.evaluate(new_solution)
                new_hawk = EvaluatedFS(new_solution, new_fitness)
                solutions.append(copy.deepcopy(new_hawk))
                
                # Update hawk if better
                if new_hawk > hawk:
                    hawks[i] = new_hawk
                
                # Update rabbit
                if new_hawk > rabbit:
                    rabbit = new_hawk
            
            iteration += 1
        
        return solutions

    def exploration_phase(self, current_values, hawks):
        """Exploration phase - random search"""
        if random.random() < 0.5:
            # Random position
            new_values = []
            for i in range(len(current_values)):
                new_value = random.randint(0, self.search_space.cardinalities[i] - 1)
                new_values.append(new_value)
        else:
            # Position based on other hawks
            other_hawks = [h for h in hawks if not np.array_equal(h.values, current_values)]
            if other_hawks:
                rand_hawk = random.choice(other_hawks)
                new_values = []
                for i in range(len(current_values)):
                    new_value = int(rand_hawk.values[i] - random.random() * abs(2 * random.random() * rand_hawk.values[i] - current_values[i]))
                    new_values.append(new_value)
            else:
                new_values = current_values.copy()
        return new_values

    def soft_besiege(self, current_values, rabbit_values, E):
        """Soft besiege strategy"""
        J = 2 * (1 - random.random())  # Jump strength
        new_values = []
        for i in range(len(current_values)):
            new_value = int(rabbit_values[i] - current_values[i] - E * abs(J * rabbit_values[i] - current_values[i]))
            new_values.append(new_value)
        return new_values

    def hard_besiege(self, current_values, rabbit_values, E):
        """Hard besiege strategy"""
        new_values = []
        for i in range(len(current_values)):
            new_value = int(rabbit_values[i] - E * abs(rabbit_values[i] - current_values[i]))
            new_values.append(new_value)
        return new_values

    def soft_besiege_progressive(self, current_values, rabbit_values, E):
        """Soft besiege with progressive rapid dives"""
        J = 2 * (1 - random.random())
        Y = self.soft_besiege(current_values, rabbit_values, E)
        
        # Apply rapid dives
        S = np.random.random(len(current_values))
        Z = []
        for i in range(len(current_values)):
            z_value = int(Y[i] + S[i] * (Y[i] - current_values[i]))
            Z.append(z_value)
        
        # Choose between Y and Z
        if random.random() < 0.5:
            return Y
        else:
            return Z

    def hard_besiege_progressive(self, current_values, rabbit_values, E):
        """Hard besiege with progressive rapid dives"""
        J = 2 * (1 - random.random())
        Y = self.hard_besiege(current_values, rabbit_values, E)
        
        # Apply rapid dives
        S = np.random.random(len(current_values))
        Z = []
        for i in range(len(current_values)):
            z_value = int(Y[i] + S[i] * (Y[i] - current_values[i]))
            Z.append(z_value)
        
        # Choose between Y and Z
        if random.random() < 0.5:
            return Y
        else:
            return Z

    def apply_bounds(self, values):
        """Apply bounds to ensure values are within search space"""
        bounded_values = []
        for i, value in enumerate(values):
            domain_size = self.search_space.cardinalities[i]
            bounded_value = max(0, min(int(value), domain_size - 1))
            bounded_values.append(bounded_value)
        return bounded_values

    def get_one(self) -> EvaluatedFS:
        solutions = self.get_one_with_attempts(self.max_iter)
        return max(solutions)
