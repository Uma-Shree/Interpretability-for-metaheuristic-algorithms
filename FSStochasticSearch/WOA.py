import random
import copy
import numpy as np
import math

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator

class WOA:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator: FSMutationOperator,
                 pop_size=40, max_iter=150):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter

    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:
        solutions = []
        
        # Initialize population (whales)
        whales = []
        for _ in range(self.pop_size):
            whale = EvaluatedFS(FullSolution.random(self.search_space), 0)
            whale.fitness = self.evaluator.evaluate(whale)
            whales.append(whale)
            solutions.append(copy.deepcopy(whale))
        
        # Find best whale (leader)
        best_whale = max(whales)
        
        iteration = 0
        
        while iteration < max_trace:
            a = 2 - iteration * (2 / max_trace)  # Linearly decreases from 2 to 0
            
            for i, whale in enumerate(whales):
                r1, r2 = random.random(), random.random()
                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1  # Shape of the spiral
                l = (a - 1) * random.random() + 1
                
                p = random.random()
                
                if p < 0.5:
                    if abs(A) >= 1:
                        # Search for prey (exploration)
                        rand_leader_idx = random.randint(0, len(whales) - 1)
                        rand_leader = whales[rand_leader_idx]
                        
                        new_values = self.update_position_exploration(whale.values, rand_leader.values, A, C)
                    else:
                        # Encircling prey (exploitation)
                        new_values = self.update_position_exploitation(whale.values, best_whale.values, A, C)
                else:
                    # Spiral updating position
                    new_values = self.update_position_spiral(whale.values, best_whale.values, b, l)
                
                # Ensure bounds
                new_values = self.apply_bounds(new_values)
                
                # Evaluate new position
                new_solution = FullSolution(new_values)
                new_fitness = self.evaluator.evaluate(new_solution)
                new_whale = EvaluatedFS(new_solution, new_fitness)
                solutions.append(copy.deepcopy(new_whale))
                
                # Update whale if better
                if new_whale > whale:
                    whales[i] = new_whale
                
                # Update best whale
                if new_whale > best_whale:
                    best_whale = new_whale
            
            iteration += 1
        
        return solutions

    def update_position_exploration(self, current_values, rand_leader_values, A, C):
        """Update position during exploration phase"""
        new_values = []
        for i in range(len(current_values)):
            D = abs(C * rand_leader_values[i] - current_values[i])
            new_value = rand_leader_values[i] - A * D
            new_values.append(int(new_value))
        return new_values

    def update_position_exploitation(self, current_values, best_values, A, C):
        """Update position during exploitation phase"""
        new_values = []
        for i in range(len(current_values)):
            D = abs(C * best_values[i] - current_values[i])
            new_value = best_values[i] - A * D
            new_values.append(int(new_value))
        return new_values

    def update_position_spiral(self, current_values, best_values, b, l):
        """Update position using spiral equation"""
        new_values = []
        for i in range(len(current_values)):
            D = abs(best_values[i] - current_values[i])
            new_value = D * math.exp(b * l) * math.cos(l * 2 * math.pi) + best_values[i]
            new_values.append(int(new_value))
        return new_values

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
