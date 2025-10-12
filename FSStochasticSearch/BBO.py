import random
import copy
import numpy as np

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator

class BBO:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator: FSMutationOperator,
                 pop_size=40, max_iter=150, max_immigration_rate=1.0, max_emigration_rate=1.0, 
                 mutation_probability=0.01, elitism_count=2):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.max_immigration_rate = max_immigration_rate
        self.max_emigration_rate = max_emigration_rate
        self.mutation_probability = mutation_probability
        self.elitism_count = elitism_count

    def calculate_immigration_emigration_rates(self, population):
        """Calculate immigration and emigration rates based on fitness"""
        # Sort population by fitness (descending)
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        immigration_rates = []
        emigration_rates = []
        
        for i in range(len(population)):
            # Immigration rate decreases with fitness rank
            immigration_rate = self.max_immigration_rate * (1 - i / (len(population) - 1))
            # Emigration rate increases with fitness rank
            emigration_rate = self.max_emigration_rate * (i / (len(population) - 1))
            
            immigration_rates.append(immigration_rate)
            emigration_rates.append(emigration_rate)
        
        return immigration_rates, emigration_rates

    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:
        solutions = []
        
        # Initialize population
        population = []
        for _ in range(self.pop_size):
            individual = EvaluatedFS(FullSolution.random(self.search_space), 0)
            individual.fitness = self.evaluator.evaluate(individual)
            population.append(individual)
            solutions.append(copy.deepcopy(individual))
        
        iteration = 0
        
        while iteration < max_trace:
            # Calculate immigration and emigration rates
            immigration_rates, emigration_rates = self.calculate_immigration_emigration_rates(population)
            
            new_population = []
            
            for i, habitat in enumerate(population):
                new_habitat = copy.deepcopy(habitat)
                
                # Immigration: migrate features from other habitats
                if random.random() < immigration_rates[i]:
                    # Select emigrating habitat based on emigration rates
                    emigration_probs = np.array(emigration_rates)
                    emigration_probs[i] = 0  # Can't emigrate to itself
                    if emigration_probs.sum() > 0:
                        emigration_probs = emigration_probs / emigration_probs.sum()
                        emigrating_idx = np.random.choice(len(population), p=emigration_probs)
                        
                        # Migrate features
                        for j in range(len(habitat.values)):
                            if random.random() < 0.5:  # 50% chance to migrate each feature
                                new_habitat.values[j] = population[emigrating_idx].values[j]
                
                # Mutation
                if random.random() < self.mutation_probability:
                    mutated_fs = self.mutation_operator.mutated(new_habitat.as_full_solution())
                    new_habitat = EvaluatedFS(mutated_fs, 0)
                    new_habitat.fitness = self.evaluator.evaluate(mutated_fs)
                
                # Evaluate new habitat
                if new_habitat.fitness == 0:  # If not evaluated yet
                    new_habitat.fitness = self.evaluator.evaluate(new_habitat)
                
                new_population.append(new_habitat)
                solutions.append(copy.deepcopy(new_habitat))
            
            # Elitism: keep best individuals
            population = sorted(new_population, key=lambda x: x.fitness, reverse=True)
            if self.elitism_count > 0:
                # Keep best individuals from previous generation
                best_previous = sorted(solutions[-self.pop_size:], key=lambda x: x.fitness, reverse=True)
                for i in range(min(self.elitism_count, len(best_previous))):
                    if i < len(population) and best_previous[i] > population[i]:
                        population[i] = best_previous[i]
            
            iteration += 1
        
        return solutions

    def get_one(self) -> EvaluatedFS:
        solutions = self.get_one_with_attempts(self.max_iter)
        return max(solutions)
