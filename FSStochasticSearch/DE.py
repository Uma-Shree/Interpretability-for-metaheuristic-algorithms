import random
import copy
import numpy as np

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator

class DE:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator: FSMutationOperator,
                 pop_size=40, max_iter=150, F=0.5, CR=0.7):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.F = F  # Differential weight
        self.CR = CR  # Crossover probability

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
            new_population = []
            
            for i, target in enumerate(population):
                # Select three different individuals for mutation
                candidates = [j for j in range(self.pop_size) if j != i]
                a, b, c = random.sample(candidates, 3)
                
                # Mutation: create donor vector
                donor_values = population[a].values.copy()
                for j in range(len(donor_values)):
                    donor_values[j] = population[a].values[j] + self.F * (population[b].values[j] - population[c].values[j])
                    # Ensure bounds
                    domain_size = self.search_space.cardinalities[j]
                    donor_values[j] = max(0, min(int(donor_values[j]), domain_size - 1))
                
                # Crossover: create trial vector
                trial_values = target.values.copy()
                j_rand = random.randint(0, len(target.values) - 1)
                
                for j in range(len(target.values)):
                    if random.random() < self.CR or j == j_rand:
                        trial_values[j] = donor_values[j]
                
                # Evaluate trial vector
                trial_solution = FullSolution(trial_values)
                trial_fitness = self.evaluator.evaluate(trial_solution)
                trial_individual = EvaluatedFS(trial_solution, trial_fitness)
                solutions.append(copy.deepcopy(trial_individual))
                
                # Selection: choose better between target and trial
                if trial_individual > target:
                    new_population.append(trial_individual)
                else:
                    new_population.append(target)
            
            population = new_population
            iteration += 1
        
        return solutions

    def get_one(self) -> EvaluatedFS:
        solutions = self.get_one_with_attempts(self.max_iter)
        return max(solutions)
