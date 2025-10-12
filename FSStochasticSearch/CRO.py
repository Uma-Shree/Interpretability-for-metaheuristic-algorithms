import random
import copy
import numpy as np

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator

class CRO:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator: FSMutationOperator,
                 pop_size=40, max_iter=150, reef_size=50, broadcast_prob=0.6, 
                 depredation_prob=0.1, asexual_reproduction_prob=0.5):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.reef_size = reef_size
        self.broadcast_prob = broadcast_prob
        self.depredation_prob = depredation_prob
        self.asexual_reproduction_prob = asexual_reproduction_prob

    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:
        solutions = []
        
        # Initialize reef (coral population)
        reef = []
        for _ in range(self.reef_size):
            coral = EvaluatedFS(FullSolution.random(self.search_space), 0)
            coral.fitness = self.evaluator.evaluate(coral)
            reef.append(coral)
            solutions.append(copy.deepcopy(coral))
        
        iteration = 0
        
        while iteration < max_trace:
            # Sort reef by fitness (descending)
            reef.sort(key=lambda x: x.fitness, reverse=True)
            
            # Broadcast spawning (sexual reproduction)
            if random.random() < self.broadcast_prob:
                reef = self.broadcast_spawning(reef, solutions)
            
            # Brooding (asexual reproduction)
            if random.random() < self.asexual_reproduction_prob:
                reef = self.brooding(reef, solutions)
            
            # Larvae settlement
            reef = self.larvae_settlement(reef, solutions)
            
            # Depredation (remove worst corals)
            if random.random() < self.depredation_prob:
                reef = self.depredation(reef)
            
            # Keep only the best corals
            reef = reef[:self.reef_size]
            
            iteration += 1
        
        return solutions

    def broadcast_spawning(self, reef, solutions):
        """Broadcast spawning - sexual reproduction between best corals"""
        new_reef = reef.copy()
        
        # Select best corals for reproduction
        num_parents = min(10, len(reef) // 2)
        parents = reef[:num_parents]
        
        for i in range(0, len(parents), 2):
            if i + 1 < len(parents):
                parent1 = parents[i]
                parent2 = parents[i + 1]
                
                # Create offspring through crossover
                offspring = self.crossover(parent1, parent2)
                offspring.fitness = self.evaluator.evaluate(offspring)
                solutions.append(copy.deepcopy(offspring))
                
                # Add to reef if there's space
                if len(new_reef) < self.reef_size:
                    new_reef.append(offspring)
        
        return new_reef

    def brooding(self, reef, solutions):
        """Brooding - asexual reproduction"""
        new_reef = reef.copy()
        
        # Select corals for asexual reproduction
        num_brooders = min(5, len(reef))
        brooders = reef[:num_brooders]
        
        for brooder in brooders:
            # Create offspring through mutation
            offspring = EvaluatedFS(self.mutation_operator.mutated(brooder.full_solution), 0)
            offspring.fitness = self.evaluator.evaluate(offspring)
            solutions.append(copy.deepcopy(offspring))
            
            # Add to reef if there's space
            if len(new_reef) < self.reef_size:
                new_reef.append(offspring)
        
        return new_reef

    def larvae_settlement(self, reef, solutions):
        """Larvae settlement - add new random corals"""
        new_reef = reef.copy()
        
        # Add new random corals
        num_larvae = min(5, self.reef_size - len(reef))
        for _ in range(num_larvae):
            larva = EvaluatedFS(FullSolution.random(self.search_space), 0)
            larva.fitness = self.evaluator.evaluate(larva)
            solutions.append(copy.deepcopy(larva))
            new_reef.append(larva)
        
        return new_reef

    def depredation(self, reef):
        """Depredation - remove worst corals"""
        if len(reef) > 5:
            # Remove worst 20% of corals
            num_to_remove = max(1, len(reef) // 5)
            return reef[:-num_to_remove]
        return reef

    def crossover(self, parent1, parent2):
        """Crossover between two parents"""
        new_values = []
        for i in range(len(parent1.values)):
            if random.random() < 0.5:
                new_values.append(parent1.values[i])
            else:
                new_values.append(parent2.values[i])
        
        return EvaluatedFS(FullSolution(new_values), 0)

    def get_one(self) -> EvaluatedFS:
        solutions = self.get_one_with_attempts(self.max_iter)
        return max(solutions)
