import random
import copy
import numpy as np

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator

class ABC:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator: FSMutationOperator,
                 pop_size=40, max_iter=150, limit=50, employed_bees_ratio=0.5):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.limit = limit  # Limit for abandoning food sources
        self.employed_bees_ratio = employed_bees_ratio
        self.employed_bees_count = int(pop_size * employed_bees_ratio)
        self.onlooker_bees_count = pop_size - self.employed_bees_count

    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:
        solutions = []
        
        # Initialize population (food sources)
        food_sources = []
        trial_counts = []  # Track trials for each food source
        
        for _ in range(self.employed_bees_count):
            individual = EvaluatedFS(FullSolution.random(self.search_space), 0)
            individual.fitness = self.evaluator.evaluate(individual)
            food_sources.append(individual)
            trial_counts.append(0)
            solutions.append(copy.deepcopy(individual))
        
        iteration = 0
        
        while iteration < max_trace:
            # Employed bees phase
            for i in range(self.employed_bees_count):
                # Generate new solution by modifying current food source
                new_solution = self.generate_new_solution(food_sources, i)
                new_fitness = self.evaluator.evaluate(new_solution)
                new_individual = EvaluatedFS(new_solution, new_fitness)
                solutions.append(copy.deepcopy(new_individual))
                
                # Greedy selection
                if new_individual > food_sources[i]:
                    food_sources[i] = new_individual
                    trial_counts[i] = 0
                else:
                    trial_counts[i] += 1
            
            # Calculate probabilities for onlooker bees
            fitness_values = [fs.fitness for fs in food_sources]
            if sum(fitness_values) > 0:
                probabilities = [f / sum(fitness_values) for f in fitness_values]
            else:
                probabilities = [1.0 / len(food_sources)] * len(food_sources)
            
            # Onlooker bees phase
            for _ in range(self.onlooker_bees_count):
                # Select food source based on probability
                selected_idx = np.random.choice(len(food_sources), p=probabilities)
                
                # Generate new solution
                new_solution = self.generate_new_solution(food_sources, selected_idx)
                new_fitness = self.evaluator.evaluate(new_solution)
                new_individual = EvaluatedFS(new_solution, new_fitness)
                solutions.append(copy.deepcopy(new_individual))
                
                # Greedy selection
                if new_individual > food_sources[selected_idx]:
                    food_sources[selected_idx] = new_individual
                    trial_counts[selected_idx] = 0
                else:
                    trial_counts[selected_idx] += 1
            
            # Scout bees phase - abandon exhausted food sources
            for i in range(self.employed_bees_count):
                if trial_counts[i] >= self.limit:
                    # Replace with random solution
                    food_sources[i] = EvaluatedFS(FullSolution.random(self.search_space), 0)
                    food_sources[i].fitness = self.evaluator.evaluate(food_sources[i])
                    trial_counts[i] = 0
                    solutions.append(copy.deepcopy(food_sources[i]))
            
            iteration += 1
        
        return solutions

    def generate_new_solution(self, food_sources, current_idx):
        """Generate new solution by modifying current food source"""
        current = food_sources[current_idx]
        
        # Select a different food source for modification
        other_indices = [i for i in range(len(food_sources)) if i != current_idx]
        if not other_indices:
            other_idx = current_idx
        else:
            other_idx = random.choice(other_indices)
        
        other = food_sources[other_idx]
        
        # Create new solution by modifying one dimension
        new_values = current.values.copy()
        dimension = random.randint(0, len(current.values) - 1)
        
        # Modify the selected dimension
        phi = random.uniform(-1, 1)
        new_value = int(current.values[dimension] + phi * (current.values[dimension] - other.values[dimension]))
        
        # Ensure bounds
        domain_size = self.search_space.cardinalities[dimension]
        new_value = max(0, min(new_value, domain_size - 1))
        new_values[dimension] = new_value
        
        return FullSolution(new_values)

    def get_one(self) -> EvaluatedFS:
        solutions = self.get_one_with_attempts(self.max_iter)
        return max(solutions)
