import random
import copy
import numpy as np

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator

class ACO:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator: FSMutationOperator,
                 pop_size=40, max_iter=150, alpha=1.0, beta=2.0, rho=0.5, q0=0.9):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.alpha = alpha  # Pheromone importance
        self.beta = beta    # Heuristic information importance
        self.rho = rho      # Pheromone evaporation rate
        self.q0 = q0        # Exploitation vs exploration threshold
        
        # Initialize pheromone matrix
        self.pheromone_matrix = self.initialize_pheromone_matrix()

    def initialize_pheromone_matrix(self):
        """Initialize pheromone matrix with equal values"""
        matrix = {}
        for i in range(len(self.search_space.cardinalities)):
            matrix[i] = {}
            for j in range(self.search_space.cardinalities[i]):
                matrix[i][j] = 1.0
        return matrix

    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:
        solutions = []
        
        # Initialize best solution
        best_solution = EvaluatedFS(FullSolution.random(self.search_space), 0)
        best_solution.fitness = self.evaluator.evaluate(best_solution)
        solutions.append(copy.deepcopy(best_solution))
        
        iteration = 0
        
        while iteration < max_trace:
            # Generate solutions using ant colony
            ant_solutions = []
            
            for _ in range(self.pop_size):
                # Construct solution using ACO
                ant_solution = self.construct_solution()
                ant_fitness = self.evaluator.evaluate(ant_solution)
                ant_evaluated = EvaluatedFS(ant_solution, ant_fitness)
                ant_solutions.append(ant_evaluated)
                solutions.append(copy.deepcopy(ant_evaluated))
                
                # Update best solution
                if ant_evaluated > best_solution:
                    best_solution = ant_evaluated
            
            # Update pheromone trails
            self.update_pheromone_trails(ant_solutions, best_solution)
            
            iteration += 1
        
        return solutions

    def construct_solution(self):
        """Construct a solution using ACO"""
        solution_values = []
        
        for i in range(len(self.search_space.cardinalities)):
            # Calculate probabilities for each possible value
            probabilities = self.calculate_probabilities(i)
            
            # Choose value based on probabilities
            if random.random() < self.q0:
                # Exploitation: choose best value
                chosen_value = max(range(len(probabilities)), key=lambda x: probabilities[x])
            else:
                # Exploration: choose based on probabilities
                chosen_value = np.random.choice(len(probabilities), p=probabilities)
            
            solution_values.append(chosen_value)
        
        return FullSolution(solution_values)

    def calculate_probabilities(self, dimension):
        """Calculate probabilities for choosing values in a dimension"""
        probabilities = []
        total = 0
        
        for j in range(self.search_space.cardinalities[dimension]):
            # Calculate heuristic information (1/distance or fitness-based)
            heuristic = 1.0  # Simplified heuristic
            
            # Calculate probability
            prob = (self.pheromone_matrix[dimension][j] ** self.alpha) * (heuristic ** self.beta)
            probabilities.append(prob)
            total += prob
        
        # Normalize probabilities
        if total > 0:
            probabilities = [p / total for p in probabilities]
        else:
            probabilities = [1.0 / len(probabilities)] * len(probabilities)
        
        return probabilities

    def update_pheromone_trails(self, solutions, best_solution):
        """Update pheromone trails based on solutions"""
        # Evaporate pheromone
        for i in range(len(self.search_space.cardinalities)):
            for j in range(self.search_space.cardinalities[i]):
                self.pheromone_matrix[i][j] *= (1 - self.rho)
        
        # Deposit pheromone for each solution
        for solution in solutions:
            fitness = solution.fitness
            if fitness > 0:  # Only deposit if fitness is positive
                pheromone_amount = fitness / 100.0  # Normalize pheromone amount
                
                for i, value in enumerate(solution.values):
                    self.pheromone_matrix[i][value] += pheromone_amount
        
        # Deposit extra pheromone for best solution
        if best_solution.fitness > 0:
            extra_pheromone = best_solution.fitness / 50.0
            for i, value in enumerate(best_solution.values):
                self.pheromone_matrix[i][value] += extra_pheromone

    def get_one(self) -> EvaluatedFS:
        solutions = self.get_one_with_attempts(self.max_iter)
        return max(solutions)
