import random
import copy
import numpy as np
import math

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator

class BRO:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator: FSMutationOperator,
                 pop_size=40, max_iter=150, elimination_rate=0.1, survival_rate=0.3):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.elimination_rate = elimination_rate  # Percentage of worst individuals to eliminate
        self.survival_rate = survival_rate  # Percentage of best individuals to keep

    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:
        solutions = []
        
        # Initialize population (players)
        players = []
        for _ in range(self.pop_size):
            player = EvaluatedFS(FullSolution.random(self.search_space), 0)
            player.fitness = self.evaluator.evaluate(player)
            players.append(player)
            solutions.append(copy.deepcopy(player))
        
        iteration = 0
        
        while iteration < max_trace and len(players) > 1:
            # Sort players by fitness (descending - higher is better)
            players.sort(key=lambda x: x.fitness, reverse=True)
            
            # Calculate elimination and survival counts
            elimination_count = max(1, int(len(players) * self.elimination_rate))
            survival_count = max(1, int(len(players) * self.survival_rate))
            
            # Eliminate worst players
            eliminated_players = players[-elimination_count:]
            players = players[:-elimination_count]
            
            # Keep best players (survivors)
            survivors = players[:survival_count]
            
            # Generate new players to replace eliminated ones
            new_players = []
            for _ in range(elimination_count):
                if len(survivors) > 0:
                    # Create new player based on survivors
                    new_player = self.create_new_player(survivors)
                    new_players.append(new_player)
                    solutions.append(copy.deepcopy(new_player))
                else:
                    # Create random player if no survivors
                    new_player = EvaluatedFS(FullSolution.random(self.search_space), 0)
                    new_player.fitness = self.evaluator.evaluate(new_player)
                    new_players.append(new_player)
                    solutions.append(copy.deepcopy(new_player))
            
            # Update population
            players = survivors + new_players
            
            # Apply mutation to some players
            for i, player in enumerate(players):
                if random.random() < 0.1:  # 10% mutation rate
                    mutated_solution = self.mutation_operator.mutated(player.full_solution)
                    mutated_fitness = self.evaluator.evaluate(mutated_solution)
                    mutated_player = EvaluatedFS(mutated_solution, mutated_fitness)
                    
                    if mutated_player > player:
                        players[i] = mutated_player
                        solutions.append(copy.deepcopy(mutated_player))
            
            iteration += 1
        
        return solutions

    def create_new_player(self, survivors):
        """Create new player based on survivors using crossover and mutation"""
        if len(survivors) < 2:
            # If only one survivor, create variation of it
            parent = survivors[0]
            new_values = parent.values.copy()
            
            # Modify some dimensions
            for i in range(len(new_values)):
                if random.random() < 0.3:  # 30% chance to modify each dimension
                    new_values[i] = random.randint(0, self.search_space.cardinalities[i] - 1)
            
            new_solution = FullSolution(new_values)
        else:
            # Crossover between two random survivors
            parent1, parent2 = random.sample(survivors, 2)
            new_values = []
            
            for i in range(len(parent1.values)):
                if random.random() < 0.5:
                    new_values.append(parent1.values[i])
                else:
                    new_values.append(parent2.values[i])
            
            new_solution = FullSolution(new_values)
        
        # Apply mutation
        if random.random() < 0.2:  # 20% chance for additional mutation
            new_solution = self.mutation_operator.mutated(new_solution)
        
        new_fitness = self.evaluator.evaluate(new_solution)
        return EvaluatedFS(new_solution, new_fitness)

    def get_one(self) -> EvaluatedFS:
        solutions = self.get_one_with_attempts(self.max_iter)
        return max(solutions)
