import random
import copy
import numpy as np

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator

class SMO:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator: FSMutationOperator,
                 pop_size=40, max_iter=150, local_leader_limit=10, global_leader_limit=10,
                 perturbation_rate=0.1, num_groups=4):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.local_leader_limit = local_leader_limit
        self.global_leader_limit = global_leader_limit
        self.perturbation_rate = perturbation_rate
        self.num_groups = num_groups
        
        # Initialize counters
        self.local_leader_limit_count = 0
        self.global_leader_limit_count = 0

    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:
        solutions = []
        
        # Initialize population (spider monkeys)
        monkeys = []
        for _ in range(self.pop_size):
            monkey = EvaluatedFS(FullSolution.random(self.search_space), 0)
            monkey.fitness = self.evaluator.evaluate(monkey)
            monkeys.append(monkey)
            solutions.append(copy.deepcopy(monkey))
        
        # Find global leader
        global_leader = max(monkeys)
        
        # Divide population into groups
        groups = self.divide_into_groups(monkeys)
        local_leaders = [max(group) for group in groups]
        
        iteration = 0
        
        while iteration < max_trace:
            # Global leader phase
            new_monkeys = []
            for monkey in monkeys:
                new_monkey = self.global_leader_phase(monkey, global_leader)
                new_monkey.fitness = self.evaluator.evaluate(new_monkey)
                new_monkeys.append(new_monkey)
                solutions.append(copy.deepcopy(new_monkey))
            
            monkeys = new_monkeys
            
            # Update global leader
            new_global_leader = max(monkeys)
            if new_global_leader > global_leader:
                global_leader = new_global_leader
                self.global_leader_limit_count = 0
            else:
                self.global_leader_limit_count += 1
            
            # Local leader phase for each group
            groups = self.divide_into_groups(monkeys)
            for group_idx, group in enumerate(groups):
                local_leader = max(group)
                new_group = []
                
                for monkey in group:
                    new_monkey = self.local_leader_phase(monkey, local_leader, global_leader)
                    new_monkey.fitness = self.evaluator.evaluate(new_monkey)
                    new_group.append(new_monkey)
                    solutions.append(copy.deepcopy(new_monkey))
                
                groups[group_idx] = new_group
                
                # Update local leader
                new_local_leader = max(new_group)
                if new_local_leader > local_leader:
                    local_leaders[group_idx] = new_local_leader
                    self.local_leader_limit_count = 0
                else:
                    self.local_leader_limit_count += 1
            
            # Global leader decision phase
            if self.global_leader_limit_count >= self.global_leader_limit:
                self.global_leader_limit_count = 0
                # Global leader learns from local leaders
                for group in groups:
                    for monkey in group:
                        if random.random() < self.perturbation_rate:
                            new_monkey = self.perturbation_phase(monkey, global_leader)
                            new_monkey.fitness = self.evaluator.evaluate(new_monkey)
                            solutions.append(copy.deepcopy(new_monkey))
                            if new_monkey > monkey:
                                monkey = new_monkey
            
            # Local leader decision phase
            if self.local_leader_limit_count >= self.local_leader_limit:
                self.local_leader_limit_count = 0
                # Local leaders learn from global leader
                for group_idx, group in enumerate(groups):
                    local_leader = max(group)
                    for monkey in group:
                        if random.random() < self.perturbation_rate:
                            new_monkey = self.perturbation_phase(monkey, global_leader)
                            new_monkey.fitness = self.evaluator.evaluate(new_monkey)
                            solutions.append(copy.deepcopy(new_monkey))
                            if new_monkey > monkey:
                                monkey = new_monkey
            
            iteration += 1
        
        return solutions

    def divide_into_groups(self, monkeys):
        """Divide monkeys into groups"""
        groups = []
        group_size = len(monkeys) // self.num_groups
        
        for i in range(self.num_groups):
            start_idx = i * group_size
            if i == self.num_groups - 1:
                end_idx = len(monkeys)
            else:
                end_idx = (i + 1) * group_size
            groups.append(monkeys[start_idx:end_idx])
        
        return groups

    def global_leader_phase(self, monkey, global_leader):
        """Global leader phase - update position based on global leader"""
        new_values = []
        for i in range(len(monkey.values)):
            r1 = random.random()
            r2 = random.random()
            r3 = random.random()
            
            if r3 >= 0.5:
                # Update based on global leader
                new_value = int(monkey.values[i] + r1 * (global_leader.values[i] - monkey.values[i]) + 
                              r2 * (monkey.values[i] - global_leader.values[i]))
            else:
                # Random update
                new_value = int(monkey.values[i] + random.randint(-1, 1))
            
            # Ensure bounds
            domain_size = self.search_space.cardinalities[i]
            new_value = max(0, min(new_value, domain_size - 1))
            new_values.append(new_value)
        
        return EvaluatedFS(FullSolution(new_values), 0)

    def local_leader_phase(self, monkey, local_leader, global_leader):
        """Local leader phase - update position based on local leader"""
        new_values = []
        for i in range(len(monkey.values)):
            r1 = random.random()
            r2 = random.random()
            
            if r2 >= 0.5:
                # Update based on local leader
                new_value = int(monkey.values[i] + r1 * (local_leader.values[i] - monkey.values[i]) + 
                              random.random() * (global_leader.values[i] - monkey.values[i]))
            else:
                # Update based on global leader
                new_value = int(monkey.values[i] + r1 * (global_leader.values[i] - monkey.values[i]) + 
                              random.random() * (local_leader.values[i] - monkey.values[i]))
            
            # Ensure bounds
            domain_size = self.search_space.cardinalities[i]
            new_value = max(0, min(new_value, domain_size - 1))
            new_values.append(new_value)
        
        return EvaluatedFS(FullSolution(new_values), 0)

    def perturbation_phase(self, monkey, global_leader):
        """Perturbation phase - random update"""
        new_values = []
        for i in range(len(monkey.values)):
            new_value = int(monkey.values[i] + random.randint(-2, 2))
            
            # Ensure bounds
            domain_size = self.search_space.cardinalities[i]
            new_value = max(0, min(new_value, domain_size - 1))
            new_values.append(new_value)
        
        return EvaluatedFS(FullSolution(new_values), 0)

    def get_one(self) -> EvaluatedFS:
        solutions = self.get_one_with_attempts(self.max_iter)
        return max(solutions)
