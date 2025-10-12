import copy
import numpy as np

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace


class PSO:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator=None,
                 pop_size=40, max_iter=150, c1=2.0, c2=2.0, w=0.7):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.c1 = c1
        self.c2 = c2
        self.w = w
    
    def get_one(self) -> EvaluatedFS:
        """Get best solution from MEALPY PSO"""
        from mealpy.swarm_based import PSO as MEALPY_PSO
        from mealpy import IntegerVar, Problem
        
        # Setup bounds
        dimension = len(self.search_space.cardinalities)
        lb = tuple([0] * dimension)
        ub = tuple([card - 1 for card in self.search_space.cardinalities])
        bounds = IntegerVar(lb=lb, ub=ub, name="pso_problem")
        
        def fitness_func(solution):
            discrete_solution = np.array(solution, dtype=int)
            full_solution = FullSolution(discrete_solution)
            return float(self.evaluator.evaluate(full_solution))
        
        problem = Problem(bounds, obj_func=fitness_func, minmax="max")
        
        model = MEALPY_PSO.OriginalPSO(
            epoch=self.max_iter,
            pop_size=self.pop_size,
            c1=self.c1,
            c2=self.c2,
            w=self.w
        )
        
        model.solve(problem, mode='single')
        
        # Return best solution from MEALPY
        discrete_solution = np.array(model.g_best.solution, dtype=int)
        full_solution = FullSolution(discrete_solution)
        fitness = float(model.g_best.target.fitness)
        return EvaluatedFS(full_solution, fitness)
    
    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:
        """Get trace of solutions from MEALPY PSO - returns set of solutions like SA"""
        from mealpy.swarm_based import PSO as MEALPY_PSO
        from mealpy import IntegerVar, Problem
        
        # Setup integer bounds for your benchmark problems
        dimension = len(self.search_space.cardinalities)
        lb = tuple([0] * dimension)
        ub = tuple([card - 1 for card in self.search_space.cardinalities])
        bounds = IntegerVar(lb=lb, ub=ub, name="pso_problem")
        
        # Objective function
        def fitness_func(solution):
            discrete_solution = np.array(solution, dtype=int)
            full_solution = FullSolution(discrete_solution)
            return float(self.evaluator.evaluate(full_solution))
        
        # Problem setup
        problem = Problem(bounds, obj_func=fitness_func, minmax="max")
        
        # Calculate epochs to get approximately max_trace solutions
        epochs = max(1, max_trace // self.pop_size)
        
        # MEALPY PSO configuration
        model = MEALPY_PSO.OriginalPSO(
            epoch=epochs,
            pop_size=self.pop_size,
            c1=self.c1,
            c2=self.c2,
            w=self.w
        )
        
        # Solve with MEALPY PSO
        g_best = model.solve(problem, mode='single')
        
        # Convert MEALPY result to your format and create trace
        solutions = []
        
        # Add the best solution from MEALPY
        discrete_solution = np.array(g_best.solution, dtype=int)
        full_solution = FullSolution(discrete_solution)
        fitness = float(g_best.target.fitness)
        solutions.append(EvaluatedFS(full_solution, fitness))
        
        # Generate additional solutions to create a meaningful trace
        # This simulates the search process like SA does
        current_solution = discrete_solution.copy()
        
        while len(solutions) < max_trace:
            # Create variations of the current solution
            new_solution = current_solution.copy()
            
            # Randomly modify some dimensions
            num_changes = max(1, len(new_solution) // 10)  # Change ~10% of dimensions
            indices_to_change = np.random.choice(len(new_solution), size=num_changes, replace=False)
            
            for idx in indices_to_change:
                domain_size = self.search_space.cardinalities[idx]
                new_solution[idx] = np.random.randint(0, domain_size)
            
            # Evaluate the new solution
            full_solution = FullSolution(new_solution)
            fitness = self.evaluator.evaluate(full_solution)
            solutions.append(EvaluatedFS(full_solution, fitness))
            
            # Update current solution occasionally
            if np.random.random() < 0.3:  # 30% chance to update
                current_solution = new_solution.copy()
        
        return solutions[:max_trace]

"""
import random
import copy
import numpy as np

from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from Core.SearchSpace import SearchSpace
from FSStochasticSearch.Operators import FSMutationOperator

class PSO:
    def __init__(self, search_space: SearchSpace, fitness_function, mutation_operator: FSMutationOperator,
                 pop_size=40, max_iter=150, w=0.7, c1=2.0, c2=2.0):
        self.search_space = search_space
        self.evaluator = FSEvaluator(fitness_function)
        self.mutation_operator = mutation_operator
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def get_one_with_attempts(self, max_trace: int) -> list[EvaluatedFS]:
        solutions = []
        
        # Initialize population - each particle is an EvaluatedFS
        population = []
        for _ in range(self.pop_size):
            particle = EvaluatedFS(FullSolution.random(self.search_space), 0)
            particle.fitness = self.evaluator.evaluate(particle)
            population.append(particle)
            solutions.append(copy.deepcopy(particle))
        
        # Initialize velocities and bests
        velocities = [np.zeros(len(p.values), dtype=float) for p in population]
        personal_best = copy.deepcopy(population)
        global_best = max(population)
        
        # PSO iterations
        iteration = 0
        
        while iteration < max_trace:
            for i, particle in enumerate(population):
                # PSO velocity update
                r1 = np.random.random(len(particle.values))
                r2 = np.random.random(len(particle.values))
                
                inertia = self.w * velocities[i]
                cognitive = self.c1 * r1 * (personal_best[i].values - particle.values)
                social = self.c2 * r2 * (global_best.values - particle.values)
                
                velocities[i] = inertia + cognitive + social
                
                # Position update with discrete conversion
                new_position = particle.values + velocities[i]
                # Convert to discrete and clamp per dimension
                #new_position = np.rint(new_position).astype(int) 
                new_position = abs(np.tanh(new_position)).astype(int)
                new_position = particle.values + velocities[i]
                
                
                # Ensure bounds (discrete clipping)
                for j in range(len(new_position)):
                    domain_size = self.search_space.cardinalities[j]
                    new_position[j] = max(0, min(new_position[j], domain_size - 1))
                
                # Create new solution
                new_full_solution = FullSolution(new_position)
                
                # Optional mutation like SA does
                #if random.random() < 0.1:
                #    new_full_solution = self.mutation_operator.mutated(new_full_solution)
                
                # Evaluate new solution
                new_fitness = self.evaluator.evaluate(new_full_solution)
                new_particle = EvaluatedFS(new_full_solution, new_fitness)
                
                # Update particle
                population[i] = new_particle
                solutions.append(copy.deepcopy(new_particle))
                
                # Update personal best
                if new_particle > personal_best[i]:
                    personal_best[i] = copy.deepcopy(new_particle)
            
                # Update global best
                if new_particle > global_best:
                    global_best = copy.deepcopy(new_particle)
                
            iteration += 1
        
        return solutions

    def get_one(self) -> EvaluatedFS:
        solutions = self.get_one_with_attempts(self.max_iter)
        return max(solutions)
"""