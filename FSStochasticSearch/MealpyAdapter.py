"""
Adapter class to integrate mealpy optimization algorithms with the existing problem structure.
This allows using mealpy's IntegerVar with your SearchSpace and fitness functions.
"""

import numpy as np
from mealpy import IntegerVar, PSO, GA, SA, WOA, GWO, BBO, BRO, CRO, ABC, HHO, SMO

from Core.SearchSpace import SearchSpace
from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator


class MealpyAdapter:

    def __init__(self, search_space: SearchSpace, fitness_function, evaluator: FSEvaluator = None):

        self.search_space = search_space
        self.fitness_function = fitness_function
        self.evaluator = evaluator or FSEvaluator(fitness_function)
        
        # Convert SearchSpace to mealpy IntegerVar format
        # Each dimension has bounds [0, cardinality-1]
        lower_bounds = [0] * search_space.dimensions
        upper_bounds = [card - 1 for card in search_space.cardinalities]
        
        self.mealpy_bounds = IntegerVar(
            lb=lower_bounds,
            ub=upper_bounds,
            name="solution_vars"
        )
    
    def objective_function(self, solution):
        # Convert mealpy solution to FullSolution
        full_solution = FullSolution(solution)
        
        # Evaluate using your existing evaluator
        return self.evaluator.evaluate(full_solution)
    
    def create_problem(self, minmax="max"):

        return {
            "obj_func": self.objective_function,
            "bounds": self.mealpy_bounds,
            "minmax": minmax,
        }
    
    def run_pso(self, epoch=100, pop_size=40, w=0.7, c1=2.0, c2=2.0, minmax="min"):

        problem = self.create_problem(minmax)
        
        # Create PSO model
        model = PSO.OriginalPSO(epoch=epoch, pop_size=pop_size, w=w, c1=c1, c2=c2)
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def run_ga(self, epoch=100, pop_size=40, pc=0.95, pm=0.025, minmax="min"):
        """
        Run Genetic Algorithm using mealpy.
        
        Args:
            epoch: Number of iterations
            pop_size: Population size
            pc: Crossover probability
            pm: Mutation probability
            minmax: "min" or "max"
            
        Returns:
            Tuple of (best_solution, best_fitness, history)
        """
        problem = self.create_problem(minmax)
        
        # Create GA model
        model = GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=pc, pm=pm)
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def run_sa(self, epoch=100, pop_size=40, max_sub_iter=5, t0=1000, cooling_rate=0.99, minmax="min"):
        """
        Run Simulated Annealing using mealpy.
        
        Args:
            epoch: Number of iterations
            pop_size: Population size
            max_sub_iter: Maximum sub-iterations
            t0: Initial temperature
            cooling_rate: Cooling rate
            minmax: "min" or "max"
            
        Returns:
            Tuple of (best_solution, best_fitness, history)
        """
        problem = self.create_problem(minmax)
        
        # Create SA model
        model = SA.OriginalSA(epoch=epoch, pop_size=pop_size, max_sub_iter=max_sub_iter, 
                         t0=t0, cooling_rate=cooling_rate)
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def run_woa(self, epoch=100, pop_size=40, minmax="min"):
        """
        Run Whale Optimization Algorithm using mealpy.
        
        Args:
            epoch: Number of iterations
            pop_size: Population size
            minmax: "min" or "max"
            
        Returns:
            Tuple of (best_solution, best_fitness, history)
        """
        problem = self.create_problem(minmax)
        
        # Create WOA model
        model = WOA.OriginalWOA(epoch=epoch, pop_size=pop_size)
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def run_gwo(self, epoch=100, pop_size=40, minmax="min"):
        """
        Run Grey Wolf Optimizer using mealpy.
        
        Args:
            epoch: Number of iterations
            pop_size: Population size
            minmax: "min" or "max"
            
        Returns:
            Tuple of (best_solution, best_fitness, history)
        """
        problem = self.create_problem(minmax)
        
        # Create GWO model
        model = GWO.OriginalGWO(epoch=epoch, pop_size=pop_size)
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def run_bbo(self, epoch=100, pop_size=40, minmax="max"):
        """
        Run Biogeography-Based Optimization using mealpy.
        
        Args:
            epoch: Number of iterations
            pop_size: Population size
            minmax: "min" or "max"
            
        Returns:
            Tuple of (best_solution, best_fitness, history)
        """
        problem = self.create_problem(minmax)
        
        # Create BBO model
        model = BBO.OriginalBBO(epoch=epoch, pop_size=pop_size)
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def run_bro(self, epoch=100, pop_size=40, minmax="max"):
        """
        Run Battle Royale Optimization using mealpy.
        
        Args:
            epoch: Number of iterations
            pop_size: Population size
            minmax: "min" or "max"
            
        Returns:
            Tuple of (best_solution, best_fitness, history)
        """
        problem = self.create_problem(minmax)
        
        # Create BRO model
        model = BRO.OriginalBRO(epoch=epoch, pop_size=pop_size, threshold=3)
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def run_cro(self, epoch=100, pop_size=40, minmax="max"):
        """
        Run Coral Reef Optimization using mealpy.
        
        Args:
            epoch: Number of iterations
            pop_size: Population size
            minmax: "min" or "max"
            
        Returns:
            Tuple of (best_solution, best_fitness, history)
        """
        problem = self.create_problem(minmax)
        
        # Create CRO model
        model = CRO.OriginalCRO(epoch=epoch, pop_size=pop_size)
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def run_abc(self, epoch=100, pop_size=40, minmax="max"):
        """
        Run Artificial Bee Colony using mealpy.
        
        Args:
            epoch: Number of iterations
            pop_size: Population size
            minmax: "min" or "max"
            
        Returns:
            Tuple of (best_solution, best_fitness, history)
        """
        problem = self.create_problem(minmax)
        
        # Create ABC model
        model = ABC.OriginalABC(
            epoch=epoch, 
            pop_size=pop_size,
            couple_bees=16,
            patch_variables=5,
            sites=3,
            e_bees=4
        )
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def run_hho(self, epoch=100, pop_size=40, minmax="max"):
        """
        Run Harris Hawks Optimization using mealpy.
        
        Args:
            epoch: Number of iterations
            pop_size: Population size
            minmax: "min" or "max"
            
        Returns:
            Tuple of (best_solution, best_fitness, history)
        """
        problem = self.create_problem(minmax)
        
        # Create HHO model
        model = HHO.OriginalHHO(epoch=epoch, pop_size=pop_size)
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def run_smo(self, epoch=100, pop_size=40, minmax="max"):
        """
        Run Spider Monkey Optimization using mealpy.
        
        Args:
            epoch: Number of iterations
            pop_size: Population size
            minmax: "min" or "max"
            
        Returns:
            Tuple of (best_solution, best_fitness, history)
        """
        problem = self.create_problem(minmax)
        
        # Create SMO model
        model = SMO.OriginalSMO(epoch=epoch, pop_size=pop_size)
        
        # Solve the problem
        g_best = model.solve(problem)
        
        # Convert result back to your format
        best_solution = FullSolution(g_best.solution)
        best_fitness = g_best.target.fitness
        
        return best_solution, best_fitness, model.history
    
    def get_available_algorithms(self):
        """
        Get list of available algorithms in mealpy.
        
        Returns:
            List of algorithm names
        """
        return [
            "PSO", "GA", "SA", "WOA", "GWO", "BBO", "BRO", "CRO", "ABC", "HHO", "SMO"
        ]
