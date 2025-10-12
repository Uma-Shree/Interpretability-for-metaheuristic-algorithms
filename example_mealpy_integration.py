"""
Example script demonstrating how to use mealpy with IntegerVar 
for your existing problem structure.
"""

import numpy as np
from Core.SearchSpace import SearchSpace
from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS
from Core.FSEvaluator import FSEvaluator
from FSStochasticSearch.MealpyAdapter import MealpyAdapter

# Example fitness function (you can replace this with your actual fitness function)
def example_fitness_function(full_solution: FullSolution) -> float:
    """
    Example fitness function - you can replace this with your actual problem.
    This is a simple function that we want to maximize.
    """
    # Example: sum of squares (to maximize)
    return float(np.sum(full_solution.values ** 2))

def one_max_fitness(full_solution: FullSolution) -> float:
    """
    OneMax problem - count the number of 1s in the solution.
    """
    return float(np.sum(full_solution.values))

def main():
    """
    Main function demonstrating mealpy integration.
    """
    print("=== Mealpy Integration Example ===")
    
    # Create a search space (example: 10 dimensions with cardinalities 2,3,4,5,6,7,8,9,10,11)
    cardinalities = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    search_space = SearchSpace(cardinalities)
    
    print(f"Search space: {search_space}")
    print(f"Dimensions: {search_space.dimensions}")
    print(f"Cardinalities: {search_space.cardinalities}")
    
    # Create evaluator
    evaluator = FSEvaluator(example_fitness_function)
    
    # Create mealpy adapter
    adapter = MealpyAdapter(search_space, example_fitness_function, evaluator)
    
    print("\n=== Testing Different Algorithms ===")
    
    # Test PSO
    print("\n--- PSO Algorithm ---")
    try:
        best_solution, best_fitness, history = adapter.run_pso(
            epoch=50, 
            pop_size=30, 
            w=0.7, 
            c1=2.0, 
            c2=2.0, 
            minmax="max"
        )
        print(f"Best solution: {best_solution}")
        print(f"Best fitness: {best_fitness}")
        print(f"Final fitness: {history.list_global_best_fit[-1]}")
    except Exception as e:
        print(f"PSO failed: {e}")
    
    # Test GA
    print("\n--- GA Algorithm ---")
    try:
        best_solution, best_fitness, history = adapter.run_ga(
            epoch=50, 
            pop_size=30, 
            pc=0.95, 
            pm=0.025, 
            minmax="max"
        )
        print(f"Best solution: {best_solution}")
        print(f"Best fitness: {best_fitness}")
        print(f"Final fitness: {history.list_global_best_fit[-1]}")
    except Exception as e:
        print(f"GA failed: {e}")
    
    # Test WOA
    print("\n--- WOA Algorithm ---")
    try:
        best_solution, best_fitness, history = adapter.run_woa(
            epoch=50, 
            pop_size=30, 
            minmax="max"
        )
        print(f"Best solution: {best_solution}")
        print(f"Best fitness: {best_fitness}")
        print(f"Final fitness: {history.list_global_best_fit[-1]}")
    except Exception as e:
        print(f"WOA failed: {e}")
    
    # Test GWO
    print("\n--- GWO Algorithm ---")
    try:
        best_solution, best_fitness, history = adapter.run_gwo(
            epoch=50, 
            pop_size=30, 
            minmax="max"
        )
        print(f"Best solution: {best_solution}")
        print(f"Best fitness: {best_fitness}")
        print(f"Final fitness: {history.list_global_best_fit[-1]}")
    except Exception as e:
        print(f"GWO failed: {e}")
    
    print("\n=== Available Algorithms ===")
    print("Available algorithms:", adapter.get_available_algorithms())
    
    print("\n=== Comparison with Your Original PSO ===")
    # You can also compare with your original PSO implementation
    from FSStochasticSearch.PSO import PSO
    from FSStochasticSearch.Operators import FSMutationOperator
    
    # Use the existing SinglePointFSMutation operator
    from FSStochasticSearch.Operators import SinglePointFSMutation
    
    mutation_operator = SinglePointFSMutation(search_space, probability=0.1)
    
    # Run your original PSO
    print("\n--- Original PSO ---")
    try:
        original_pso = PSO(
            search_space=search_space,
            fitness_function=example_fitness_function,
            mutation_operator=mutation_operator,
            pop_size=30,
            max_iter=50,
            w=0.7,
            c1=2.0,
            c2=2.0
        )
        
        solutions = original_pso.get_one_with_attempts(50)
        best_original = max(solutions)
        print(f"Best solution: {best_original}")
        print(f"Best fitness: {best_original.fitness}")
    except Exception as e:
        print(f"Original PSO failed: {e}")

if __name__ == "__main__":
    main()
