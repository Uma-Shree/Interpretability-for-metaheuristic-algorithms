def pRef_from_PSO(benchmark_problem, sample_size: int, max_trace: int) -> PRef:
    """Generate PRef using MEALPY PSO with proper error handling"""
    from mealpy.swarm_based import PSO as MEALPY_PSO
    from mealpy import IntegerVar, Problem
    import numpy as np
    from Core.FullSolution import FullSolution
    from Core.EvaluatedFS import EvaluatedFS
    from Core.PRef import PRef
    
    print(f"🔄 Running MEALPY PSO on {benchmark_problem.__class__.__name__}")
    
    # Setup integer bounds
    dimension = len(benchmark_problem.search_space.cardinalities)
    lb = tuple([0] * dimension)
    ub = tuple([card - 1 for card in benchmark_problem.search_space.cardinalities])
    bounds = IntegerVar(lb=lb, ub=ub, name="pso_problem")
    
    print(f"   Dimension: {dimension}")
    print(f"   Domain sizes: {benchmark_problem.search_space.cardinalities[:5]}{'...' if len(benchmark_problem.search_space.cardinalities) > 5 else ''}")
    
    # Fitness function wrapper
    def fitness_func(solution):
        discrete_solution = np.array(solution, dtype=int)
        full_solution = FullSolution(discrete_solution)
        return float(benchmark_problem.fitness_function(full_solution))
    
    # FIXED: Use problem_dict instead of Problem object (this is the key fix!)
    problem_dict = {
        "obj_func": fitness_func,
        "bounds": bounds,
        "minmax": "max",
        "save_population": True, 
        "log_to": None
    }
    
    # FIXED: Use proper PSO parameters
    pop_size = 100  # Increase population size
    epochs = max(50, max_trace // pop_size)  # Ensure sufficient epochs
    
    print(f"   PSO parameters: epoch={epochs}, pop_size={pop_size}")
    
    model = MEALPY_PSO.OriginalPSO(
        epoch=epochs,
        pop_size=pop_size,
        c1=2.0,
        c2=2.0,
        w=0.7
    )
    
    # FIXED: Don't try to set model.history.save_population - it's already set in problem_dict
    # model.history.save_population = True  # REMOVED - this causes the error!
    
    # Solve PSO
    model.solve(problem_dict, mode='single')
    
    solutions = []
    
    print(f"   Available history attributes: {dir(model.history)}")
    
    # FIXED: Extract from actual PSO history with proper error handling
    if hasattr(model.history, 'list_population') and model.history.list_population:
        print(f"    Extracting from {len(model.history.list_population)} generations")
        
        for generation_idx, population in enumerate(model.history.list_population):
            if len(solutions) >= max_trace:
                break
                
            for agent in population:
                if len(solutions) >= max_trace:
                    break
                    
                try:
                    discrete_solution = np.array(agent.solution, dtype=int)
                    full_solution = FullSolution(discrete_solution)
                    fitness = float(agent.target.fitness)
                    solutions.append(EvaluatedFS(full_solution, fitness))
                except Exception as e:
                    print(f"    Error processing agent: {e}")
                    continue
    else:
        print("    No list_population in history, using fallback method")
        
        # Fallback: Use final population if history is empty
        if hasattr(model, 'population') and model.population:
            print(f"    Using final population ({len(model.population)} agents)")
            for agent in model.population:
                if len(solutions) >= max_trace:
                    break
                try:
                    discrete_solution = np.array(agent.solution, dtype=int)
                    full_solution = FullSolution(discrete_solution)
                    fitness = float(agent.target.fitness)
                    solutions.append(EvaluatedFS(full_solution, fitness))
                except Exception as e:
                    print(f"    Error processing final agent: {e}")
                    continue
    
    # Trim to max_trace
    solutions = solutions[:max_trace]
    
    # Then select sample_size from collected solutions for final PRef
    if len(solutions) > sample_size:
        # Sort by fitness and take best sample_size solutions
        solutions.sort(key=lambda x: x.fitness, reverse=True)
        solutions = solutions[:sample_size]
    
    print(f"    ✅ Generated {len(solutions)} solutions from MEALPY PSO")
    
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

