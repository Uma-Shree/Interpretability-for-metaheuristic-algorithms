from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core import TerminationCriteria
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef
from BenchmarkProblems.BenchmarkProblem import BenchmarkProblem
from Core import TerminationCriteria
from Core.EvaluatedFS import EvaluatedFS
from Core.PRef import PRef
from FSStochasticSearch.GA import GA
from FSStochasticSearch.Operators import SinglePointFSMutation, TwoPointFSCrossover, TournamentSelection
from FSStochasticSearch.SA import SA
from FSStochasticSearch.TabuSearch import TabuSearch
from FSStochasticSearch.PSO import PSO
from FSStochasticSearch.DE import DE
from FSStochasticSearch.BBO import BBO
from FSStochasticSearch.ABC import ABC
from FSStochasticSearch.BRO import BRO
from FSStochasticSearch.WOA import WOA
from FSStochasticSearch.HHO import HHO
from FSStochasticSearch.SMO import SMO
from FSStochasticSearch.CRO import CRO
from FSStochasticSearch.ACO import ACO
import numpy as np

import utils 
from mealpy.math_based import AOA
from mealpy import FloatVar
from mealpy.swarm_based import PSO as MealpyPSO
from mealpy.bio_based import BBO as MealpyBBO
from mealpy.swarm_based import ABC as MealpyABC
from mealpy.evolutionary_based import CRO as MealpyCRO
from mealpy.human_based import BRO as MealpyBRO
from mealpy.swarm_based import ACOR as MealpyACOR
from mealpy.swarm_based import SSO as MealpySSO

def get_problem_specific_epochs(benchmark_problem, max_trace, pop_size):
    """Get problem-specific epoch configuration"""
    problem_name = benchmark_problem.__class__.__name__
    
    # Base calculation
    base_epochs = max(50, max_trace // pop_size)
    
    if "BT" in problem_name:
        # BT problems need extensive exploration
        if hasattr(benchmark_problem, 'depth'):
            depth = benchmark_problem.depth
            if depth >= 5:
                multiplier = 3.0
            elif depth >= 4:
                multiplier = 2.5
            else:
                multiplier = 2.0
        else:
            multiplier = 2.0
        epochs = int(base_epochs * multiplier)
        
    elif "SAT" in problem_name:
        
        if "SAT_L" in problem_name:
            multiplier = 1.8
        elif "SAT_M" in problem_name:
            multiplier = 1.4
        else:  # SAT_S
            multiplier = 1.0
        epochs = int(base_epochs * multiplier)
        
    elif "GC" in problem_name:
        # Graph coloring needs balanced exploration
        if "GC_L" in problem_name:
            multiplier = 1.6
        else:  # GC_S
            multiplier = 1.2
        epochs = int(base_epochs * multiplier)
        
    else:
        # Default for unknown problems
        epochs = int(base_epochs * 1.5)
    
    # Ensure minimum epochs
    epochs = max(epochs, 100)
    
    return epochs

def pRef_from_PSO(benchmark_problem, sample_size: int, max_trace: int) -> PRef:
   
    
    from mealpy.swarm_based import PSO as MEALPY_PSO
    from mealpy import IntegerVar, Problem
    import numpy as np
    from Core.FullSolution import FullSolution
    from Core.EvaluatedFS import EvaluatedFS
    from Core.PRef import PRef
    
    print(f"Running MEALPY PSO on {benchmark_problem.__class__.__name__}")
    
    # Setup integer bounds
    dimension = len(benchmark_problem.search_space.cardinalities)
    lb = tuple([0] * dimension)
    ub = tuple([card - 1 for card in benchmark_problem.search_space.cardinalities])
    bounds = IntegerVar(lb=lb, ub=ub, name="pso_problem")
    
    # Fitness function wrapper
    def fitness_func(solution):
        discrete_solution = np.array(solution, dtype=int)
        full_solution = FullSolution(discrete_solution)
        return float(benchmark_problem.fitness_function(full_solution))
    
  
    problem_dict = {
        "obj_func": fitness_func,
        "bounds": bounds,
        "minmax": "max",
        "save_population": True, 
        "log_to": None
    }
    
    
    pop_size = 40
    #epochs = max(10, max_trace // pop_size)
    
    print(f"   PSO parameters: pop_size={pop_size}")
    

    model = MEALPY_PSO.OriginalPSO(
        epoch= get_problem_specific_epochs(benchmark_problem, max_trace, pop_size),
        pop_size=pop_size,
        c1=2.0,
        c2=2.0,
        w=0.4
    )
    

    model.solve(problem_dict, mode='single')
    

    solutions = []
    
    print(f"   Available history attributes: {dir(model.history)}")
    
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
        print("    No list_population in history, using other history data")


    solutions = solutions[:max_trace]

    print(f"Generated {len(solutions)} solutions from MEALPY PSO history")
    
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

    
def uniformly_random_distribution_pRef(benchmark_problem: BenchmarkProblem,
                                       sample_size: int) -> PRef:
    return benchmark_problem.get_reference_population(sample_size=sample_size)


def pRef_from_PSO_best(benchmark_problem, sample_size: int, top_n_per_gen: int = 15) -> PRef:
    """PSO that keeps only top N solutions from each generation"""
    
    from mealpy.swarm_based import PSO as MEALPY_PSO
    from mealpy import IntegerVar, Problem
    import numpy as np
    from Core.FullSolution import FullSolution
    from Core.EvaluatedFS import EvaluatedFS
    from Core.PRef import PRef
    
    print(f"__ Running MEALPY PSO (top {top_n_per_gen} per generation) on {benchmark_problem.__class__.__name__}")
    
    # Setup integer bounds
    dimension = len(benchmark_problem.search_space.cardinalities)
    lb = tuple([0] * dimension)
    ub = tuple([card - 1 for card in benchmark_problem.search_space.cardinalities])
    bounds = IntegerVar(lb=lb, ub=ub, name="pso_problem")
    
    def fitness_func(solution):
        discrete_solution = np.array(solution, dtype=int)
        full_solution = FullSolution(discrete_solution)
        return float(benchmark_problem.fitness_function(full_solution))
    
    problem_dict = {
        "obj_func": fitness_func,
        "bounds": bounds,
        "minmax": "max",
        "save_population": True,
        "log_to": None
    }
    
    pop_size = 40
    # Calculate epochs needed
    epochs_needed = max(10, (sample_size // top_n_per_gen) + 5)
    
    print(f"   PSO parameters: pop_size={pop_size}, epochs={epochs_needed}")
    print(f"   Collecting top {top_n_per_gen} solutions per generation")
    
    model = MEALPY_PSO.OriginalPSO(
        epoch=epochs_needed,
        pop_size=pop_size,
        c1=2.0,
        c2=2.0,
        w=0.7
    )
    
    model.solve(problem_dict, mode='single')
    
    # COLLECT TOP N FROM EACH GENERATION
    solutions = []
    
    if hasattr(model.history, 'list_population') and model.history.list_population:
        print(f"    Extracting from {len(model.history.list_population)} generations")
        
        for generation_idx, population in enumerate(model.history.list_population):
         
            gen_solutions = []
            for agent in population:
                try:
                    discrete_solution = np.array(agent.solution, dtype=int)
                    fitness = float(agent.target.fitness)
                    full_solution = FullSolution(discrete_solution)
                    gen_solutions.append(EvaluatedFS(full_solution, fitness))
                except Exception as e:
                    print(f"    Error processing agent: {e}")
                    continue
    
            gen_solutions.sort(key=lambda x: x.fitness, reverse=True)  # Best first
            top_solutions = gen_solutions[:top_n_per_gen]  # Take only top N
            
            solutions.extend(top_solutions)
            
            print(f"    Gen {generation_idx+1}: Added top {len(top_solutions)} solutions "
                  f"(fitness range: [{top_solutions[-1].fitness:.2f}, {top_solutions[0].fitness:.2f}])")
            

            if len(solutions) >= sample_size:
                break
    

    solutions = solutions[:sample_size]
    
    # Show quality progression across generations
    if len(solutions) > top_n_per_gen:
        print(f" __Quality progression:")
        
        # Show best from early vs late generations
        early_best = max(solutions[:top_n_per_gen], key=lambda x: x.fitness)
        late_best = max(solutions[-top_n_per_gen:], key=lambda x: x.fitness)
        
        print(f"      Early generation best: {early_best.fitness:.3f}")
        print(f"      Late generation best:  {late_best.fitness:.3f}")
        print(f"      Improvement:           {late_best.fitness - early_best.fitness:.3f}")
    
    print(f"__Collected {len(solutions)} elite solutions from PSO generations")
    
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_GA(benchmark_problem: BenchmarkProblem,
                 ga_population_size: int,
                 sample_size: int) -> PRef:
    """returns the population obtained by concatenating all the generations the GA will go through"""
    algorithm = GA(search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   crossover_operator=TwoPointFSCrossover(),
                   selection_operator=TournamentSelection(),
                   crossover_rate=0.5,
                   elite_proportion=0.02,
                   tournament_size=3,
                   population_size=ga_population_size,
                   fitness_function=benchmark_problem.fitness_function)

    solutions : list[EvaluatedFS] = []
    solutions.extend(algorithm.current_population)

    while len(solutions) < sample_size:
        algorithm.step()
        solutions.extend(algorithm.current_population)

    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_SA(benchmark_problem: BenchmarkProblem,
                 sample_size: int,
                 max_trace: int) -> PRef:
    algorithm = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   cooling_coefficient=0.9995)

    solutions : list[EvaluatedFS] = []

    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace= max_trace))

    solutions = solutions[:sample_size]

    # best_solution = max(solutions)
    # df = benchmark_problem.details_of_solution(best_solution.full_solution)   # Experimental
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_tabu_search(benchmark_problem: BenchmarkProblem,
                          sample_size: int,
                          max_trace: int) -> PRef:
    algorithm = TabuSearch(fitness_function=benchmark_problem.fitness_function,
                           mutation_operator=SinglePointFSMutation(benchmark_problem.search_space))

    solutions: list[EvaluatedFS] = []

    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))

    solutions = solutions[:sample_size]

    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)




def pRef_from_GA_best(benchmark_problem: BenchmarkProblem,
                      fs_evaluation_budget: int,
                      sample_size: int) -> PRef:
    """
    Returns the population of the last iteration of the algorithm, after having used the given evaluation budget.
    """
    algorithm =  GA(search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   crossover_operator=TwoPointFSCrossover(),
                   selection_operator=TournamentSelection(),
                   crossover_rate=0.5,
                   elite_proportion=0.02,
                   tournament_size=3,
                   population_size=sample_size,
                   fitness_function=benchmark_problem.fitness_function)


    algorithm.run(termination_criteria=TerminationCriteria.FullSolutionEvaluationLimit(fs_evaluation_budget))
    solutions = algorithm.get_results(sample_size)

    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_SA_best(benchmark_problem: BenchmarkProblem,
                 sample_size: int) -> PRef:
    """returns only the end results of each run of SA. There will be _sample\_size_ runs in total.
    Note that this is significantly slower that using all of the attempts"""

    algorithm = SA(fitness_function=benchmark_problem.fitness_function,
                   search_space=benchmark_problem.search_space,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   cooling_coefficient=0.9995)

    solutions = [algorithm.get_one() for _ in range(sample_size)]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_BBO_all(benchmark_problem, sample_size: int, max_trace: int) -> PRef:
    
    from mealpy.bio_based import BBO as MEALPY_BBO
    from mealpy import IntegerVar, Problem
    import numpy as np
    from Core.FullSolution import FullSolution
    from Core.EvaluatedFS import EvaluatedFS
    from Core.PRef import PRef
    
    print(f"Running MEALPY BBO on {benchmark_problem.__class__.__name__}")
    
    # Setup integer bounds
    dimension = len(benchmark_problem.search_space.cardinalities)
    lb = tuple([0] * dimension)
    ub = tuple([card - 1 for card in benchmark_problem.search_space.cardinalities])
    bounds = IntegerVar(lb=lb, ub=ub, name="bbo_problem")
    
    # Fitness function wrapper
    def fitness_func(solution):
        discrete_solution = np.array(solution, dtype=int)
        full_solution = FullSolution(discrete_solution)
        return float(benchmark_problem.fitness_function(full_solution))
    
    problem_dict = {
        "obj_func": fitness_func,
        "bounds": bounds,
        "minmax": "max",
        "save_population": True, 
        "log_to": None
    }

    pop_size = 50
    
    print(f"BBO parameters: pop_size={pop_size}")
    
    model = MEALPY_BBO.OriginalBBO(
        epoch=get_problem_specific_epochs(benchmark_problem, max_trace, pop_size),
        pop_size=pop_size,
        p_m=0.01,
        n_elites=2
    )
    
    model.solve(problem_dict, mode='single')
    
    solutions = []
    
    print(f"   Available history attributes: {dir(model.history)}")
    
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
        print("    No list_population in history, using other history data")

    solutions = solutions[:max_trace]

    print(f"___Generated {len(solutions)} solutions from MEALPY BBO history")
    
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_CRO_all(benchmark_problem, sample_size: int, max_trace: int) -> PRef:
    
    from mealpy.evolutionary_based import CRO as MEALPY_CRO
    from mealpy import IntegerVar, Problem
    import numpy as np
    from Core.FullSolution import FullSolution
    from Core.EvaluatedFS import EvaluatedFS
    from Core.PRef import PRef
    
    print(f"Running MEALPY CRO on {benchmark_problem.__class__.__name__}")
    
    # Setup integer bounds
    dimension = len(benchmark_problem.search_space.cardinalities)
    lb = tuple([0] * dimension)
    ub = tuple([card - 1 for card in benchmark_problem.search_space.cardinalities])
    bounds = IntegerVar(lb=lb, ub=ub, name="cro_problem")
    
    # Fitness function wrapper
    def fitness_func(solution):
        discrete_solution = np.array(solution, dtype=int)
        full_solution = FullSolution(discrete_solution)
        return float(benchmark_problem.fitness_function(full_solution))
    
    problem_dict = {
        "obj_func": fitness_func,
        "bounds": bounds,
        "minmax": "max",
        "save_population": True, 
        "log_to": None
    }

    pop_size = 50
    
    print(f"   CRO parameters: pop_size={pop_size}")
    
    model = MEALPY_CRO.OriginalCRO(
        epoch=get_problem_specific_epochs(benchmark_problem, max_trace, pop_size),
        pop_size=pop_size,
        po=0.4,
        Fb=0.9,
        Fa=0.1,
        Fd=0.1,
        Pd=0.5,
        GCR=0.1,
        gamma_min=0.02,
        gamma_max=0.2,
        n_trials=5
    )
    
    model.solve(problem_dict, mode='single')
    
    solutions = []
    
    print(f"   Available history attributes: {dir(model.history)}")
    
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
        print("    No list_population in history, using other history data")

    solutions = solutions[:max_trace]

    print(f"__Generated {len(solutions)} solutions from MEALPY CRO history")
    
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_ACO_all(benchmark_problem, sample_size: int, max_trace: int) -> PRef:
    
    from mealpy.swarm_based import ACOR as MEALPY_ACOR
    from mealpy import IntegerVar, Problem
    import numpy as np
    from Core.FullSolution import FullSolution
    from Core.EvaluatedFS import EvaluatedFS
    from Core.PRef import PRef
    
    print(f"Running MEALPY ACOR on {benchmark_problem.__class__.__name__}")
    
    # Setup integer bounds
    dimension = len(benchmark_problem.search_space.cardinalities)
    lb = tuple([0] * dimension)
    ub = tuple([card - 1 for card in benchmark_problem.search_space.cardinalities])
    bounds = IntegerVar(lb=lb, ub=ub, name="acor_problem")
    
    # Fitness function wrapper
    def fitness_func(solution):
        discrete_solution = np.array(solution, dtype=int)
        full_solution = FullSolution(discrete_solution)
        return float(benchmark_problem.fitness_function(full_solution))
    
    problem_dict = {
        "obj_func": fitness_func,
        "bounds": bounds,
        "minmax": "max",
        "save_population": True, 
        "log_to": None
    }

    pop_size = 40
    
    print(f"   ACOR parameters: pop_size={pop_size}")
    
    model = MEALPY_ACOR.OriginalACOR(
        epoch=get_problem_specific_epochs(benchmark_problem, max_trace, pop_size),
        pop_size=pop_size,
        sample_count=25,
        intent_factor=0.5,
        zeta=1.0
    )
    
    model.solve(problem_dict, mode='single')
    
    solutions = []
    
    print(f"   Available history attributes: {dir(model.history)}")
    
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
        print("    No list_population in history, using other history data")

    solutions = solutions[:max_trace]

    print(f"__Generated {len(solutions)} solutions from MEALPY ACOR history")
    
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_BRO_all(benchmark_problem, sample_size: int, max_trace: int) -> PRef:
    
    from mealpy.human_based import BRO as MEALPY_BRO
    from mealpy import IntegerVar, Problem
    import numpy as np
    from Core.FullSolution import FullSolution
    from Core.EvaluatedFS import EvaluatedFS
    from Core.PRef import PRef
    
    print(f"Running MEALPY BRO on {benchmark_problem.__class__.__name__}")
    
    # Setup integer bounds
    dimension = len(benchmark_problem.search_space.cardinalities)
    lb = tuple([0] * dimension)
    ub = tuple([card - 1 for card in benchmark_problem.search_space.cardinalities])
    bounds = IntegerVar(lb=lb, ub=ub, name="bro_problem")
    
    # Fitness function wrapper
    def fitness_func(solution):
        discrete_solution = np.array(solution, dtype=int)
        full_solution = FullSolution(discrete_solution)
        return float(benchmark_problem.fitness_function(full_solution))
    
    problem_dict = {
        "obj_func": fitness_func,
        "bounds": bounds,
        "minmax": "max",
        "save_population": True, 
        "log_to": None
    }

    pop_size = 40
    
    print(f"   BRO parameters: pop_size={pop_size}")
    
    model = MEALPY_BRO.OriginalBRO(
        epoch=get_problem_specific_epochs(benchmark_problem, max_trace, pop_size),
        pop_size=pop_size,
        threshold=3,
        max_diff=0.1
    )
    
    model.solve(problem_dict, mode='single')
    
    solutions = []
    
    print(f"   Available history attributes: {dir(model.history)}")
    
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
        print("    No list_population in history, using other history data")

    solutions = solutions[:max_trace]

    print(f"__Generated {len(solutions)} solutions from MEALPY BRO history")
    
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)





















def pRef_from_DE(benchmark_problem: BenchmarkProblem,
                 sample_size: int,
                 max_trace: int) -> PRef:
    algorithm = DE(search_space=benchmark_problem.search_space,
                   fitness_function=benchmark_problem.fitness_function,
                   mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                   pop_size=40,
                   max_iter=max_trace)

    solutions: list[EvaluatedFS] = []
    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))
    solutions = solutions[:sample_size]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_BBO(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:
    algorithm = BBO(search_space=benchmark_problem.search_space,
                    fitness_function=benchmark_problem.fitness_function,
                    mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                    pop_size=40,
                    max_iter=max_trace)

    solutions: list[EvaluatedFS] = []
    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))
    solutions = solutions[:sample_size]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_ABC(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:
    algorithm = ABC(search_space=benchmark_problem.search_space,
                    fitness_function=benchmark_problem.fitness_function,
                    mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                    pop_size=40,
                    max_iter=max_trace)

    solutions: list[EvaluatedFS] = []
    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))
    solutions = solutions[:sample_size]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_CRO(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:
    algorithm = CRO(search_space=benchmark_problem.search_space,
                    fitness_function=benchmark_problem.fitness_function,
                    mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                    pop_size=40,
                    max_iter=max_trace)

    solutions: list[EvaluatedFS] = []
    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))
    solutions = solutions[:sample_size]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_BRO(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:
    algorithm = BRO(search_space=benchmark_problem.search_space,
                    fitness_function=benchmark_problem.fitness_function,
                    mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                    pop_size=40,
                    max_iter=max_trace)

    solutions: list[EvaluatedFS] = []
    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))
    solutions = solutions[:sample_size]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)



def pRef_from_AOA_using_history(benchmark_problem, sample_size, max_trace):
    """Use MEALPY's built-in AOA history to collect sample_size solutions"""
    from mealpy.math_based import AOA
    from mealpy import FloatVar
    import numpy as np
    
    lb, ub, dimension, info = create_mealpy_bounds(benchmark_problem)
    objective_func = create_universal_objective_function(benchmark_problem)
    
    problem_dict = {
        "obj_func": objective_func,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "max",
        "save_population": True,
    }
    
    pop_size = 50
    epochs_needed = 1000
    
    model = AOA.OriginalAOA(epoch=epochs_needed, pop_size=pop_size, alpha=5, mu=0.499)
    model.solve(problem_dict)
    
    solutions = []
    for epoch_population in model.history.list_population:
        for agent in epoch_population:
            if len(solutions) >= sample_size:
                break
            try:
                discrete_solution = np.round(agent.solution).astype(int)
                discrete_solution = np.clip(discrete_solution, lb, ub)
                
                from Core.FullSolution import FullSolution
                fs = FullSolution(discrete_solution)
                fitness = float(agent.target.fitness)
                solutions.append(EvaluatedFS(fs, fitness))
            except:
                continue
        
        if len(solutions) >= sample_size:
            break
    
    print(f"Collected {len(solutions)} solutions from {len(model.history.list_population)} epochs")
    return PRef.from_evaluated_full_solutions(solutions[:sample_size], benchmark_problem.search_space)


def pRef_from_ACO(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:
    algorithm = ACO(search_space=benchmark_problem.search_space,
                    fitness_function=benchmark_problem.fitness_function,
                    mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                    pop_size=40,
                    max_iter=max_trace)

    solutions: list[EvaluatedFS] = []
    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))
    solutions = solutions[:sample_size]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_SMO(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:
    algorithm = SMO(search_space=benchmark_problem.search_space,
                    fitness_function=benchmark_problem.fitness_function,
                    mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                    pop_size=40,
                    max_iter=max_trace)

    solutions: list[EvaluatedFS] = []
    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))
    solutions = solutions[:sample_size]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_WOA(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:
    algorithm = WOA(search_space=benchmark_problem.search_space,
                    fitness_function=benchmark_problem.fitness_function,
                    mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                    pop_size=40,
                    max_iter=max_trace)

    solutions: list[EvaluatedFS] = []
    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))
    solutions = solutions[:sample_size]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_ABC(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:

    from mealpy.swarm_based import ABC
    from mealpy import FloatVar
    import numpy as np
    
    # Get problem-specific bounds and info
    lb, ub, dimension, info = create_mealpy_bounds(benchmark_problem)
    
    # Create objective function
    objective_func = create_universal_objective_function(benchmark_problem, minmax="max")
    
    # Create MEALPY problem dictionary
    problem_dict = {
        "obj_func": objective_func,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "max",
        "log_to": None,
        "save_population": True,
    }

    epoch = 1000
    pop_size = 100
    # Adjust for BT problems
    if info['problem_type'] == 'BTProblem':
        epoch = int(epoch * 1.25)
    
    print(f"🔧 ABC Configuration for {info['problem_type']}:")
    print(f"   Dimension: {dimension}")
    print(f"   Domain sizes: {info['domain_info'][:5]}{'...' if len(info['domain_info']) > 5 else ''}")
    print(f"   Uniform domains: {info['is_uniform']}")
    print(f"   ABC parameters: epoch={epoch}, pop_size={pop_size}")
    
    # Configure and run ABC
    model = ABC.OriginalABC(
        epoch=epoch,
        pop_size=pop_size,
        couple_bees=16,     # Number of bees which provided for good location and number of selected sites
        patch_variables=5,  # Patch variables
        sites=3,           # Selected sites
        e_bees=4           # Elite bees
    )
    
    model.solve(problem_dict)
    
    # Extract solutions with proper discrete conversion
    solutions: list[EvaluatedFS] = []
    
    if hasattr(model, 'population') and model.population:
        for individual in model.population:
            if len(solutions) >= sample_size:
                break
            try:
                # Convert to discrete solution
                discrete_solution = np.round(individual.solution).astype(int)
                discrete_solution = np.clip(discrete_solution, lb, ub)
                
                from Core.FullSolution import FullSolution
                fs = FullSolution(discrete_solution)
                fitness = float(individual.target.fitness)
                solutions.append(EvaluatedFS(fs, fitness))
            except Exception as e:
                print(f"    Error processing individual: {e}")
                continue
    """ 
    # Pad with random solutions if needed
    while len(solutions) < sample_size:
        try:
            # Generate random discrete solution
            random_solution = [np.random.randint(lb[i], ub[i] + 1) for i in range(dimension)]
            
            from Core.FullSolution import FullSolution
            fs = FullSolution(random_solution)
            fitness = benchmark_problem.fitness_function(fs)
            solutions.append(EvaluatedFS(fs, fitness))
        except:
            break
        
        if len(solutions) >= sample_size * 2:  # Safety limit
            break
    """
    # Trim to requested size
    solutions = solutions[:sample_size]
    
    print(f"    Generated {len(solutions)} solutions")
    
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_HHO(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:
    algorithm = HHO(search_space=benchmark_problem.search_space,
                    fitness_function=benchmark_problem.fitness_function,
                    mutation_operator=SinglePointFSMutation(benchmark_problem.search_space),
                    pop_size=40,
                    max_iter=max_trace)

    solutions: list[EvaluatedFS] = []
    while len(solutions) < sample_size:
        solutions.extend(algorithm.get_one_with_attempts(max_trace=max_trace))
    solutions = solutions[:sample_size]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)


def pRef_from_BRO(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:

    from mealpy.human_based import BRO
    from mealpy import FloatVar
    import numpy as np
    
    # Get problem-specific bounds and info
    lb, ub, dimension, info = create_mealpy_bounds(benchmark_problem)
    
    # Create objective function
    objective_func = create_universal_objective_function(benchmark_problem, minmax="max")
    
    # Create MEALPY problem dictionary
    problem_dict = {
        "obj_func": objective_func,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "max",
        "log_to": None,
        "save_population": True,
    }

    epoch = 1000
    pop_size = 100
    # Adjust for BT problems
    if info['problem_type'] == 'BTProblem':
        epoch = int(epoch * 1.15)
    
    print(f"   BRO Configuration for {info['problem_type']}:")
    print(f"   Dimension: {dimension}")
    print(f"   Domain sizes: {info['domain_info'][:5]}{'...' if len(info['domain_info']) > 5 else ''}")
    print(f"   Uniform domains: {info['is_uniform']}")
    print(f"   BRO parameters: epoch={epoch}, pop_size={pop_size}")
    
    # Configure and run BRO
    model = BRO.OriginalBRO(
        epoch=epoch,
        pop_size=pop_size,
        threshold=3     # Damage threshold
    )
    
    model.solve(problem_dict)
    
    # Extract solutions with proper discrete conversion
    solutions: list[EvaluatedFS] = []
    
    if hasattr(model, 'population') and model.population:
        for individual in model.population:
            if len(solutions) >= sample_size:
                break
            try:
                # Convert to discrete solution
                discrete_solution = np.round(individual.solution).astype(int)
                discrete_solution = np.clip(discrete_solution, lb, ub)
                
                from Core.FullSolution import FullSolution
                fs = FullSolution(discrete_solution)
                fitness = float(individual.target.fitness)
                solutions.append(EvaluatedFS(fs, fitness))
            except Exception as e:
                print(f"    Error processing individual: {e}")
                continue
    
    # Pad with random solutions if needed
    while len(solutions) < sample_size:
        try:
            # Generate random discrete solution
            random_solution = [np.random.randint(lb[i], ub[i] + 1) for i in range(dimension)]
            
            from Core.FullSolution import FullSolution
            fs = FullSolution(random_solution)
            fitness = benchmark_problem.fitness_function(fs)
            solutions.append(EvaluatedFS(fs, fitness))
        except:
            break
        
        if len(solutions) >= sample_size * 2:  # Safety limit
            break
    
    # Trim to requested size
    solutions = solutions[:sample_size]
    
    print(f"    Generated {len(solutions)} solutions")
    
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_AOA(benchmark_problem: BenchmarkProblem,
                  sample_size: int,
                  max_trace: int) -> PRef:

    from mealpy.math_based import AOA
    from mealpy import FloatVar
    import numpy as np
    
    # Get problem-specific bounds and info
    lb, ub, dimension, info = create_mealpy_bounds(benchmark_problem)
    
    # Create objective function
    objective_func = create_universal_objective_function(benchmark_problem, minmax="max")
    
    # Create MEALPY problem dictionary
    problem_dict = {
        "obj_func": objective_func,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": "max",
        "log_to": None,
        "save_population": True,
    }

    epoch = 1000
    pop_size = 100
    # Adjust for BT problems
    if info['problem_type'] == 'BTProblem':
        epoch = int(epoch * 1.25)
    
    print(f"   AOA Configuration for {info['problem_type']}:")
    print(f"   Dimension: {dimension}")
    print(f"   Domain sizes: {info['domain_info'][:5]}{'...' if len(info['domain_info']) > 5 else ''}")
    print(f"   Uniform domains: {info['is_uniform']}")
    print(f"   AOA parameters: epoch={epoch}, pop_size={pop_size}")
    
    # Configure and run AOA
    model = AOA.OriginalAOA(
        epoch=epoch,
        pop_size=pop_size,
        c1=2.0,         # Acceleration coefficient 1
        c2=6.0,         # Acceleration coefficient 2
        c3=2.0,         # Acceleration coefficient 3
        c4=0.5,         # Acceleration coefficient 4
        U=0.9,          # Positive random number [0, 1]
        alpha=50        # Density of material
    )
    
    model.solve(problem_dict)
    
    # Extract solutions with proper discrete conversion
    solutions: list[EvaluatedFS] = []
    
    if hasattr(model, 'population') and model.population:
        for individual in model.population:
            if len(solutions) >= sample_size:
                break
            try:
                # Convert to discrete solution
                discrete_solution = np.round(individual.solution).astype(int)
                discrete_solution = np.clip(discrete_solution, lb, ub)
                
                from Core.FullSolution import FullSolution
                fs = FullSolution(discrete_solution)
                fitness = float(individual.target.fitness)
                solutions.append(EvaluatedFS(fs, fitness))
            except Exception as e:
                print(f"    Error processing individual: {e}")
                continue
    
    # Pad with random solutions if needed
    while len(solutions) < sample_size:
        try:
            # Generate random discrete solution
            random_solution = [np.random.randint(lb[i], ub[i] + 1) for i in range(dimension)]
            
            from Core.FullSolution import FullSolution
            fs = FullSolution(random_solution)
            fitness = benchmark_problem.fitness_function(fs)
            solutions.append(EvaluatedFS(fs, fitness))
        except:
            break
        
        if len(solutions) >= sample_size * 2:  # Safety limit
            break
    
    # Trim to requested size
    solutions = solutions[:sample_size]
    
    print(f"    Generated {len(solutions)} solutions")
    
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)

def pRef_from_WOA_best(benchmark_problem: BenchmarkProblem,
                       sample_size: int) -> PRef:
    algorithm = WOA(search_space=benchmark_problem.search_space,
                    fitness_function=benchmark_problem.fitness_function,
                    mutation_operator=SinglePointFSMutation(benchmark_problem.search_space))
    solutions = [algorithm.get_one() for _ in range(sample_size)]
    return PRef.from_evaluated_full_solutions(solutions, benchmark_problem.search_space)




def get_search_space_info(benchmark_problem):
    """
    Universal function to extract search space information from any benchmark problem
    Works with SAT, GraphColouring, BT, and future problems
    """
    search_space = benchmark_problem.search_space
    problem_type = type(benchmark_problem).__name__
    
    # Method 1: Try to access search space domain information directly
    domain_info = None
    dimension = None
    
    # Try different attribute names that might contain domain sizes
    for attr_name in ['domain_sizes', 'variable_cardinalities', 'bounds']:
        if hasattr(search_space, attr_name):
            domain_info = getattr(search_space, attr_name)
            dimension = len(domain_info)
            break
    
    # Method 2: Problem-specific reconstruction
    if domain_info is None:
        if problem_type == 'SATProblem':
            dimension = benchmark_problem.amount_of_variables
            domain_info = [2] * dimension  # Binary variables
            
        elif problem_type == 'GraphColouring':
            dimension = benchmark_problem.amount_of_nodes
            domain_info = [benchmark_problem.amount_of_colours] * dimension
            
        elif problem_type == 'BTProblem':
            # Reconstruct cardinalities from workers (avoiding utils.join_lists)
            cardinalities = []
            for worker in benchmark_problem.workers:
                worker_cardinalities = worker.get_variable_cardinalities(custom_starting_days=False)
                cardinalities.extend(worker_cardinalities)
            
            dimension = len(cardinalities)
            domain_info = cardinalities
    
    # Method 3: Generic fallback
    if dimension is None:
        try:
            dimension = len(search_space)
            # Assume binary if no other info available
            domain_info = [2] * dimension
        except:
            raise AttributeError(f"Cannot determine search space for {problem_type}")
    
    return {
        'dimension': dimension,
        'domain_info': domain_info,
        'is_uniform': len(set(domain_info)) == 1,  # All domains same size?
        'problem_type': problem_type,
        'search_space': search_space
    }

def create_mealpy_bounds(benchmark_problem):
    """Create MEALPY-compatible bounds from any benchmark problem"""
    info = get_search_space_info(benchmark_problem)
    
    dimension = info['dimension']
    domain_info = info['domain_info']
    
    # Create bounds: [0, domain_size-1] for each variable
    lb = [0] * dimension
    ub = [domain_size - 1 for domain_size in domain_info]
    
    return lb, ub, dimension, info

def create_universal_objective_function(benchmark_problem, minmax="max"):
    """Create objective function that works with discrete domains"""
    import numpy as np
    
    def objective_function(solution):
        try:
            from Core.FullSolution import FullSolution
            
            # Convert continuous MEALPY solution to discrete integers
            discrete_solution = abs(np.tanh(solution)).astype(int)
            
            # Get bounds to ensure solution is valid
            lb, ub, _, _ = create_mealpy_bounds(benchmark_problem)
            discrete_solution = np.clip(discrete_solution, lb, ub)
            
            # Create FullSolution and evaluate
            fs = FullSolution(discrete_solution)
            result = benchmark_problem.fitness_function(fs)
            
            # Validate result
            if result is None or not np.isfinite(result) or np.isnan(result):
                return -1000000.0 if minmax == "max" else 1000000.0
                
            return float(result)
            
        except Exception as e:
            # Return penalty for invalid solutions
            penalty = -1000000.0 if minmax == "max" else 1000000.0
            return penalty + np.random.uniform(-1000, 1000)
    
    return objective_function

def get_bounds_for_mealpy(benchmark_problem):
    """
    Extract bounds suitable for MEALPY from any benchmark problem
    """
    info = get_search_space_info(benchmark_problem)
    dimension = info['dimension']
    domain_info = info['domain_info']
    
    # For discrete problems, map to [0, max_domain_size-1] for each variable
    if domain_info is not None:
        # Each variable has its own domain size
        lb = [0] * dimension
        ub = [domain_size - 1 for domain_size in domain_info]
    else:
        # Default to binary [0, 1] for unknown discrete problems
        lb = [0] * dimension
        ub = [1] * dimension
    
    return lb, ub, dimension

def create_mealpy_problem_dict(benchmark_problem, minmax="max"):
    """
    Create MEALPY problem dictionary from any benchmark problem
    """
    from mealpy import FloatVar
    import numpy as np
    
    # Get bounds and dimension
    lb, ub, dimension = get_bounds_for_mealpy(benchmark_problem)
    
    # Objective function wrapper
    def objective_function(solution):
        try:
            from Core.FullSolution import FullSolution
            
            # For discrete problems, round continuous solutions to integers
            discrete_solution = np.round(solution).astype(int)
            
            # Ensure bounds are respected
            discrete_solution = np.clip(discrete_solution, lb, ub)
            
            fs = FullSolution(discrete_solution)
            result = benchmark_problem.fitness_function(fs)
            
            if result is None or not np.isfinite(result) or np.isnan(result):
                return -1000000.0 if minmax == "max" else 1000000.0
                
            return float(result)
        except Exception as e:
            penalty = -1000000.0 if minmax == "max" else 1000000.0
            return penalty + np.random.uniform(-1000, 1000)
    
    # Create MEALPY problem dictionary
    problem_dict = {
        "obj_func": objective_function,
        "bounds": FloatVar(lb=lb, ub=ub),
        "minmax": minmax,
        "log_to": None,
        "save_population": True,
    }
    
    return problem_dict, dimension

