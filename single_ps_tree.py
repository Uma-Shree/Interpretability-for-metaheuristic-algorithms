import numpy as np

# === Assumed imports, adjust as per your project ===
from mealpy.bio_based import BBO as MEALPY_BBO
from mealpy import IntegerVar
from Core.FullSolution import FullSolution
from Core.EvaluatedFS import EvaluatedFS

# Replace this with your actual import or object
# from your_benchmark_module import SAT_L_Problem

def run_bbo_satL_depth5_and_log(benchmark_problem, pop_size=30, epoch=50, max_trace=1000, log_filename="satl_bbo_depth5_log.txt"):
    """
    Run Biogeography-Based Optimization for SAT_L at depth 5, logging every agent at every generation.
    """
    # Set up integer bounds for the problem
    dimension = len(benchmark_problem.search_space.cardinalities)
    lb = tuple([0] * dimension)
    ub = tuple([card - 1 for card in benchmark_problem.search_space.cardinalities])
    bounds = IntegerVar(lb=lb, ub=ub, name="bbo_problem")

    # Define the MEALPY-compatible fitness function
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

    # Initialize model
    model = MEALPY_BBO.OriginalBBO(
        epoch=epoch,
        pop_size=pop_size,
        p_m=0.01,
        n_elites=2
    )

    # Solve problem
    model.solve(problem_dict)

    # Log steps
    with open(log_filename, "w") as log_file:
        log_file.write("Step-by-step BBO search process for SAT_L (depth 5):\n\n")
        step_count = 0
        for gen_idx, population in enumerate(model.history.list_population):
            for ind_idx, agent in enumerate(population):
                sol = np.array(agent.solution, dtype=int)
                fitness = agent.target.fitness if hasattr(agent.target, 'fitness') else agent.fitness
                log_file.write(f"Gen {gen_idx}, Agent {ind_idx}: Solution {sol.tolist()}, Fitness {fitness}\n")
                step_count += 1
                if step_count >= max_trace:
                    break
            if step_count >= max_trace:
                break
    print(f"Step-by-step log written to {log_filename}")

# ========= EXAMPLE USAGE =========
if __name__ == "__main__":
    # Replace with your actual SAT_L benchmark object set to depth 5:
    # satL_benchmark = SAT_L_Problem(depth=5)
    satL_benchmark = ... # Replace this line with correct initialization!

    run_bbo_satL_depth5_and_log(
        benchmark_problem=satL_benchmark,
        pop_size=30,
        epoch=50,
        max_trace=1000,
        log_filename="satl_bbo_depth5_log.txt"
    )
