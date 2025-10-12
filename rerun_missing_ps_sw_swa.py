import os
import utils
import random
from config import get_compare_own_data_folder
from VarianceDecisionTree.compare_prediction_powers import get_problems_with_names, get_datapoint_for_instance

missing = [
    # Fill this list with tuples from your notebook output
    # Example: ("SAT_L", "uniform", 3, "simplicity variance"),
    #          ("SAT_L", "uniform", 3, "simplicity variance estimated_atomicity"),
    # ...
('SAT_L', 'uniform', 3, 'simplicity variance'),
('SAT_L', 'uniform', 3, 'simplicity variance estimated_atomicity'),
('SAT_L', 'uniform', 4, 'simplicity variance'),
('SAT_L', 'uniform', 4, 'simplicity variance estimated_atomicity'),
('SAT_L', 'uniform', 5, 'simplicity variance'),
('SAT_L', 'uniform', 5, 'simplicity variance estimated_atomicity'),
('SAT_L', 'SA', 3, 'simplicity variance'),
('SAT_L', 'SA', 3, 'simplicity variance estimated_atomicity'),
('SAT_L', 'SA', 4, 'simplicity variance'),
('SAT_L', 'SA', 4, 'simplicity variance estimated_atomicity'),
('SAT_L', 'SA', 5, 'simplicity variance'),
('SAT_L', 'SA', 5, 'simplicity variance estimated_atomicity'),
('SAT_L', 'Tabu', 3, 'simplicity variance'),
('SAT_L', 'Tabu', 3, 'simplicity variance estimated_atomicity'),
('SAT_L', 'Tabu', 4, 'simplicity variance'),
('SAT_L', 'Tabu', 4, 'simplicity variance estimated_atomicity'),
('SAT_L', 'Tabu', 5, 'simplicity variance'),
('SAT_L', 'Tabu', 5, 'simplicity variance estimated_atomicity'),
('GC_S', 'GA', 3, 'simplicity variance'),
('GC_S', 'GA', 3, 'simplicity variance estimated_atomicity'),
('GC_S', 'GA', 4, 'simplicity variance'),
('GC_S', 'GA', 4, 'simplicity variance estimated_atomicity'),
('GC_S', 'GA', 5, 'simplicity variance'),
('GC_S', 'GA', 5, 'simplicity variance estimated_atomicity'),
('GC_S', 'SA', 3, 'simplicity variance'),
('GC_S', 'SA', 3, 'simplicity variance estimated_atomicity'),
('GC_S', 'SA', 4, 'simplicity variance'),
('GC_S', 'SA', 4, 'simplicity variance estimated_atomicity'),
('GC_S', 'SA', 5, 'simplicity variance'),
('GC_S', 'SA', 5, 'simplicity variance estimated_atomicity'),
('GC_S', 'Tabu', 3, 'simplicity variance'),
('GC_S', 'Tabu', 3, 'simplicity variance estimated_atomicity'),
('GC_S', 'Tabu', 4, 'simplicity variance'),
('GC_S', 'Tabu', 4, 'simplicity variance estimated_atomicity'),
('GC_S', 'Tabu', 5, 'simplicity variance'),
('GC_S', 'Tabu', 5, 'simplicity variance estimated_atomicity'),
('BT', 'uniform', 3, 'simplicity variance'),
('BT', 'uniform', 3, 'simplicity variance estimated_atomicity'),
('BT', 'uniform', 4, 'simplicity variance'),
('BT', 'uniform', 4, 'simplicity variance estimated_atomicity'),
('BT', 'uniform', 5, 'simplicity variance'),
('BT', 'uniform', 5, 'simplicity variance estimated_atomicity'),
('BT', 'SA', 3, 'simplicity variance'),
('BT', 'SA', 3, 'simplicity variance estimated_atomicity'),
('BT', 'SA', 4, 'simplicity variance'),
('BT', 'SA', 4, 'simplicity variance estimated_atomicity'),
('BT', 'SA', 5, 'simplicity variance'),
('BT', 'SA', 5, 'simplicity variance estimated_atomicity'),
('BT', 'Tabu', 3, 'simplicity variance'),
('BT', 'Tabu', 3, 'simplicity variance estimated_atomicity'),
('BT', 'Tabu', 4, 'simplicity variance'),
('BT', 'Tabu', 4, 'simplicity variance estimated_atomicity'),
('BT', 'Tabu', 5, 'simplicity variance'),
('BT', 'Tabu', 5, 'simplicity variance estimated_atomicity'),
('SAT_S', 'GA', 3, 'simplicity variance'),
('SAT_S', 'GA', 3, 'simplicity variance estimated_atomicity'),
('SAT_S', 'GA', 4, 'simplicity variance'),
('SAT_S', 'GA', 4, 'simplicity variance estimated_atomicity'),
('SAT_S', 'GA', 5, 'simplicity variance'),
('SAT_S', 'GA', 5, 'simplicity variance estimated_atomicity'),
('SAT_S', 'uniform', 3, 'simplicity variance'),
('SAT_S', 'uniform', 3, 'simplicity variance estimated_atomicity'),
('SAT_S', 'uniform', 4, 'simplicity variance'),
('SAT_S', 'uniform', 4, 'simplicity variance estimated_atomicity'),
('SAT_S', 'uniform', 5, 'simplicity variance'),
('SAT_S', 'uniform', 5, 'simplicity variance estimated_atomicity'),
('SAT_S', 'SA', 3, 'simplicity variance'),
('SAT_S', 'SA', 3, 'simplicity variance estimated_atomicity'),
('SAT_S', 'SA', 4, 'simplicity variance'),
('SAT_S', 'SA', 4, 'simplicity variance estimated_atomicity'),
('SAT_S', 'SA', 5, 'simplicity variance'),
('SAT_S', 'SA', 5, 'simplicity variance estimated_atomicity'),
('SAT_S', 'Tabu', 3, 'simplicity variance'),
('SAT_S', 'Tabu', 3, 'simplicity variance estimated_atomicity'),
('SAT_S', 'Tabu', 4, 'simplicity variance'),
('SAT_S', 'Tabu', 4, 'simplicity variance estimated_atomicity'),
('SAT_S', 'Tabu', 5, 'simplicity variance'),
('SAT_S', 'Tabu', 5, 'simplicity variance estimated_atomicity'),
('SAT_M', 'GA', 3, 'simplicity variance'),
('SAT_M', 'GA', 3, 'simplicity variance estimated_atomicity'),
('SAT_M', 'GA', 4, 'simplicity variance'),
('SAT_M', 'GA', 4, 'simplicity variance estimated_atomicity'),
('SAT_M', 'GA', 5, 'simplicity variance'),
('SAT_M', 'GA', 5, 'simplicity variance estimated_atomicity'),
('SAT_M', 'uniform', 3, 'simplicity variance'),
('SAT_M', 'uniform', 3, 'simplicity variance estimated_atomicity'),
('SAT_M', 'uniform', 4, 'simplicity variance'),
('SAT_M', 'uniform', 4, 'simplicity variance estimated_atomicity'),
('SAT_M', 'uniform', 5, 'simplicity variance'),
('SAT_M', 'uniform', 5, 'simplicity variance estimated_atomicity'),
('SAT_M', 'SA', 3, 'simplicity variance'),
('SAT_M', 'SA', 3, 'simplicity variance estimated_atomicity'),
('SAT_M', 'SA', 4, 'simplicity variance'),
('SAT_M', 'SA', 4, 'simplicity variance estimated_atomicity'),
('SAT_M', 'SA', 5, 'simplicity variance'),
('SAT_M', 'SA', 5, 'simplicity variance estimated_atomicity'),
('SAT_M', 'Tabu', 3, 'simplicity variance'),
('SAT_M', 'Tabu', 3, 'simplicity variance estimated_atomicity'),
('SAT_M', 'Tabu', 4, 'simplicity variance'),
('SAT_M', 'Tabu', 4, 'simplicity variance estimated_atomicity'),
('SAT_M', 'Tabu', 5, 'simplicity variance'),
('SAT_M', 'Tabu', 5, 'simplicity variance estimated_atomicity'),
('GC_L', 'GA', 3, 'simplicity variance'),
('GC_L', 'GA', 3, 'simplicity variance estimated_atomicity'),
('GC_L', 'GA', 4, 'simplicity variance'),
('GC_L', 'GA', 4, 'simplicity variance estimated_atomicity'),
('GC_L', 'GA', 5, 'simplicity variance'),
('GC_L', 'GA', 5, 'simplicity variance estimated_atomicity'),
('GC_L', 'uniform', 3, 'simplicity variance'),
('GC_L', 'uniform', 3, 'simplicity variance estimated_atomicity'),
('GC_L', 'uniform', 4, 'simplicity variance'),
('GC_L', 'uniform', 4, 'simplicity variance estimated_atomicity'),
('GC_L', 'uniform', 5, 'simplicity variance'),
('GC_L', 'uniform', 5, 'simplicity variance estimated_atomicity'),
('GC_L', 'SA', 3, 'simplicity variance'),
('GC_L', 'SA', 3, 'simplicity variance estimated_atomicity'),
('GC_L', 'SA', 4, 'simplicity variance'),
('GC_L', 'SA', 4, 'simplicity variance estimated_atomicity'),
('GC_L', 'SA', 5, 'simplicity variance'),
('GC_L', 'SA', 5, 'simplicity variance estimated_atomicity'),
('GC_L', 'Tabu', 3, 'simplicity variance'),
('GC_L', 'Tabu', 3, 'simplicity variance estimated_atomicity'),
('GC_L', 'Tabu', 4, 'simplicity variance'),
('GC_L', 'Tabu', 4, 'simplicity variance estimated_atomicity'),
('GC_L', 'Tabu', 5, 'simplicity variance'),
('GC_L', 'Tabu', 5, 'simplicity variance estimated_atomicity')
]

sample_size = 10000
ps_budget = 5000
ps_population = 100
avoid_ancestors = False

destination_folder = get_compare_own_data_folder()
utils.make_directory(destination_folder)

for problem, method, depth, metrics in missing:
    print(f"Generating: {problem}, {method}, {depth}, {metrics}")
    problems = get_problems_with_names()
    tree_settings = [{
        "kind": "ps",
        "ps_budget": ps_budget,
        "ps_population": ps_population,
        "depths": [depth],
        "avoid_ancestors": avoid_ancestors,
        "metrics": metrics
    }]

    import random


    datapoint = get_datapoint_for_instance(
        problem_name=problem,
        problem=problems[problem],
        tree_settings_list=tree_settings,
        sample_size=sample_size,
        pRef_method=method,
        crash_on_error=False,
        seed=random.randint(0, 2**32 - 1)  # Use a random seed
    )
    json_file_name = os.path.join(destination_folder, f"output_{problem}_{method}_{depth}_{metrics}.json")
    with open(json_file_name, "w") as file:
        import json
        json.dump([datapoint], file, indent=4)