import os
import json
import random
from config import get_compare_own_data_folder
from VarianceDecisionTree.compare_prediction_powers import get_problems_with_names, get_datapoint_for_instance

# Set your output folder
output_folder = os.path.join(os.getcwd(), "output_psw_pswa")
os.makedirs(output_folder, exist_ok=True)

# Output file name (single file for all data)
output_json_path = os.path.join(output_folder, "output_psw_pswa.json")

# Define metrics for PS-W and PS-WA
metrics_dict = {
    "PS-W": "variance",
    "PS-WA": "variance estimated_atomicity"
}

problems = ["BT", "GC_S", "GC_L", "SAT_S", "SAT_M", "SAT_L"]
methods = ["GA", "Tabu", "SA", "uniform"]  # Update if needed
depths = [3, 4, 5]
pRef_size = 10000

all_datapoints = []

for problem in problems:
    for method in methods:
        for depth in depths:
            for kind, metrics in metrics_dict.items():
                print(f"Generating: {problem}, {method}, {depth}, {kind}")
                try:
                    datapoint = get_datapoint_for_instance(
                        problem_name=problem,
                        problem=get_problems_with_names()[problem],
                        tree_settings_list=[{
                            "metrics": metrics,
                            "depth": [depth],
                            "kind": "ps",
                            "ps_budget": 50,  # Example value, adjust as needed
                            "ps_population": 100  # Example value, adjust as needed
                        }],
                        sample_size=pRef_size,
                        pRef_method=method,
                        crash_on_error=False,
                        seed=random.randint(0, 2**32 - 1)
                    )
                    all_datapoints.append(datapoint)
                except Exception as e:
                    print(f"Failed for {problem}, {method}, {depth}, {kind}: {e}")

# Save all datapoints to one JSON file
with open(output_json_path, "w") as f:
    json.dump(all_datapoints, f, indent=2)
print(f"All data saved to {output_json_path}")