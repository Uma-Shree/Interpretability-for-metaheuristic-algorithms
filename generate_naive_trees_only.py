#!/usr/bin/env python3
"""
Generate only naive trees to add to your existing dataset.
This will create naive tree data that you can combine with your PS and IAI data.
"""

import json
import os
import random
import sys
import warnings

import utils
from config import get_compare_own_data_folder
from VarianceDecisionTree.compare_prediction_powers import get_datapoint_for_instance, get_problems_with_names

def generate_naive_trees_only():
    """Generate only naive trees for all problems and methods"""
    
    problems = get_problems_with_names()
    pRef_methods = ["GA", "uniform", "SA", "Tabu"]
    sample_size = 10000
    depths = [2, 3, 4, 5]
    
    # Only naive trees
    tree_dicts = [{"kind": "naive", "depths": depths}]
    
    mode = "local"
    repeats = 2
    debug = False
    print_progress = True
    
    print("=== GENERATING NAIVE TREES ONLY ===")
    print(f"Problems: {list(problems.keys())}")
    print(f"Search methods: {pRef_methods}")
    print(f"Tree types: naive only")
    print(f"Expected combinations: {len(problems)} × {len(pRef_methods)} × {len(depths)} × {repeats}")
    
    def make_file_with_json_contents(json_dict, seed):
        # Save to your existing data folder
        c
        json_file_name = os.path.join(destination_folder, f"naive_output_{seed}_" + utils.get_formatted_timestamp() + ".json")
        print(f"Storing naive results at {json_file_name}")
        with open(json_file_name, "w") as file:
            json.dump(json_dict, file, indent=4)

    def single_run(seed: int):
        results = []
        warnings.warn(f"The seed is {seed}")

        for problem_name, problem in problems.items():
            for pRef_method in pRef_methods:
                if print_progress:
                    warnings.warn(f"{problem_name = }, {pRef_method = }")
                try:
                    datapoint = get_datapoint_for_instance(problem_name=problem_name,
                                                           problem=problem,
                                                           tree_settings_list=tree_dicts,
                                                           sample_size=sample_size,
                                                           pRef_method=pRef_method,
                                                           crash_on_error=debug,
                                                           seed=seed)
                    results.append(datapoint)
                except Exception as e:
                    print(f"Error with {problem_name}, {pRef_method}: {e}")
                    continue

        make_file_with_json_contents(results, seed)

    # Generate naive trees with different seeds
    seeds = [1001, 1002]  # Use different seeds to avoid conflicts
    for seed in seeds:
        single_run(seed)


if __name__ == "__main__":
    generate_naive_trees_only()
