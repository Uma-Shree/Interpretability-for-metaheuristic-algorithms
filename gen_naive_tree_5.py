#!/usr/bin/env python3
"""
Generate only naive trees 
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
    
    # Updated to include all 8 optimization algorithms
    pRef_methods = [
         "CRO", "BBO", "ACO", "BRO", "PSO"
           ] 
    '''      # Particle Swarm Optimization
        "CRO",        # Coral Reef Optimization
        "BRO",        # Battle Royale Optimization
        "AOA",        # Arithmetic Optimization Algorithm
        "ABC",        # Artificial Bee Colony
        "BBO",        # Biogeography-Based Optimization
        "ACO",        # Ant Colony Optimization
        "SSO"         # Social Spider Optimization
    '''
    
    sample_size = 10000
    depths = [3, 4, 5]
    
    # Only naive trees
    tree_dicts = [{"kind": "naive", "depths": depths}]
    
    mode = "local"
    repeats = 1
    debug = False
    print_progress = True
    
    print("=== GENERATING NAIVE TREES FOR ALL ALGORITHMS ===")
    print(f"Problems: {list(problems.keys())}")
    print(f"Search methods: {pRef_methods}")
    print(f"Tree types: naive only")
    print(f"Depths: {depths}")
    print(f"Expected combinations: {len(problems)} × {len(pRef_methods)} × {len(depths)} × {repeats}")
    print(f"Total expected runs: {len(problems) * len(pRef_methods) * len(depths) * repeats}")
    
    def make_file_with_json_contents(json_dict, seed):
        # Save to your existing data folder
        destination_folder = get_compare_own_data_folder()
        
        # CREATE THE DIRECTORY IF IT DOESN'T EXIST
        utils.make_directory(destination_folder)
        
        json_file_name = os.path.join(destination_folder, 
                                     f"naive_all_algorithms_{seed}_" + utils.get_formatted_timestamp() + ".json")
        print(f"Storing naive results at {json_file_name}")
        with open(json_file_name, "w") as file:
            json.dump(json_dict, file, indent=4)

    def single_run(seed: int):
        results = []
        warnings.warn(f"The seed is {seed}")
        
        # Counter for progress tracking
        total_combinations = len(problems) * len(pRef_methods)
        current_combination = 0

        for problem_name, problem in problems.items():
            for pRef_method in pRef_methods:
                current_combination += 1
                if print_progress:
                    progress_pct = (current_combination / total_combinations) * 100
                    warnings.warn(f"[{progress_pct:.1f}%] Processing: {problem_name} + {pRef_method}")
                
                try:
                    datapoint = get_datapoint_for_instance(
                        problem_name=problem_name,
                        problem=problem,
                        tree_settings_list=tree_dicts,
                        sample_size=sample_size,
                        pRef_method=pRef_method,
                        crash_on_error=debug,
                        seed=seed
                    )
                    results.append(datapoint)
                    
                    # Success message
                    if print_progress:
                        print(f"Success: {problem_name} + {pRef_method}")
                        
                except Exception as e:
                    error_msg = f"Error with {problem_name} + {pRef_method}: {str(e)[:100]}..."
                    print(error_msg)
                    warnings.warn(error_msg)
                    
                    # Create error record for tracking
                    error_record = {
                        "problem_name": problem_name,
                        "pRef_method": pRef_method,
                        "error": str(e),
                        "tree_settings_list": tree_dicts,
                        "sample_size": sample_size
                    }
                    results.append(error_record)
                    continue

        print(f"\nCompleted seed {seed}: {len(results)} results generated")
        make_file_with_json_contents(results, seed)

    # Generate naive trees with different seeds for reproducibility
    seeds = [2001, 2002, 2003, 2004, 2005]  # Use different seeds to avoid conflicts with existing data
    
    print(f"\nStarting naive tree generation with seeds: {seeds}")
    
    for seed_idx, seed in enumerate(seeds):
        print(f"\n--- SEED {seed} ({seed_idx + 1}/{len(seeds)}) ---")
        single_run(seed)
    
    print("\nNAIVE TREE GENERATION COMPLETED!")
    print(f"Generated files for {len(seeds)} seeds with all algorithm combinations.")
    print("Check your data folder for the generated naive_all_algorithms_*.json files.")


def generate_specific_algorithm_naive_trees(algorithm_name: str):
    """Generate naive trees for a specific algorithm only (for testing/debugging)"""
    
    if algorithm_name not in ["PSO", "CRO", "BRO", "AOA", "ABC", "BBO", "ACO", "SSO"]:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    problems = get_problems_with_names()
    pRef_methods = [algorithm_name]  # Single algorithm
    sample_size = 10000
    depths = [2, 3, 4, 5]
    tree_dicts = [{"kind": "naive", "depths": depths}]
    
    print(f"=== GENERATING NAIVE TREES FOR {algorithm_name} ONLY ===")
    
    def single_run_specific(seed: int):

        results = []
        for problem_name, problem in problems.items():
            print(f"Processing: {problem_name} with {algorithm_name}")
            try:
                datapoint = get_datapoint_for_instance(
                    problem_name=problem_name,
                    problem=problem,
                    tree_settings_list=tree_dicts,
                    sample_size=sample_size,
                    pRef_method=algorithm_name,
                    crash_on_error=True,  # Crash on error for debugging
                    seed=seed
                )
                results.append(datapoint)
                print(f"Success: {problem_name}")
            except Exception as e:
                print(f"Error with {problem_name}: {e}")
                raise 
        
        # Save results
        destination_folder = get_compare_own_data_folder()
        utils.make_directory(destination_folder)
        json_file_name = os.path.join(destination_folder, 
                                     f"naive_{algorithm_name}_only_{seed}_" + utils.get_formatted_timestamp() + ".json")
        with open(json_file_name, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Saved: {json_file_name}")
    
    single_run_specific(3001)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Allow testing specific algorithm: python script.py PSO
        algorithm = sys.argv[1].upper()
        generate_specific_algorithm_naive_trees(algorithm)
    else:
        # Default: generate for all algorithms
        generate_naive_trees_only()
