#!/usr/bin/env python3
"""
Generate IAI (Interpretable AI) data for the new 8 metaheuristic algorithms.
Runs: 8 algorithms × 6 problems × 3 depths × 5 repeats = 720 experiments with IAI trees
"""

import json
import os
import warnings
import sys
import random
import time

import utils
from config import get_iai_run_folder
from VarianceDecisionTree.compare_prediction_powers import get_datapoint_for_instance, get_problems_with_names


def gather_data_compare_iai():
    """
    Generate IAI data for all 6 problems using the new 8 metaheuristic algorithms.
    Uses tree depths 3, 4, 5 with cp=0.25 for interpretable decision trees.
    """

    problems = get_problems_with_names()
    
    # All 8 new metaheuristic algorithms
    pRef_methods = ["PSO", "CRO", "BRO", "AOA", "ABC", "BBO", "ACO", "SSO"]
    
    sample_sizes = [1000]
    depths = [3, 4, 5]
    
    # IAI tree configuration with complexity parameter
    tree_dicts = [
        {"kind": "iai", "cp": 0.25, "depths": depths}
    ]

    mode = "local"
    debug = False  # Set to True for debugging single problem
    print_progress = True  # Set to False to reduce verbose output

    print("=== GENERATING IAI DATA FOR NEW 8 ALGORITHMS ===")
    print(f"Problems: {list(problems.keys())}")
    print(f"Algorithms: {pRef_methods}")
    print(f"Tree depths: {depths}")
    print(f"Sample sizes: {sample_sizes}")
    print(f"IAI complexity parameter: {tree_dicts[0]['cp']}")
    
    total_experiments = len(problems) * len(pRef_methods) * len(sample_sizes) * 5  # 5 seeds
    print(f"Total experiments: {total_experiments}")

    if debug:
        print(" DEBUG MODE ENABLED")
        problems = dict(list(problems.items())[:1])  # Only first problem
        pRef_methods = pRef_methods[:2]  # Only first 2 algorithms
        tree_dicts = tree_dicts[:1]
        tree_dicts[0]["depths"] = [3]  # Only depth 3
        sample_sizes = [1000]  # Smaller sample for faster testing

    def make_file_with_json_contents(json_dict, seed):
        """Save results to timestamped JSON file"""
        filename = f"output_{seed}_iai_new_algos_" + utils.get_formatted_timestamp() + ".json"
        json_file_path = os.path.join(destination_folder, filename)
        print(f" Storing results at: {json_file_path}")
        with open(json_file_path, "w") as file:
            json.dump(json_dict, file, indent=4)

    def single_run(seed: int):
        """Run all experiments for a single seed"""
        results = []
        warnings.warn(f" Starting IAI generation for seed: {seed}")
        
        experiment_count = 0
        total_combinations = len(problems) * len(pRef_methods) * len(sample_sizes)

        for problem_name, problem in problems.items():
            for pRef_method in pRef_methods:
                for sample_size in sample_sizes:
                    experiment_count += 1
                    progress = (experiment_count / total_combinations) * 100
                    
                    if print_progress:
                        warnings.warn(f"[{progress:.1f}%] Processing: {problem_name} + {pRef_method} (size={sample_size})")
                    
                    try:
                        """
                        start_time = time.time()

                        datapoint = get_datapoint_for_instance(
                            problem_name=problem_name,
                            problem=problem,
                            tree_settings_list=tree_dicts,
                            sample_size=sample_size,
                            pRef_method=pRef_method,
                            crash_on_error=debug,
                            seed=seed
                        )
                        runtime = round(time.time() - start_time, 3)

                        if isinstance(datapoint, dict):
                            datapoint["runtime_seconds"] = runtime
                            datapoint["problem"] = problem_name
                            datapoint["algorithm"] = pRef_method

                        results.append(datapoint)

                        """

                        start_time = time.time()

                        datapoint = get_datapoint_for_instance(
                            problem_name=problem_name,
                            problem=problem,
                            tree_settings_list=tree_dicts,
                            sample_size=sample_size,
                            pRef_method=pRef_method,
                            crash_on_error=debug,
                            seed=seed
                        )

                        runtime = round(time.time() - start_time, 3)

                        # ✅ Attach runtime at both the top level and inside each depth result
                        if isinstance(datapoint, dict):
                            datapoint["runtime_seconds_total"] = runtime
                            datapoint["problem"] = problem_name
                            datapoint["algorithm"] = pRef_method

                            if "results_by_tree" in datapoint and isinstance(datapoint["results_by_tree"], list):
                                for tree_result in datapoint["results_by_tree"]:
                                    try:
                                        if isinstance(tree_result, dict):
                                            tree_result["runtime_seconds"] = runtime / len(datapoint["results_by_tree"])
                                    except Exception:
                                        pass

                                    
                        else:
                            datapoint = {
                                        "data": datapoint,
                                        "runtime_seconds_total": runtime,
                                        "problem": problem_name,
                                        "algorithm": pRef_method
                            }

                            
                                    
                        results.append(datapoint)
                        print(f"   ✅ Success: {problem_name} + {pRef_method} | Total Runtime: {runtime}s")

                        
                        if print_progress:
                            print(f"   Success: {problem_name} + {pRef_method}")
                            
                    except Exception as e:
                        error_msg = f" Error with {problem_name} + {pRef_method}: {str(e)[:80]}..."
                        print(f"  {error_msg}")
                        
                        if debug:
                            raise e  # Re-raise in debug mode
                        
                        # Record error for analysis
                        error_record = {
                            "problem_name": problem_name,
                            "pRef_method": pRef_method,
                            "sample_size": sample_size,
                            "error": str(e),
                            "tree_settings_list": tree_dicts,
                            "seed": seed
                        }
                        results.append(error_record)
                        continue

        print(f" Completed seed {seed}: {len(results)} results generated")
        
        if mode == "local":
            make_file_with_json_contents(results, seed)
        else:
            print(json.dumps(results, indent=4))

    if mode == "local":
        # Local mode: generate multiple files with different seeds
        destination_folder = get_iai_run_folder()
        utils.make_directory(destination_folder)
        print(f" Results will be stored in: {destination_folder}")

        # Generate 5 repetitions with different seeds
        seeds = list(range(2001, 2002))  # Seeds: 2001, 2002, 2003, 2004, 2005
        
        print(f" Running {len(seeds)} repetitions with seeds: {seeds}")
        
        for i, seed in enumerate(seeds):
            print(f"\n--- REPETITION {i+1}/{len(seeds)} (Seed: {seed}) ---")
            single_run(seed)
            
        print("\n IAI DATA GENERATION COMPLETED!")
        print(f"Generated {len(seeds)} files with IAI tree data")

    else:
        # Console mode: single run with command line or random seed
        if len(sys.argv) < 2:
            seed = random.randrange(10000)
            print(f"Using random seed: {seed}")
        else:
            try:
                seed = int(sys.argv[1])
                print(f"Using provided seed: {seed}")
            except ValueError:
                raise Exception(f"Invalid seed argument: {sys.argv[1]}. Must be an integer.")
        
        single_run(seed)


def gather_data_compare_iai_single_algorithm(algorithm_name: str):
    """
    Generate IAI data for a specific algorithm only (for testing/debugging)
    Usage: python script.py PSO
    """
    if algorithm_name not in ["PSO", "CRO", "BRO", "AOA", "ABC", "BBO", "ACO", "SSO"]:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    problems = get_problems_with_names()
    pRef_methods = [algorithm_name]  # Single algorithm
    sample_sizes = [10000]
    depths = [3, 4, 5]
    tree_dicts = [{"kind": "iai", "cp": 0.25, "depths": depths}]
    
    print(f"=== GENERATING IAI DATA FOR {algorithm_name} ONLY ===")
    
    def single_run_specific(seed: int):
        results = []
        for problem_name, problem in problems.items():
            for sample_size in sample_sizes:
                print(f"Processing: {problem_name} with {algorithm_name} (size={sample_size})")
                try:
                    '''
                    datapoint = get_datapoint_for_instance(
                        problem_name=problem_name,
                        problem=problem,
                        tree_settings_list=tree_dicts,
                        sample_size=sample_size,
                        pRef_method=algorithm_name,
                        crash_on_error=True,  # Crash on error for debugging
                        seed=seed
                    )
                    '''

                    start_time = time.time()

                    datapoint = get_datapoint_for_instance(
                        problem_name=problem_name,
                        problem=problem,
                        tree_settings_list=tree_dicts,
                        sample_size=sample_size,
                        pRef_method=algorithm_name,
                        crash_on_error=True,
                        seed=seed
                    )

                    total_runtime = round(time.time() - start_time, 3)

                    # Add total and per-depth runtime
                    if isinstance(datapoint, dict):
                        datapoint["runtime_seconds_total"] = total_runtime
                        datapoint["problem"] = problem_name
                        datapoint["algorithm"] = algorithm_name
                        if "results_by_tree" in datapoint:
                            num_trees = len(datapoint["results_by_tree"])
                            per_depth_runtime = round(total_runtime / num_trees, 3) if num_trees > 0 else total_runtime
                            for tree_result in datapoint["results_by_tree"]:
                                if isinstance(tree_result, dict):
                                    tree_result["runtime_seconds"] = per_depth_runtime
                    else:
                        datapoint = {
                            "data": datapoint,
                            "runtime_seconds_total": total_runtime,
                            "problem": problem_name,
                            "algorithm": algorithm_name
                        }
                    results.append(datapoint)
                   
                    print(f"✅ Success: {problem_name}")
                except Exception as e:
                    print(f"❌ Error with {problem_name}: {e}")
                    raise  # Re-raise for debugging
        
        # Save results
        destination_folder = get_iai_run_folder()
        utils.make_directory(destination_folder)
        filename = f"output_iai_{algorithm_name}_only_{seed}_" + utils.get_formatted_timestamp() + ".json"
        json_file_path = os.path.join(destination_folder, filename)
        
        with open(json_file_path, "w") as file:
            json.dump(results, file, indent=4)
        print(f"Saved: {json_file_path}")
    
    single_run_specific(4001)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ["PSO", "CRO", "BRO", "AOA", "ABC", "BBO", "ACO", "SSO"]:
        # Test specific algorithm: python script.py PSO
        algorithm = sys.argv[1].upper()
        gather_data_compare_iai_single_algorithm(algorithm)
    else:
        # Default: generate for all algorithms
        gather_data_compare_iai()
