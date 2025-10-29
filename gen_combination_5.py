#!/usr/bin/env python3
"""
Complete PS Tree Generator - Windows Compatible with Threading Timeouts
"""

import os
import json
import random
import time
import threading
import queue
import utils
import numpy as np
from config import get_compare_own_data_folder
from VarianceDecisionTree.compare_prediction_powers import get_problems_with_names, get_datapoint_for_instance

# Patch builtins.len to be safe with numpy scalars (like in main.py)
original_len = len

def safe_len(obj):
    try:
        if isinstance(obj, (int, float, np.number)):
            return 0
        return original_len(obj)
    except TypeError:
        return 0
import builtins
builtins.len = safe_len

class TimeoutError(Exception):
    pass

def run_with_timeout(func, args, kwargs, timeout_seconds):

    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def target():
        try:
            result = func(*args, **kwargs)
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True  # Dies when main thread dies
    thread.start()
    thread.join(timeout_seconds)
    
    if thread.is_alive():
        # Thread is still running, timeout occurred
        raise TimeoutError(f"Operation timed out after {timeout_seconds} seconds")
    
    # Check for exceptions
    if not exception_queue.empty():
        raise exception_queue.get()
    
    # Check for results
    if not result_queue.empty():
        return result_queue.get()
    
    raise TimeoutError("No result returned")

def generate_all_ps_combinations():
    
    problems = get_problems_with_names()
    
    # PHASE 2 
    #["PSO", "BBO", "CRO", "BRO", "AOA", "ABC"]
    #algorithms = ["CRO", "WOA", "HHO", "SSA", "SMA", "AOA", "SCA", "BRO", "PSO", "BBO", ]
    algorithms = ["CRO"] #[ "BBO", "ACO",  "BRO", "CRO", "PSO"] # "AOA", "SSO"]
    #algorithms = ["BRO", "CRO", "AOA", "SSO"]
    
    depths = [3, 4, 5]
    metrics_combinations = [
        "simplicity variance", 
        "simplicity variance estimated_atomicity",
    ]
   
    
    # Full parameters to match dataset scale used by notebooks
    sample_size = 10000
    ps_budget = 5000
    ps_population = 100
    avoid_ancestors = False
    
    # TIMEOUT SETTINGS
    MAX_TIME_PER_COMBINATION = 600  # Allow up to 10 minutes per combination on Windows
    
    # Setup output directory
    destination_folder = get_compare_own_data_folder()
    utils.make_directory(destination_folder)
    
    # Calculate total combinations
    total_combinations = (
        len(problems) * 
        len(algorithms) * 
        len(depths) * 
        len(metrics_combinations)
    )
    
    print("=" * 80)
    print("COMPLETE PS TREE GENERATION - WINDOWS SAFE MODE WITH TIMEOUTS")
    print("=" * 80)
    print(f"Problems: {list(problems.keys())}")
    print(f"Algorithms: {algorithms}")
    print(f"Depths: {depths}")
    print(f"Metrics: {metrics_combinations}")
    print(f"Sample size: {sample_size}")
    print(f"PS budget: {ps_budget}")
    print(f"PS population: {ps_population}")
    print(f"Timeout per combination: {MAX_TIME_PER_COMBINATION} seconds")
    print(f"Total combinations: {total_combinations}")
    print(f"Output folder: {destination_folder}")
    print("=" * 80)
    
    # Counters
    current_combination = 0
    successful_generations = 0
    failed_generations = 0
    timeout_failures = 0
    
    # Generate all combinations with timeout protection
    for problem_name, problem in problems.items():
        for algorithm in algorithms:
            for depth in depths:
                for metrics in metrics_combinations:
                    current_combination += 1
                    start_time = time.time()
                    
                    print(f"\n [{current_combination}/{total_combinations}] Processing:")
                    print(f"   Problem: {problem_name}")
                    print(f"   Algorithm: {algorithm}")
                    print(f"   Depth: {depth}")
                    print(f"   Metrics: {metrics}")
                    print(f"   Started at: {time.strftime('%H:%M:%S')}")
                    
                    # Tree settings
                    tree_settings = [{
                        "kind": "ps",
                        "ps_budget": ps_budget,
                        "ps_population": ps_population,
                        "depths": [depth],
                        "avoid_ancestors": avoid_ancestors,
                        "metrics": metrics
                    }]
                    
                    try:
                        """
                        # Generate with timeout protection using threading
                        datapoint = run_with_timeout(
                            get_datapoint_for_instance,
                            args=(),
                            kwargs={
                                'problem_name': problem_name,
                                'problem': problem,
                                'tree_settings_list': tree_settings,
                                'sample_size': sample_size,
                                'pRef_method': algorithm,
                                'crash_on_error': False,
                                'seed': random.randint(0, 2**32 - 1)
                            },
                            timeout_seconds=MAX_TIME_PER_COMBINATION
                        )
                        """

                        
                        datapoint = get_datapoint_for_instance(
                            problem_name=problem_name,
                            problem=problem,
                            tree_settings_list=tree_settings,
                            sample_size=sample_size,
                            pRef_method=algorithm,
                            crash_on_error=False,
                            seed=random.randint(0, 2**32 - 1)
                        )

                        
                        # Calculate runtime
                        runtime = time.time() - start_time
                        runtime = round(runtime, 3)

                        # ✅ Add runtime info directly inside JSON output
                        if isinstance(datapoint, dict):
                            datapoint["runtime_seconds"] = runtime
                            datapoint["problem"] = problem_name
                            datapoint["algorithm"] = algorithm
                            datapoint["depth"] = depth
                            datapoint["metrics_used"] = metrics
                        else:
                            datapoint = {
                                "data": datapoint,
                                "runtime_seconds": runtime,
                                "problem": problem_name,
                                "algorithm": algorithm,
                                "depth": depth,
                                "metrics_used": metrics
                            }

                        # Save result immediately
                        safe_metrics = metrics.replace(" ", "_")
                        timestamp = utils.get_formatted_timestamp()
                        filename = f"output_{problem_name}_{algorithm}_{depth}_{safe_metrics}_{timestamp}.json"
                        filepath = os.path.join(destination_folder, filename)
                        
                        with open(filepath, "w") as file:
                            json.dump([datapoint], file, indent=4)
                        
                        successful_generations += 1
                        print(f"    SUCCESS: {filename}")
                        print(f"    Runtime: {runtime:.1f} seconds")
                        
                        # Display metrics if available
                        if "results_by_tree" in datapoint and len(datapoint["results_by_tree"]) > 0:
                            results = datapoint["results_by_tree"][0].get("results", {})
                            if isinstance(results, dict) and "mae" in results:
                                mae = results.get("mae", "N/A")
                                mse = results.get("mse", "N/A") 
                                r2 = results.get("r2", "N/A")
                                print(f"      📊 MAE: {mae}, MSE: {mse}, R2: {r2}")
                    
                    except TimeoutError as te:
                        timeout_failures += 1
                        failed_generations += 1
                        runtime = time.time() - start_time
                        print(f"   ⏰ TIMEOUT: {runtime:.1f} seconds (max: {MAX_TIME_PER_COMBINATION})")
                        
                        # Save timeout error
                        safe_metrics = metrics.replace(" ", "_")
                        error_filename = f"TIMEOUT_{problem_name}_{algorithm}_{depth}_{safe_metrics}.json"
                        error_filepath = os.path.join(destination_folder, error_filename)
                        
                        error_info = {
                            "error": str(te),
                            "max_timeout": MAX_TIME_PER_COMBINATION,
                            "runtime": runtime,
                            "combination": {
                                "problem": problem_name,
                                "algorithm": algorithm,
                                "depth": depth,
                                "metrics": metrics
                            },
                            "tree_settings": tree_settings
                        }
                        
                        with open(error_filepath, "w") as file:
                            json.dump(error_info, file, indent=4)
                        
                        print(f"      💾 Timeout info saved: {error_filename}")
                    
                    except Exception as e:
                        failed_generations += 1
                        runtime = time.time() - start_time
                        print(f"  FAILED: {str(e)} (after {runtime:.1f} seconds)")
                        
                        # Save error information
                        safe_metrics = metrics.replace(" ", "_")
                        error_filename = f"ERROR_{problem_name}_{algorithm}_{depth}_{safe_metrics}.json"
                        error_filepath = os.path.join(destination_folder, error_filename)
                        
                        error_info = {
                            "error": str(e),
                            "runtime": runtime,
                            "combination": {
                                "problem": problem_name,
                                "algorithm": algorithm,
                                "depth": depth,
                                "metrics": metrics
                            },
                            "tree_settings": tree_settings
                        }
                        
                        with open(error_filepath, "w") as file:
                            json.dump(error_info, file, indent=4)
                        
                        print(f" Error saved: {error_filename}")
                    
                    # Progress update
                    if current_combination % 2 == 0:  # Every 2 combinations
                        print(f"\n PROGRESS UPDATE:")
                        print(f"   Completed: {current_combination}/{total_combinations}")
                        print(f"   Success rate: {(successful_generations/current_combination)*100:.1f}%")
                        print(f"   Timeouts: {timeout_failures}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE - FINAL SUMMARY")
    print("=" * 80)
    print(f"Total combinations processed: {current_combination}")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_generations}")
    print(f"Timeout failures: {timeout_failures}")
    if current_combination > 0:
        success_rate = (successful_generations / current_combination) * 100
        print(f"Success rate: {success_rate:.1f}%")
    print(f"Output directory: {destination_folder}")
    print("=" * 80)

def generate_reduced_test_set():
    """Quick test with minimal parameters"""
    print(" TESTING MODE - Reduced parameters for debugging")
    
    problems = get_problems_with_names()
    test_combinations = [
        ('SAT_S', 'BBO', 2, 'simplicity variance'),  # Start with one
    ]
    
    sample_size = 100   # Very small for testing
    ps_budget = 100     # Very small for testing
    ps_population = 20  # Very small for testing
    MAX_TIME_PER_TEST = 3000  # 1 minute timeout
    
    destination_folder = get_compare_own_data_folder()
    utils.make_directory(destination_folder)
    
    print(f"Testing {len(test_combinations)} combinations with timeout={MAX_TIME_PER_TEST}s")
    
    for i, (problem_name, algorithm, depth, metrics) in enumerate(test_combinations, 1):
        start_time = time.time()
        print(f"\n [{i}/{len(test_combinations)}] Testing: {problem_name}, {algorithm}, depth {depth}")
        
        tree_settings = [{
            "kind": "ps",
            "ps_budget": ps_budget,
            "ps_population": ps_population,
            "depths": [depth],
            "avoid_ancestors": False,
            "metrics": metrics
        }]
        
        try:
            datapoint = run_with_timeout(
                get_datapoint_for_instance,
                args=(),
                kwargs={
                    'problem_name': problem_name,
                    'problem': problems[problem_name],
                    'tree_settings_list': tree_settings,
                    'sample_size': sample_size,
                    'pRef_method': algorithm,
                    'crash_on_error': False,
                    'seed': 42
                },
                timeout_seconds=MAX_TIME_PER_TEST
            )
            
            runtime = time.time() - start_time
            
            filename = f"TEST_{problem_name}_{algorithm}_{depth}_{metrics.replace(' ', '_')}.json"
            filepath = os.path.join(destination_folder, filename)
            
            with open(filepath, "w") as file:
                json.dump([datapoint], file, indent=4)
            
            print(f"   ✅ SUCCESS: {filename} (runtime: {runtime:.1f}s)")
            
        except TimeoutError:
            runtime = time.time() - start_time
            print(f"    TIMEOUT after {runtime:.1f}s")
            
        except Exception as e:
            runtime = time.time() - start_time
            print(f"    FAILED: {e} (after {runtime:.1f}s)")

def generate_specific_missing():
    pass

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        
        if mode == "missing":
            print("Mode: Generate specific missing combinations")
            generate_specific_missing()
        elif mode == "test":
            print("Mode: Generate reduced test set")
            generate_reduced_test_set()
        elif mode == "all":
            print("Mode: Generate ALL combinations")
            generate_all_ps_combinations()
        else:
            print(f"Unknown mode: {mode}")
            print("Available modes: all, missing, test")
    else:
        print("Mode: Generate ALL combinations (default)")
        generate_all_ps_combinations()
