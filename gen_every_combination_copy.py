#!/usr/bin/env python3
"""
Complete PS Tree Generator - Generate all combinations of PS trees
Follows the successful pattern from working missing_file code
Includes all numpy.float64 safety fixes
"""

import os
import json
import random
import utils
from config import get_compare_own_data_folder
from VarianceDecisionTree.compare_prediction_powers import get_problems_with_names, get_datapoint_for_instance


def generate_all_ps_combinations():
    """Generate ALL possible PS tree combinations using working pattern"""
    
    # Define all possible combinations
    problems = get_problems_with_names()
    algorithms = ["uniform", "GA", "SA", "Tabu", "PSO"]
    #algorithms = [ "PSO", "BBO", "CRO", "BRO", "AOA", "ABC"] #"uniform", "GA", "SA", "Tabu", "PSO",

    depths = [2, 3, 4, 5]
    
    # All metrics combinations
    metrics_combinations = [
        "simplicity variance", 
        "simplicity variance estimated_atomicity",
    ]
    
    # Configuration parameters
    sample_size = 10000
    ps_budget = 5000
    ps_population = 100
    avoid_ancestors = False
    
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
    print("COMPLETE PS TREE GENERATION - ALL COMBINATIONS")
    print("=" * 80)
    print(f"Problems: {list(problems.keys())}")
    print(f"Algorithms: {algorithms}")
    print(f"Depths: {depths}")
    print(f"Metrics: {metrics_combinations}")
    print(f"Sample size: {sample_size}")
    print(f"PS budget: {ps_budget}")
    print(f"PS population: {ps_population}")
    print(f"Total combinations: {total_combinations}")
    print(f"Output folder: {destination_folder}")
    print("=" * 80)
    
    # Counters
    current_combination = 0
    successful_generations = 0
    failed_generations = 0
    
    # Generate all combinations - INDIVIDUAL PROCESSING like working code
    for problem_name, problem in problems.items():
        for algorithm in algorithms:
            for depth in depths:
                for metrics in metrics_combinations:
                    current_combination += 1
                    
                    print(f"\n🔄 [{current_combination}/{total_combinations}] Processing:")
                    print(f"   Problem: {problem_name}")
                    print(f"   Algorithm: {algorithm}")
                    print(f"   Depth: {depth}")
                    print(f"   Metrics: {metrics}")
                    
                    # SAME TREE SETTINGS STRUCTURE AS WORKING CODE
                    tree_settings = [{
                        "kind": "ps",
                        "ps_budget": ps_budget,
                        "ps_population": ps_population,
                        "depths": [depth],  # CRITICAL: Single depth in array
                        "avoid_ancestors": avoid_ancestors,
                        "metrics": metrics
                    }]
                    
                    try:
                        # SAME GENERATION CALL AS WORKING CODE
                        datapoint = get_datapoint_for_instance(
                            problem_name=problem_name,
                            problem=problem,
                            tree_settings_list=tree_settings,
                            sample_size=sample_size,
                            pRef_method=algorithm,
                            crash_on_error=False,
                            seed=random.randint(0, 2**32 - 1)  # Random seed
                        )

                        actual_algorithm = algorithm  # Default assumption
                        if hasattr(datapoint, 'get') and datapoint.get('metadata', {}).get('actual_algorithm_used'):
                            actual_algorithm = datapoint['metadata']['actual_algorithm_used']
                            print(f"🔍 ALGORITHM TRACKING: {algorithm} → {actual_algorithm}")
                        
                        # IMMEDIATE FILE SAVING - same as working code
                        safe_metrics = metrics.replace(" ", "_")
                        timestamp = utils.get_formatted_timestamp()
                        filename = f"output_{problem_name}_{algorithm}_{depth}_{safe_metrics}_{timestamp}.json"
                        filepath = os.path.join(destination_folder, filename)
                        
                        # Save individual file immediately
                        with open(filepath, "w") as file:
                            json.dump([datapoint], file, indent=4)
                        
                        successful_generations += 1
                        print(f"   ✅ SUCCESS: {filename}")
                        
                        # Display metrics if available
                        if "results_by_tree" in datapoint and len(datapoint["results_by_tree"]) > 0:
                            results = datapoint["results_by_tree"][0].get("results", {})
                            if isinstance(results, dict) and "mae" in results:
                                mae = results.get("mae", "N/A")
                                mse = results.get("mse", "N/A") 
                                r2 = results.get("r2", "N/A")
                                print(f"      📊 MAE: {mae}, MSE: {mse}, R2: {r2}")
                    
                    except Exception as e:
                        failed_generations += 1
                        print(f"   ❌ FAILED: {str(e)}")
                        
                        # Save error information
                        safe_metrics = metrics.replace(" ", "_")
                        error_filename = f"ERROR_{problem_name}_{algorithm}_{depth}_{safe_metrics}.json"
                        error_filepath = os.path.join(destination_folder, error_filename)
                        
                        error_info = {
                            "error": str(e),
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
                        
                        print(f"      💾 Error saved: {error_filename}")
    
    # Final summary
    print("\n" + "=" * 80)
    print("GENERATION COMPLETE - FINAL SUMMARY")
    print("=" * 80)
    print(f"Total combinations processed: {current_combination}")
    print(f"Successful generations: {successful_generations}")
    print(f"Failed generations: {failed_generations}")
    if current_combination > 0:
        success_rate = (successful_generations / current_combination) * 100
        print(f"Success rate: {success_rate:.1f}%")
    print(f"Output directory: {destination_folder}")
    print("=" * 80)

"""
def generate_specific_missing():
    
    # Your complete missing list - expand this as needed
    missing = [
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
        # Add more combinations from your complete missing list
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
        ('BT', 'uniform', 3, 'simplicity variance'),
        ('BT', 'uniform', 3, 'simplicity variance estimated_atomicity'),
        ('BT', 'uniform', 4, 'simplicity variance'),
        ('BT', 'uniform', 4, 'simplicity variance estimated_atomicity'),
        ('BT', 'uniform', 5, 'simplicity variance'),
        ('BT', 'uniform', 5, 'simplicity variance estimated_atomicity'),
        ('SAT_S', 'GA', 3, 'simplicity variance'),
        ('SAT_S', 'GA', 3, 'simplicity variance estimated_atomicity'),
        ('SAT_S', 'GA', 4, 'simplicity variance'),
        ('SAT_S', 'GA', 4, 'simplicity variance estimated_atomicity'),
        ('SAT_S', 'GA', 5, 'simplicity variance'),
        ('SAT_S', 'GA', 5, 'simplicity variance estimated_atomicity'),
        ('SAT_M', 'GA', 3, 'simplicity variance'),
        ('SAT_M', 'GA', 3, 'simplicity variance estimated_atomicity'),
        ('SAT_M', 'GA', 4, 'simplicity variance'),
        ('SAT_M', 'GA', 4, 'simplicity variance estimated_atomicity'),
        ('SAT_M', 'GA', 5, 'simplicity variance'),
        ('SAT_M', 'GA', 5, 'simplicity variance estimated_atomicity'),
    ]
    
    # Same parameters as working code
    sample_size = 10000
    ps_budget = 5000
    ps_population = 100
    avoid_ancestors = False
    
    destination_folder = get_compare_own_data_folder()
    utils.make_directory(destination_folder)
    
    problems = get_problems_with_names()
    
    print("=" * 60)
    print("GENERATING SPECIFIC MISSING COMBINATIONS")
    print("=" * 60)
    print(f"Missing combinations: {len(missing)}")
    print(f"Output folder: {destination_folder}")
    print("=" * 60)
    
    successful = 0
    failed = 0
    
    for i, (problem_name, algorithm, depth, metrics) in enumerate(missing, 1):
        print(f"\n🔄 [{i}/{len(missing)}] Generating:")
        print(f"   {problem_name} - {algorithm} - depth {depth} - {metrics}")
        
        # EXACT SAME STRUCTURE AS WORKING CODE
        tree_settings = [{
            "kind": "ps",
            "ps_budget": ps_budget,
            "ps_population": ps_population,
            "depths": [depth],  # Single depth in array
            "avoid_ancestors": avoid_ancestors,
            "metrics": metrics
        }]
        
        try:
            # EXACT SAME CALL AS WORKING CODE
            datapoint = get_datapoint_for_instance(
                problem_name=problem_name,
                problem=problems[problem_name],
                tree_settings_list=tree_settings,
                sample_size=sample_size,
                pRef_method=algorithm,
                crash_on_error=False,
                seed=random.randint(0, 2**32 - 1)
            )
            
            # EXACT SAME FILE SAVING AS WORKING CODE
            safe_metrics = metrics.replace(" ", "_")
            json_file_name = os.path.join(
                destination_folder, 
                f"output_{problem_name}_{algorithm}_{depth}_{safe_metrics}.json"
            )
            
            with open(json_file_name, "w") as file:
                json.dump([datapoint], file, indent=4)
            
            successful += 1
            print(f"   ✅ SUCCESS: {os.path.basename(json_file_name)}")
            
        except Exception as e:
            failed += 1
            print(f"   ❌ FAILED: {str(e)}")
    
    print(f"\n" + "=" * 60)
    print(f"SUMMARY: {successful} successful, {failed} failed")
    print("=" * 60)


def generate_reduced_test_set():
    
    print("Generating reduced test set for debugging...")
    
    # Smaller test combinations
    test_combinations = [
        ('SAT_S', 'uniform', 2, 'variance'),
        ('SAT_S', 'GA', 2, 'variance'), 
        ('SAT_S', 'PSO', 2, 'variance'),
    ]
    
    sample_size = 1000  # Smaller for faster testing
    ps_budget = 1000    # Smaller for faster testing
    ps_population = 50  # Smaller for faster testing
    
    destination_folder = get_compare_own_data_folder()
    utils.make_directory(destination_folder)
    problems = get_problems_with_names()
    
    for i, (problem_name, algorithm, depth, metrics) in enumerate(test_combinations, 1):
        print(f"[{i}/{len(test_combinations)}] Testing: {problem_name}, {algorithm}, {depth}, {metrics}")
        
        tree_settings = [{
            "kind": "ps",
            "ps_budget": ps_budget,
            "ps_population": ps_population,
            "depths": [depth],
            "avoid_ancestors": False,
            "metrics": metrics
        }]
        
        try:
            datapoint = get_datapoint_for_instance(
                problem_name=problem_name,
                problem=problems[problem_name],
                tree_settings_list=tree_settings,
                sample_size=sample_size,
                pRef_method=algorithm,
                crash_on_error=False,
                seed=random.randint(0, 2**32 - 1)
            )
            
            filename = f"output_{problem_name}_{algorithm}_{depth}_{metrics}.json"
            filepath = os.path.join(destination_folder, filename)
            
            with open(filepath, "w") as file:
                json.dump([datapoint], file, indent=4)
            
            print(f"✅ SUCCESS: {filename}")
            
        except Exception as e:
            print(f"❌ FAILED: {e}")
"""
def generate_specific_missing():
    pass
def generate_reduced_test_set():
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
