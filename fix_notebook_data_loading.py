#!/usr/bin/env python3
"""
Script to fix the data loading issue in the Jupyter notebook.
This script will:
1. Check both data folders for valid data
2. Generate proper CSV files
3. Provide the correct path to use in the notebook
"""

import json
import os
import pandas as pd
from typing import List, Dict, Any

def is_valid_data_file(file_name: str) -> bool:
    return file_name.endswith("json") or file_name.endswith("txt")

def json_to_entries(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert JSON data to flat entries for DataFrame"""
    
    def item_to_list_of_entries(item) -> List[Dict[str, Any]]:
        # Skip if 'results_by_tree' is not present
        if "results_by_tree" not in item:
            return []

        surrounding_information = {
            "problem": item.get("problem_name", "Unknown"),
            "pRef_method": item.get("pRef_method", "Unknown"),
            "sample_size": item.get("sample_size", 0),
            "seed": item.get("seed", 0)
        }

        entries = []
        for tree_result in item["results_by_tree"]:
            entry = surrounding_information.copy()
            
            # Add tree-specific information
            entry.update({
                "kind": tree_result.get("kind", "Unknown"),
                "depth": tree_result.get("depth", 0),
                "ps_budget": tree_result.get("ps_budget", None),
                "ps_population": tree_result.get("ps_population", None),
                "avoid_ancestors": tree_result.get("avoid_ancestors", False),
                "metrics": tree_result.get("metrics", ""),
                "cp": tree_result.get("cp", None)  # For IAI trees
            })
            
            # Add results if present
            if "results" in tree_result:
                results = tree_result["results"]
                entry.update({
                    "mse": results.get("mse", None),
                    "mae": results.get("mae", None),
                    "r_sq": results.get("r_sq", None),
                    "evs": results.get("evs", None)
                })
            
            entries.append(entry)
        
        return entries

    return [entry for item in data for entry in item_to_list_of_entries(item)]

def convert_accuracy_data_to_df(input_directory: str, output_filename: str):
    """Convert JSON data to CSV format"""
    
    all_dicts = []
    
    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        file_path = os.path.join(input_directory, filename)

        # Check if the file is a JSON file
        if not os.path.isfile(file_path):
            continue

        if not is_valid_data_file(file_path):
            continue

        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                entries = json_to_entries(data)
                all_dicts.extend(entries)
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue

    # Convert list of dictionaries to DataFrame
    if len(all_dicts) > 0:
        df = pd.DataFrame(all_dicts)
    else:
        # Create empty DataFrame with expected columns
        df = pd.DataFrame(columns=['problem', 'pRef_method', 'sample_size', 'seed', 'kind', 'depth', 
                                  'ps_budget', 'ps_population', 'avoid_ancestors', 'metrics', 'cp',
                                  'mse', 'mae', 'r_sq', 'evs'])

    # Write the DataFrame to a CSV file
    df.to_csv(output_filename, index=False)
    return df

def prettify_kind_column(df: pd.DataFrame):
    """Add prettified kind column for better display"""
    if 'kind' in df.columns:
        df['kind_pretty'] = df['kind'].map({
            'naive': 'Traditional',
            'ps': 'PS Tree',
            'iai': 'IAI Tree'
        }).fillna(df['kind'])

def find_valid_data_folder():
    """Find a data folder with valid results"""
    
    base_path = r"A:\metahuristic_benchmark\PS-descriptors\resources\variance_tree_materials\compare_own_data"
    
    # Get all subdirectories
    if not os.path.exists(base_path):
        print(f"Base path does not exist: {base_path}")
        return None
    
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    print(f"Found {len(subdirs)} data folders:")
    for subdir in subdirs:
        print(f"  - {subdir}")
    
    # Check each folder for valid data
    for subdir in subdirs:
        folder_path = os.path.join(base_path, subdir)
        results_csv = os.path.join(folder_path, "results.csv")
        
        print(f"\nChecking folder: {subdir}")
        
        # Try to generate CSV
        df = convert_accuracy_data_to_df(folder_path, results_csv)
        
        if len(df) > 0:
            print(f"  ✓ Found {len(df)} data entries")
            print(f"  ✓ Generated CSV: {results_csv}")
            return folder_path
        else:
            print(f"  ✗ No valid data found")
    
    return None

def main():
    """Main function to fix the notebook data loading issue"""
    
    print("=== Fixing Notebook Data Loading Issue ===\n")
    
    # Find a valid data folder
    valid_folder = find_valid_data_folder()
    
    if valid_folder is None:
        print("\n❌ ERROR: No valid data folders found!")
        print("\nPossible solutions:")
        print("1. Run main.py successfully to generate proper data")
        print("2. Check that the JSON files contain 'results_by_tree' field")
        print("3. Verify the data folder paths are correct")
        return
    
    print(f"\n✅ SUCCESS: Found valid data in {valid_folder}")
    
    # Generate both CSV files
    results_csv = os.path.join(valid_folder, "results.csv")
    tree_data_csv = os.path.join(valid_folder, "tree_data.csv")
    
    # Load and display sample data
    df = pd.read_csv(results_csv)
    prettify_kind_column(df)
    
    print(f"\n📊 Data Summary:")
    print(f"  - Total entries: {len(df)}")
    print(f"  - Problems: {df['problem'].unique()}")
    print(f"  - Tree types: {df['kind'].unique()}")
    print(f"  - Search methods: {df['pRef_method'].unique()}")
    print(f"  - Depths: {sorted(df['depth'].unique())}")
    
    print(f"\n📁 Files generated:")
    print(f"  - {results_csv}")
    print(f"  - {tree_data_csv}")
    
    print(f"\n🔧 To fix your notebook:")
    print(f"1. Change the run_location variable to:")
    print(f'   run_location = r"{valid_folder}"')
    print(f"2. Re-run the cell that loads the CSV data")
    
    print(f"\n📋 Sample data:")
    print(df.head())

if __name__ == "__main__":
    main()
