"""
Detailed step-by-step script for BBO with PS-SWA (simplicity variance estimated_atomicity)
on SAT_M problem with depth 5, printing all intermediate steps.
"""

import sys
from pathlib import Path
import os

# Add project root to path
current_dir = Path(os.getcwd())
# Try to find project root by looking for main.py
project_root = current_dir
while project_root != project_root.parent:
    if (project_root / "main.py").exists():
        break
    project_root = project_root.parent

project_root_str = str(project_root.resolve())
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# Import utils and verify it's the correct one
# Force reload if already imported to avoid conflicts with installed package
if 'utils' in sys.modules:
    utils_module = sys.modules['utils']
    utils_file = getattr(utils_module, '__file__', None)
    if utils_file and 'site-packages' in str(utils_file):
        del sys.modules['utils']
        # Also remove any submodules
        keys_to_remove = [k for k in sys.modules.keys() if k.startswith('utils.')]
        for k in keys_to_remove:
            del sys.modules[k]

import utils
# Verify utils has required functions
if not hasattr(utils, 'unzip'):
    utils_file = getattr(utils, '__file__', 'unknown location')
    raise ImportError(f"Wrong utils module imported! Missing 'unzip' function. Got: {utils_file}")
if not hasattr(utils, 'announce'):
    utils_file = getattr(utils, '__file__', 'unknown location')
    raise ImportError(f"Wrong utils module imported! Missing 'announce' function. Got: {utils_file}")

from BenchmarkProblems.SATProblem import SATProblem
from VarianceDecisionTree.PSDecisionTree import PSDecisionTree
from Explanation.PRefManager import PRefManager
from Core.PRef import PRef
from Core.PS import PS, contains
from Core.FullSolution import FullSolution
import numpy as np


class VerbosePSDecisionTree(PSDecisionTree):
    """Extended PSDecisionTree that prints detailed step-by-step information."""
    
    def __init__(self, maximum_depth, ps_budget, ps_search_population_size, 
                 problem, metrics_to_use, ancestor_splits=None, 
                 avoid_ancestors=False, depth_level=0, node_id='root'):
        self.depth_level = depth_level
        self.node_id = node_id
        super().__init__(
            maximum_depth=maximum_depth,
            ps_budget=ps_budget,
            ps_search_population_size=ps_search_population_size,
            problem=problem,
            metrics_to_use=metrics_to_use,
            ancestor_splits=ancestor_splits,
            avoid_ancestors=avoid_ancestors
        )
    
    def train_from_pRef(self, pRef: PRef, random_state: int = 42, verbose: bool = True) -> None:
        """Override to add detailed printing."""
        indent = "  " * self.depth_level
        
        print(f"\n{'='*80}")
        print(f"{indent}NODE: {self.node_id} (Depth: {self.depth_level})")
        print(f"{'='*80}")
        
        # Calculate and print node statistics
        try:
            fitness_array = pRef.fitness_array
            if len(fitness_array) == 0:
                print(f"{indent}⚠️  Empty pRef - cannot proceed")
                return

            pRef_variance = float(np.var(fitness_array))
            pRef_average = float(np.average(fitness_array))
            pRef_std = float(np.std(fitness_array))
            pRef_min = float(np.min(fitness_array))
            pRef_max = float(np.max(fitness_array))
            
            self.own_variance = pRef_variance
            self.own_average = pRef_average
            self.own_sd = pRef_std
            self.mean_error = float(np.average(np.abs(fitness_array - pRef_average)))
            
            print(f"\n{indent}📊 NODE STATISTICS:")
            print(f"{indent}   Sample Size: {pRef.sample_size}")
            print(f"{indent}   Average Fitness: {pRef_average:.4f}")
            print(f"{indent}   Variance: {pRef_variance:.4f}")
            print(f"{indent}   Standard Deviation: {pRef_std:.4f}")
            print(f"{indent}   Min Fitness: {pRef_min:.4f}")
            print(f"{indent}   Max Fitness: {pRef_max:.4f}")
            print(f"{indent}   Mean Absolute Error: {self.mean_error:.4f}")
            
            if (self.maximum_depth < 1) or (pRef_variance < 1e-05):
                print(f"\n{indent}🛑 STOPPING CRITERIA MET:")
                if self.maximum_depth < 1:
                    print(f"{indent}   - Maximum depth reached")
                if pRef_variance < 1e-05:
                    print(f"{indent}   - Variance too low ({pRef_variance:.6f} < 1e-05)")
                print(f"{indent}✅ This node will be a LEAF")
                return
                
        except Exception as e:
            print(f"{indent}❌ Error calculating statistics: {e}")
            self.own_variance = 0.0
            self.own_average = 0.0
            self.own_sd = 0.0
            self.mean_error = 0.0
            return

        # Get best solution
        best_solution = pRef.get_best_solution()
        print(f"\n{indent}🏆 BEST SOLUTION IN THIS NODE:")
        print(f"{indent}   Fitness: {best_solution.fitness:.4f}")
        print(f"{indent}   Solution: {best_solution.values}")
        
        # Get unexplained variables
        from GuestLecture.show_off_problems import get_unexplained_parts
        unexplained_vars = get_unexplained_parts(best_solution, self.ancestor_splits)
        print(f"\n{indent}🔍 UNEXPLAINED VARIABLES:")
        print(f"{indent}   Count: {np.sum(unexplained_vars)}/{len(unexplained_vars)}")
        print(f"{indent}   Mask: {unexplained_vars}")
        
        # Search for PS
        print(f"\n{indent}🔎 SEARCHING FOR SPLITTING PS:")
        print(f"{indent}   PS Budget: {self.ps_budget}")
        print(f"{indent}   Population Size: {self.ps_search_population_size}")
        print(f"{indent}   Metrics: {self.metrics_to_use}")
        print(f"{indent}   Ancestor Splits: {len(self.ancestor_splits)}")
        
        with utils.announce(f"Searching for a ps in a branch with {pRef.sample_size} datapoints", verbose):
            from VarianceDecisionTree.SimplePSSearchTask import find_ps_in_solution
            pss = find_ps_in_solution(pRef=pRef,
                                      ps_budget=self.ps_budget,
                                      culling_method="elbow",
                                      population_size=self.ps_search_population_size,
                                      to_explain=best_solution,
                                      unexplained_mask=unexplained_vars,
                                      proportion_unexplained_that_needs_used=0,
                                      proportion_used_that_should_be_unexplained=0.9 if self.avoid_ancestors else 0,
                                      problem=self.optimisation_problem,
                                      metrics=self.metrics_to_use,
                                      verbose=True)  # Enable verbose for PS search

        if not pss:
            print(f"{indent}❌ No PS found - this will be a leaf")
            return
            
        self.split_ps = pss[0]
        
        print(f"\n{indent}✅ CHOSEN SPLITTING PS:")
        print(f"{indent}   PS: {self.split_ps}")
        print(f"{indent}   Order (Fixed Variables): {self.split_ps.fixed_count()}")
        print(f"{indent}   Fixed Positions: {self.split_ps.get_fixed_variable_positions()}")
        print(f"{indent}   Values: {self.split_ps.values}")
        
        # Print metric scores if available
        if hasattr(self.split_ps, 'metric_scores') and self.split_ps.metric_scores is not None:
            print(f"\n{indent}📈 PS METRIC SCORES:")
            metric_names = self.metrics_to_use.split()
            for i, metric_name in enumerate(metric_names):
                if i < len(self.split_ps.metric_scores):
                    print(f"{indent}   {metric_name}: {self.split_ps.metric_scores[i]:.4f}")
        else:
            print(f"\n{indent}📈 PS METRIC SCORES: Not available (PS may not be EvaluatedPS)")
        
        # Split the pRef
        from VarianceDecisionTree.recursive_pRef_splitting import split_pRef_using_ps
        match_pRef, unmatch_pRef = split_pRef_using_ps(pRef, self.split_ps)
        
        print(f"\n{indent}📊 SPLIT RESULTS:")
        print(f"{indent}   Matching Branch Size: {match_pRef.sample_size}")
        print(f"{indent}   Non-Matching Branch Size: {unmatch_pRef.sample_size}")
        
        if match_pRef.sample_size > 0:
            match_avg = float(np.average(match_pRef.fitness_array))
            match_var = float(np.var(match_pRef.fitness_array))
            print(f"{indent}   Matching Branch - Avg: {match_avg:.4f}, Var: {match_var:.4f}")
        
        if unmatch_pRef.sample_size > 0:
            unmatch_avg = float(np.average(unmatch_pRef.fitness_array))
            unmatch_var = float(np.var(unmatch_pRef.fitness_array))
            print(f"{indent}   Non-Matching Branch - Avg: {unmatch_avg:.4f}, Var: {unmatch_var:.4f}")
        
        # Create child nodes
        print(f"\n{indent}🌳 CREATING CHILD NODES:")
        
        self.matching_branch = VerbosePSDecisionTree(
            maximum_depth=self.maximum_depth - 1,
            ps_budget=self.ps_budget,
            ps_search_population_size=self.ps_search_population_size,
            ancestor_splits=self.ancestor_splits + [self.split_ps],
            problem=self.optimisation_problem,
            metrics_to_use=self.metrics_to_use,
            avoid_ancestors=self.avoid_ancestors,
            depth_level=self.depth_level + 1,
            node_id=f"{self.node_id}.match"
        )
        
        self.unmatching_branch = VerbosePSDecisionTree(
            maximum_depth=self.maximum_depth - 1,
            ps_budget=self.ps_budget,
            ps_search_population_size=self.ps_search_population_size,
            ancestor_splits=self.ancestor_splits,
            problem=self.optimisation_problem,
            metrics_to_use=self.metrics_to_use,
            avoid_ancestors=self.avoid_ancestors,
            depth_level=self.depth_level + 1,
            node_id=f"{self.node_id}.unmatch"
        )
        
        # Recursively train children
        print(f"\n{indent}➡️  TRAINING MATCHING BRANCH:")
        self.matching_branch.train_from_pRef(match_pRef, random_state, verbose)
        
        print(f"\n{indent}➡️  TRAINING NON-MATCHING BRANCH:")
        self.unmatching_branch.train_from_pRef(unmatch_pRef, random_state, verbose)


def main():
    """Main function to run BBO with PS-SWA on SAT_M with depth 5."""
    
    print("="*80)
    print("BBO with PS-SWA on SAT_M Problem - Depth 5 - Detailed Step-by-Step")
    print("="*80)
    
    # Load SAT_M problem
    print("\n📁 LOADING SAT_M PROBLEM...")
    # Use project root to find resources directory
    resources_dir = os.path.join(project_root_str, "resources")
    sat_directory = os.path.join(resources_dir, "problem_definitions", "SAT")
    sat_m_file = os.path.join(sat_directory, "uf50-01.cnf")
    
    if not os.path.exists(sat_m_file):
        print(f"❌ Error: SAT_M file not found at {sat_m_file}")
        print("   Please ensure the file exists or update the path.")
        return
    
    problem = SATProblem.from_cnf_file(sat_m_file)
    print(f"✅ Loaded SAT_M problem:")
    print(f"   Variables: {problem.amount_of_variables}")
    print(f"   Clauses: {problem.amount_of_clauses}")
    print(f"   Solvable: {problem.solvable}")
    
    # Generate pRef using BBO
    print("\n" + "="*80)
    print("GENERATING PREF USING BBO ALGORITHM")
    print("="*80)
    
    with utils.announce("Generating pRef with BBO", verbose=True):
        pRef = PRefManager.generate_pRef(
            problem=problem,
            sample_size=10000,
            which_algorithm="BBO"
        )
    
    print(f"\n✅ Generated pRef:")
    print(f"   Sample Size: {pRef.sample_size}")
    print(f"   Average Fitness: {np.average(pRef.fitness_array):.4f}")
    print(f"   Best Fitness: {pRef.get_best_solution().fitness:.4f}")
    print(f"   Variance: {np.var(pRef.fitness_array):.4f}")
    
    # Create PSDecisionTree with PS-SWA metrics
    print("\n" + "="*80)
    print("CREATING PS DECISION TREE WITH PS-SWA METRICS")
    print("="*80)
    print("\n📋 TREE CONFIGURATION:")
    print("   Algorithm: BBO")
    print("   Metrics: simplicity variance estimated_atomicity (PS-SWA)")
    print("   Maximum Depth: 5")
    print("   PS Budget: 5000")
    print("   PS Search Population Size: 100")
    
    metrics = "simplicity variance estimated_atomicity"  # PS-SWA
    decision_tree = VerbosePSDecisionTree(
        maximum_depth=5,
        ps_budget=5000,
        ps_search_population_size=100,
        problem=problem,
        metrics_to_use=metrics,
        avoid_ancestors=False,
        depth_level=0,
        node_id="root"
    )
    
    # Set PS representation for pretty printing
    decision_tree.set_repr_ps(problem.repr_ps)
    
    # Train the tree
    print("\n" + "="*80)
    print("TRAINING DECISION TREE")
    print("="*80)
    
    decision_tree.train_from_pRef(pRef, verbose=True)
    
    # Print final tree structure
    print("\n" + "="*80)
    print("FINAL TREE STRUCTURE")
    print("="*80)
    print(decision_tree.repr_long())
    
    print("\n" + "="*80)
    print("✅ COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

