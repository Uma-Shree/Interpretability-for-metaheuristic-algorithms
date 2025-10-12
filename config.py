"""
Central configuration file for all directory paths and settings.
This eliminates hardcoded paths throughout the codebase.
"""
import os
import platform
from pathlib import Path

class PathConfig:
    """Centralized path configuration for the entire project."""
    
    def __init__(self):
        # Detect current project root automatically
        self.project_root = self._get_project_root()
        
        # Core directories
        self.resources_dir = os.path.join(self.project_root, "resources")
        self.variance_tree_materials_dir = os.path.join(self.resources_dir, "variance_tree_materials")
        
        # Experiment output directories
        self.compare_own_data_dir = os.path.join(self.variance_tree_materials_dir, "compare_own_data")
        self.dt_data_dir = os.path.join(self.variance_tree_materials_dir, "dt_data")
        
        # User study directories (if needed)
        self.user_study_dir = os.path.join(self.project_root, "UserStudy")
        self.instances_dir = os.path.join(self.user_study_dir, "Instances")
        
        # Explanations directories (if needed)
        self.explanations_dir = os.path.join(self.resources_dir, "explanations")
    
    def _get_project_root(self):
        """Automatically detect project root directory."""
        # Start from current file location and go up until we find main.py
        current_path = Path(__file__).parent.absolute()
        
        # Look for main.py to identify project root
        while current_path.parent != current_path:  # Not at filesystem root
            if (current_path / "main.py").exists():
                return str(current_path)
            current_path = current_path.parent
        
        # Fallback: assume current directory is project root
        return str(Path(__file__).parent.absolute())
    
    def get_experiment_folder(self, experiment_type="compare_own_data", timestamp=None):
        """Generate timestamped experiment folder path."""
        import utils  # Import here to avoid circular imports
        
        if timestamp is None:
            timestamp = utils.get_formatted_timestamp()
        
        if experiment_type == "compare_own_data":
            return os.path.join(self.compare_own_data_dir, f"compare_own_data_{timestamp}")
        elif experiment_type == "dt_data":
            return os.path.join(self.dt_data_dir, f"dt_data_{timestamp}")
        elif experiment_type == "iai_run":
            return os.path.join(self.compare_own_data_dir, f"iai_run_3_{timestamp}")
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
    
    def get_data_subfolder(self, experiment_folder):
        """Get the data subfolder within an experiment folder."""
        return os.path.join(experiment_folder, "data")
    
    def get_results_csv_path(self, experiment_folder):
        """Get the results.csv path for an experiment folder."""
        return os.path.join(experiment_folder, "results.csv")
    
    def get_tree_data_csv_path(self, experiment_folder):
        """Get the tree_data.csv path for an experiment folder."""
        return os.path.join(experiment_folder, "tree_data.csv")
    
    @staticmethod
    def is_windows():
        """Check if running on Windows."""
        return platform.system() == "Windows"

# Global instance for easy access
paths = PathConfig()

# Convenience functions for backward compatibility
def get_compare_own_data_folder(timestamp=None):
    """Get a timestamped compare_own_data folder path."""
    return paths.get_experiment_folder("compare_own_data", timestamp)

def get_dt_data_folder(timestamp=None):
    """Get a timestamped dt_data folder path."""
    return paths.get_experiment_folder("dt_data", timestamp)

def get_iai_run_folder(timestamp=None):
    """Get a timestamped IAI run folder path."""
    return paths.get_experiment_folder("iai_run", timestamp)
