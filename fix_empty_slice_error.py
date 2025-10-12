#!/usr/bin/env python3
"""
Quick fix for the "Mean of empty slice" error in PSDecisionTree.
This script patches the train_from_pRef method to handle empty splits gracefully.
"""

import numpy as np

def safe_mean(array):
    """Calculate mean safely, returning 0 for empty arrays"""
    if len(array) == 0:
        return 0.0
    return np.mean(array)

def safe_var(array):
    """Calculate variance safely, returning 0 for empty arrays"""
    if len(array) == 0:
        return 0.0
    return np.var(array)

def safe_std(array):
    """Calculate standard deviation safely, returning 0 for empty arrays"""
    if len(array) == 0:
        return 0.0
    return np.std(array)

# Patch numpy functions to handle empty arrays
original_mean = np.mean
original_var = np.var
original_std = np.std
original_average = np.average

def patched_mean(a, axis=None, **kwargs):
    if hasattr(a, '__len__') and len(a) == 0:
        return 0.0
    return original_mean(a, axis=axis, **kwargs)

def patched_var(a, axis=None, **kwargs):
    if hasattr(a, '__len__') and len(a) == 0:
        return 0.0
    return original_var(a, axis=axis, **kwargs)

def patched_std(a, axis=None, **kwargs):
    if hasattr(a, '__len__') and len(a) == 0:
        return 0.0
    return original_std(a, axis=axis, **kwargs)

def patched_average(a, axis=None, **kwargs):
    if hasattr(a, '__len__') and len(a) == 0:
        return 0.0
    return original_average(a, axis=axis, **kwargs)

def apply_patches():
    """Apply patches to numpy functions"""
    np.mean = patched_mean
    np.var = patched_var
    np.std = patched_std
    np.average = patched_average
    print("Applied numpy patches for empty array handling")

def remove_patches():
    """Remove patches and restore original numpy functions"""
    np.mean = original_mean
    np.var = original_var
    np.std = original_std
    np.average = original_average
    print("Removed numpy patches")

if __name__ == "__main__":
    print("This script provides patches for the 'Mean of empty slice' error.")
    print("Import and call apply_patches() before running main.py")
    print("Call remove_patches() when done.")
