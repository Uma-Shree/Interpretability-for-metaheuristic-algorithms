# Add debug code to test PSO separately
try:
    from mealpy.swarm_based.PSO import BaselineFluctuationPSO
    # Test PSO with your specific search space
    print("PSO imported successfully")
except Exception as e:
    print(f"PSO import/setup error: {e}")
